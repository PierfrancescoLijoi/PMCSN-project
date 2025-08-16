from utils.sim_utils import *
from utils.improved_simulation_stats import SimulationStats_improved
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def edge_scalability_simulation_improved(stop, forced_lambda=None, slot_index=None):
    """
    Edge SCALABILE (1..EDGE_SERVERS_MAX) che serve SOLO job 'E'.
    Dopo il Cloud, i job NON tornano all'Edge: vanno nella coda 'feedback' FIFO
    a singolo server con servizio Exp(mean=0.5*2=1.0s) e poi escono.
    Coordinator e Cloud restano come nel modello attuale.
    """
    seed = getSeed()
    reset_arrival_temp()

    # Almeno 1 server SEMPRE attivo
    cs.EDGE_SERVERS = max(1, getattr(cs, "EDGE_SERVERS", 1))

    stats = SimulationStats_improved()
    stats.reset(cs.START)
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)

    # --- Stato feedback (nuovo stadio post-Cloud) ---
    if not hasattr(stats, 'queue_feedback'):
        stats.queue_feedback = []
    if not hasattr(stats, 'number_feedback'):
        stats.number_feedback = 0
    if not hasattr(stats.t, 'completion_feedback'):
        stats.t.completion_feedback = cs.INFINITY

    interval = 1000.0
    last_checkpoint = stats.t.current

    # Finestre di misura utilizzo per ciascun conteggio server (1..MAX)
    server_util_data = {i: {"active_time": 0.0, "area_service": 0.0} for i in range(1, cs.EDGE_SERVERS_MAX + 1)}
    scalability_trace = []

    # Stato dei server Edge
    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy       = [False]       * cs.EDGE_SERVERS_MAX
    edge_server_jobtype    = [None]        * cs.EDGE_SERVERS_MAX  # "E" (non più "C")

    def _service_feedback():
        # Exp(mean = 0.5 * 2 = 1.0 s)
        return Exponential(0.5 * 2)

    def assign_job_to_server(sidx):
        """Prova ad assegnare un job dal fronte coda al server sidx (se libero)."""
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # nel nuovo modello è sempre "E"
            service = GetServiceEdgeE()
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job
            stats.area_edge.service += service
            stats.area_E.service += service
            return True
        return False

    while stats.t.arrival < stop or (stats.number_edge + stats.number_cloud + stats.number_coord + stats.number_feedback) > 0:
        # Prossimo completamento tra i server attivi
        next_completion_edge = min(edge_completion_times[:cs.EDGE_SERVERS])
        stats.t.next = Min(
            stats.t.arrival,
            next_completion_edge,
            stats.t.completion_cloud,
            stats.t.completion_coord,
            stats.t.completion_feedback  # nuovo: feedback single-server
        )

        delta = stats.t.next - stats.t.current

        # Aree numero in nodo (per Little integrali)
        if stats.number_edge > 0:
            stats.area_edge.node += delta * stats.number_edge
        if stats.number_cloud > 0:
            stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0:
            stats.area_coord.node += delta * stats.number_coord
        if stats.number_E > 0:
            stats.area_E.node += delta * stats.number_E
        # (la coda feedback non ha metriche dedicate)

        # Accumulo finestra di utilizzo per i server attivi (1..EDGE_SERVERS)
        for i in range(cs.EDGE_SERVERS):
            server_util_data[i + 1]["active_time"] += delta
            if edge_server_busy[i]:
                server_util_data[i + 1]["area_service"] += delta

        stats.t.current = stats.t.next

        # Checkpoint finestra utilizzo
        if stats.t.current - last_checkpoint >= interval:
            total_service = sum(server_util_data[i + 1]["area_service"] for i in range(cs.EDGE_SERVERS))
            total_time    = sum(server_util_data[i + 1]["active_time"]  for i in range(cs.EDGE_SERVERS))
            utilization   = (total_service / total_time) if total_time > 0 else 0.0
            scalability_trace.append((stats.t.current, cs.EDGE_SERVERS, utilization))

            # Decisione scaling
            if utilization > cs.UTILIZATION_UPPER and cs.EDGE_SERVERS < cs.EDGE_SERVERS_MAX:
                cs.EDGE_SERVERS += 1
                edge_completion_times[cs.EDGE_SERVERS - 1] = cs.INFINITY
                edge_server_busy[cs.EDGE_SERVERS - 1] = False
                edge_server_jobtype[cs.EDGE_SERVERS - 1] = None
            elif utilization < cs.UTILIZATION_LOWER and cs.EDGE_SERVERS > 1:
                target = cs.EDGE_SERVERS - 1
                if not edge_server_busy[target]:
                    edge_completion_times[target] = cs.INFINITY
                    edge_server_jobtype[target] = None
                    cs.EDGE_SERVERS = max(1, target + 0)

            # Reset finestra per prossimi 1000s
            for k in range(1, cs.EDGE_SERVERS_MAX + 1):
                server_util_data[k]["active_time"] = 0.0
                server_util_data[k]["area_service"] = 0.0
            last_checkpoint = stats.t.current

        # Eventi: Arrivo
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E += 1
            stats.queue_edge.append("E")
            stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
            if stats.t.arrival > stop:
                stats.t.arrival = cs.INFINITY

            # Prova ad assegnare il job in arrivo ad un server libero
            for i in range(cs.EDGE_SERVERS):
                if assign_job_to_server(i):
                    break

        # Eventi: Completamento Edge (gestiamo al massimo un completamento)
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                completed_type = edge_server_jobtype[i]  # "E"

                # Libera il server i
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1

                # Routing del job COMPLETATO (solo 'E'):
                stats.number_E -= 1
                selectStream(3)
                rand_val = rng_random()
                if rand_val < cs.P_C:
                    # Va al Cloud
                    stats.number_cloud += 1
                    if stats.number_cloud == 1:
                        service = GetServiceCloud()
                        stats.t.completion_cloud = stats.t.current + service
                        stats.area_cloud.service += service
                        stats.area_C.service += service
                else:
                    # Va al Coordinator (classi P1..P4)
                    stats.number_coord += 1
                    coord_rand = (rand_val - cs.P_C) / (1.0 - cs.P_C)
                    if coord_rand < cs.P1_PROB:
                        stats.queue_coord_low.append("P1")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB:
                        stats.queue_coord_low.append("P2")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                        stats.queue_coord_high.append("P3")
                    else:
                        stats.queue_coord_high.append("P4")

                    if stats.number_coord == 1:
                        service = GetServiceCoordP3P4() if stats.queue_coord_high else GetServiceCoordP1P2()
                        stats.t.completion_coord = stats.t.current + service
                        stats.area_coord.service += service

                # Assegna un nuovo job (se presente) allo stesso server i
                assign_job_to_server(i)
                break  # gestito un completamento Edge

        # Eventi: Completamento Cloud
        if stats.t.current == stats.t.completion_cloud:
            stats.index_cloud += 1
            stats.number_cloud -= 1
            if stats.number_cloud > 0:
                service = GetServiceCloud()
                stats.t.completion_cloud = stats.t.current + service
                stats.area_cloud.service += service
            else:
                stats.t.completion_cloud = cs.INFINITY

            # **NUOVO**: niente ritorno all'Edge; va nella coda feedback
            stats.number_feedback += 1
            stats.queue_feedback.append("FB")
            if stats.number_feedback == 1:
                service = _service_feedback()
                stats.t.completion_feedback = stats.t.current + service

        # Eventi: Completamento Coordinator
        if stats.t.current == stats.t.completion_coord:
            stats.index_coord += 1
            stats.number_coord -= 1

            if stats.queue_coord_high:
                stats.queue_coord_high.pop(0)
                stats.count_E += 1
            elif stats.queue_coord_low:
                stats.queue_coord_low.pop(0)
                stats.count_E += 1

            if stats.number_coord > 0:
                service = GetServiceCoordP3P4() if stats.queue_coord_high else GetServiceCoordP1P2()
                stats.t.completion_coord = stats.t.current + service
                stats.area_coord.service += service
            else:
                stats.t.completion_coord = cs.INFINITY

        # Eventi: Completamento Feedback (uscita sistema)
        if stats.t.current == stats.t.completion_feedback:
            if stats.queue_feedback:
                stats.queue_feedback.pop(0)
            stats.number_feedback -= 1
            if stats.number_feedback > 0:
                service = _service_feedback()
                stats.t.completion_feedback = stats.t.current + service
            else:
                stats.t.completion_feedback = cs.INFINITY

    # Calcolo aree coda a fine simulazione
    stats.calculate_area_queue()

    # Utilizzo per ciascun conteggio server (sull’intera simulazione)
    per_server_utilization = {}
    for s in range(1, cs.EDGE_SERVERS_MAX + 1):
        data = server_util_data[s]
        per_server_utilization[s] = (data["area_service"] / data["active_time"]) if data["active_time"] > 0 else None

    T = stats.t.current if stats.t.current > 0 else 1.0

    edge_W = stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0.0
    edge_Wq = stats.area_edge.queue / stats.index_edge if stats.index_edge > 0 else 0.0
    cloud_W = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
    cloud_Wq = stats.area_cloud.queue / stats.index_cloud if stats.index_cloud > 0 else 0.0
    coord_W = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
    coord_Wq = stats.area_coord.queue / stats.index_coord if stats.index_coord > 0 else 0.0

    results = {
        'seed': seed,
        'lambda': forced_lambda,
        'slot': slot_index,

        # Edge
        'edge_avg_wait': edge_W,
        'edge_avg_delay': edge_Wq,
        'edge_L': stats.area_edge.node / T,
        'edge_Lq': stats.area_edge.queue / T,
        'edge_server_service': (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_server_utilization': stats.area_edge.service / T,
        'edge_weight_utilization': (stats.area_edge.node / (max(1, cs.EDGE_SERVERS) * T)),
        'edge_server_number': max(1, cs.EDGE_SERVERS),

        'edge_E_avg_delay': (stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0,
        'edge_E_avg_response': ((stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0) \
                               + cs.EDGE_SERVICE_E,

        # Cloud
        'cloud_avg_wait': cloud_W,
        'cloud_avg_delay': cloud_Wq,
        'cloud_L': stats.area_cloud.node / T,
        'cloud_Lq': stats.area_cloud.queue / T,
        'cloud_service_time_mean': (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0,
        'cloud_avg_busy_servers': stats.area_cloud.service / T,
        'cloud_throughput': (stats.index_cloud / T),

        # Coordinator
        'coord_avg_wait': coord_W,
        'coord_avg_delay': coord_Wq,
        'coord_L': stats.area_coord.node / T,
        'coord_Lq': stats.area_coord.queue / T,
        'coord_service_time_mean': (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0,
        'coord_utilization': stats.area_coord.service / T,
        'coord_throughput': stats.index_coord / T,

        # extra
        'server_utilization_by_count': per_server_utilization,
        'scalability_trace': scalability_trace
    }
    return results, stats
