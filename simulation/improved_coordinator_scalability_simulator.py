from utils.sim_utils import *
from utils.improved_simulation_stats import SimulationStats_improved
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def coordinator_scalability_simulation_improved(stop, forced_lambda=None, slot_index=None, fixed_edge_servers=1):
    """
    Edge con N server PARALLELI FISSI (= fixed_edge_servers) che servono SOLO job 'E'.
    Coordinator con server PARALLELI SCALABILI (min 1, max cs.COORD_SERVERS_MAX) per P1..P4.
    Cloud ∞-server.
    **Dopo il Cloud i job vanno in una coda 'feedback' FIFO (1 server, Exp(1.0s)) e poi escono.**
    """
    seed = getSeed()
    reset_arrival_temp()

    # Fissa i server Edge (almeno 1)
    fixed_edge_servers = max(1, min(int(round(fixed_edge_servers)), cs.EDGE_SERVERS_MAX))
    cs.EDGE_SERVERS = fixed_edge_servers

    # Coordinator parte con almeno 1 server
    cs.COORD_EDGE_SERVERS = max(1, getattr(cs, "COORD_EDGE_SERVERS", 1))

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

    # Finestra per utilizzo Coordinator
    coord_util_data = {i: {"active_time": 0.0, "area_service": 0.0} for i in range(1, cs.COORD_SERVERS_MAX + 1)}
    scalability_trace = []

    # Stato Edge (FISSI)
    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy       = [False]       * cs.EDGE_SERVERS_MAX
    edge_server_jobtype    = [None]        * cs.EDGE_SERVERS_MAX  # "E" (non più "C")

    # Stato Coordinator (SCALABILE)
    coord_completion_times = [cs.INFINITY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False]       * cs.COORD_SERVERS_MAX
    coord_server_ptype     = [None]        * cs.COORD_SERVERS_MAX  # "P1"/"P2"/"P3"/"P4"

    def _service_feedback():
        # Exp(mean = 0.5 * 2 = 1.0 s)
        return Exponential(0.5 * 2)

    def edge_assign_if_possible(sidx):
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E"
            service = GetServiceEdgeE()
            stats.area_E.service += service
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job
            stats.area_edge.service += service
            return True
        return False

    def coord_assign_if_possible(sidx):
        if not coord_server_busy[sidx]:
            # priorità alta a P3/P4
            ptype = None
            if stats.queue_coord_high:
                ptype = stats.queue_coord_high.pop(0)
                service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                ptype = stats.queue_coord_low.pop(0)
                service = GetServiceCoordP1P2()
            else:
                return False

            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
            coord_server_ptype[sidx] = ptype
            stats.area_coord.service += service
            return True
        return False

    while stats.t.arrival < stop or (stats.number_edge + stats.number_cloud + stats.number_coord + stats.number_feedback) > 0:
        next_completion_edge = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else cs.INFINITY
        next_completion_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else cs.INFINITY
        stats.t.next = Min(
            stats.t.arrival,
            next_completion_edge,
            stats.t.completion_cloud,
            next_completion_coord,
            stats.t.completion_feedback  # nuovo: feedback
        )

        delta = stats.t.next - stats.t.current

        # Aree numero in nodo
        if stats.number_edge > 0:
            stats.area_edge.node += delta * stats.number_edge
        if stats.number_cloud > 0:
            stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0:
            stats.area_coord.node += delta * stats.number_coord
        if stats.number_E > 0:
            stats.area_E.node += delta * stats.number_E

        # Accumulo finestra utilizzo Coordinator
        for i in range(cs.COORD_EDGE_SERVERS):
            coord_util_data[i + 1]["active_time"] += delta
            if coord_server_busy[i]:
                coord_util_data[i + 1]["area_service"] += delta

        stats.t.current = stats.t.next

        # Checkpoint e decisione scalabilità Coordinator
        if stats.t.current - last_checkpoint >= interval:
            total_service = sum(coord_util_data[i + 1]["area_service"] for i in range(cs.COORD_EDGE_SERVERS))
            total_time = sum(coord_util_data[i + 1]["active_time"] for i in range(cs.COORD_EDGE_SERVERS))
            utilization = (total_service / total_time) if total_time > 0 else 0.0
            scalability_trace.append((stats.t.current, cs.COORD_EDGE_SERVERS, utilization))

            if utilization > cs.UTILIZATION_UPPER and cs.COORD_EDGE_SERVERS < cs.COORD_SERVERS_MAX:
                cs.COORD_EDGE_SERVERS += 1
                coord_completion_times[cs.COORD_EDGE_SERVERS - 1] = cs.INFINITY
                coord_server_busy[cs.COORD_EDGE_SERVERS - 1] = False
                coord_server_ptype[cs.COORD_EDGE_SERVERS - 1] = None
            elif utilization < cs.UTILIZATION_LOWER and cs.COORD_EDGE_SERVERS > 1:
                target = cs.COORD_EDGE_SERVERS - 1
                if not coord_server_busy[target]:
                    coord_completion_times[target] = cs.INFINITY
                    coord_server_ptype[target] = None
                    cs.COORD_EDGE_SERVERS = max(1, target + 0)

            # reset finestra
            for k in range(1, cs.COORD_SERVERS_MAX + 1):
                coord_util_data[k]["active_time"] = 0.0
                coord_util_data[k]["area_service"] = 0.0

            last_checkpoint = stats.t.current

        # Arrivo all'Edge
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E += 1
            stats.queue_edge.append("E")
            stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
            if stats.t.arrival > stop:
                stats.t.arrival = cs.INFINITY
            for i in range(cs.EDGE_SERVERS):
                if edge_assign_if_possible(i):
                    break

        # Completamento Edge
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                completed_type = edge_server_jobtype[i]  # "E"
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1

                # Solo 'E': routing Cloud/Coordinator
                stats.number_E -= 1
                selectStream(3)
                rand_val = rng_random()
                if rand_val < cs.P_C:
                    # Vai al Cloud
                    stats.number_cloud += 1
                    if stats.number_cloud == 1:
                        service = GetServiceCloud()
                        stats.t.completion_cloud = stats.t.current + service
                        stats.area_cloud.service += service
                        stats.area_C.service += service
                else:
                    # Vai al Coordinator (P1..P4)
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
                    for j in range(cs.COORD_EDGE_SERVERS):
                        if coord_assign_if_possible(j):
                            break

                edge_assign_if_possible(i)
                break

        # Completamento Cloud
        if stats.t.current == stats.t.completion_cloud:
            stats.index_cloud += 1
            stats.number_cloud -= 1
            if stats.number_cloud > 0:
                service = GetServiceCloud()
                stats.t.completion_cloud = stats.t.current + service
                stats.area_cloud.service += service
            else:
                stats.t.completion_cloud = cs.INFINITY

            # **NUOVO**: instrada alla coda feedback (singolo server)
            stats.number_feedback += 1
            stats.queue_feedback.append("FB")
            if stats.number_feedback == 1:
                service = _service_feedback()
                stats.t.completion_feedback = stats.t.current + service

        # Completamento Coordinator
        for j in range(cs.COORD_EDGE_SERVERS):
            if stats.t.current == coord_completion_times[j]:
                coord_server_busy[j] = False
                coord_completion_times[j] = cs.INFINITY
                finished_ptype = coord_server_ptype[j]
                coord_server_ptype[j] = None

                stats.index_coord += 1
                stats.number_coord -= 1
                stats.count_E += 1
                if finished_ptype == "P1":
                    stats.count_E_P1 += 1
                elif finished_ptype == "P2":
                    stats.count_E_P2 += 1
                elif finished_ptype == "P3":
                    stats.count_E_P3 += 1
                elif finished_ptype == "P4":
                    stats.count_E_P4 += 1

                if stats.number_coord > 0:
                    coord_assign_if_possible(j)
                break

        # Completamento Feedback (uscita)
        if stats.t.current == stats.t.completion_feedback:
            if stats.queue_feedback:
                stats.queue_feedback.pop(0)
            stats.number_feedback -= 1
            if stats.number_feedback > 0:
                service = _service_feedback()
                stats.t.completion_feedback = stats.t.current + service
            else:
                stats.t.completion_feedback = cs.INFINITY

    # Calcola aree di coda
    stats.calculate_area_queue()

    # Utilizzo per ciascun conteggio server Coordinator (sull’intera sim)
    per_server_utilization = {}
    for s in range(1, cs.COORD_SERVERS_MAX + 1):
        data = coord_util_data[s]
        per_server_utilization[s] = (data["area_service"] / data["active_time"]) if data["active_time"] > 0 else None

    T = stats.t.current if stats.t.current > 0 else 1.0

    edge_W  = stats.area_edge.node  / stats.index_edge  if stats.index_edge  > 0 else 0.0
    edge_Wq = stats.area_edge.queue / stats.index_edge  if stats.index_edge  > 0 else 0.0
    cloud_W  = stats.area_cloud.node  / stats.index_cloud if stats.index_cloud > 0 else 0.0
    cloud_Wq = stats.area_cloud.queue / stats.index_cloud if stats.index_cloud > 0 else 0.0
    coord_W  = stats.area_coord.node  / stats.index_coord if stats.index_coord > 0 else 0.0
    coord_Wq = stats.area_coord.queue / stats.index_coord if stats.index_coord > 0 else 0.0

    results = {
        'seed': seed,
        'lambda': forced_lambda,
        'slot': slot_index,

        # Edge (fisso)
        'edge_server_number': fixed_edge_servers,
        'edge_avg_wait': edge_W,
        'edge_avg_delay': edge_Wq,
        'edge_L': stats.area_edge.node / T,
        'edge_Lq': stats.area_edge.queue / T,
        'edge_service_time_mean': (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_avg_busy_servers': stats.area_edge.service / T,
        'edge_throughput': (stats.index_edge / T),

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

        # Coordinator (scalabile)
        'coord_server_number': max(1, cs.COORD_EDGE_SERVERS),
        'coord_avg_wait': coord_W,
        'coord_avg_delay': coord_Wq,
        'coord_L': stats.area_coord.node / T,
        'coord_Lq': stats.area_coord.queue / T,
        'coord_service_time_mean': (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0,
        'coord_utilization': stats.area_coord.service / T,
        'coord_throughput': (stats.index_coord / T),

        'server_utilization_by_count': per_server_utilization,
        'scalability_trace': scalability_trace
    }
    return results, stats
