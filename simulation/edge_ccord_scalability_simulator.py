from utils.sim_utils import (
    Exponential, GetArrival, Min, reset_arrival_temp,
    GetServiceEdgeE, GetServiceEdgeC, GetServiceCloud,
    GetServiceCoordP1P2, GetServiceCoordP3P4, GetLambda
)
from utils.simulation_stats import SimulationStats
import utils.constants as cs
from libraries.rngs import getSeed, selectStream, random as rng_random
import math

def edge_coord_scalability_simulation(stop, forced_lambda=None, slot_index=None):
    """
    Simulatore unificato Edge + Coordinator con anti-stallo e statistiche consistenti.
    - Arrivi: Poisson non omogeneo (GetArrival) oppure λ forzato per slot.
    - Edge: pool scalabile (1..EDGE_SERVERS_MAX) per job "E" (nuovi) e "C" (ritorno dal Cloud).
    - Cloud: ∞-server (serializzato da t.completion_cloud).
    - Coordinator: pool scalabile (1..COORD_SERVERS_MAX) per P1..P4 (P3/P4 prioritarie).
    """
    seed = getSeed()
    reset_arrival_temp()

    # Clamp limiti server
    cs.EDGE_SERVERS = max(1, int(getattr(cs, "EDGE_SERVERS", 1)))
    cs.EDGE_SERVERS_MAX = max(cs.EDGE_SERVERS, int(getattr(cs, "EDGE_SERVERS_MAX", 6)))
    cs.COORD_EDGE_SERVERS = max(1, int(getattr(cs, "COORD_EDGE_SERVERS", 1)))
    cs.COORD_SERVERS_MAX = max(cs.COORD_EDGE_SERVERS, int(getattr(cs, "COORD_SERVERS_MAX", 6)))

    stats = SimulationStats()
    stats.reset(cs.START)

    # Primo arrivo
    if forced_lambda is not None:
        lam = max(1e-12, float(forced_lambda))
        selectStream(0)
        stats.t.arrival = stats.t.current + Exponential(1.0 / lam)
    else:
        stats.t.arrival = GetArrival(stats.t.current)

    # Contatori
    stats.index_edge = stats.index_cloud = stats.index_coord = 0
    stats.number_edge = stats.number_cloud = stats.number_coord = 0
    stats.job_arrived = 0
    stats.count_E = 0
    stats.count_E_P1 = stats.count_E_P2 = stats.count_E_P3 = stats.count_E_P4 = 0

    # Autoscaling
    decision_interval = float(getattr(cs, "SCALING_WINDOW", 1000.0))
    last_checkpoint_edge = stats.t.current
    last_checkpoint_coord = stats.t.current

    edge_scal_trace = []
    coord_scal_trace = []

    # Stato server Edge
    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy       = [False]       * cs.EDGE_SERVERS_MAX
    edge_server_jobtype    = [None]        * cs.EDGE_SERVERS_MAX  # "E" | "C"

    # Stato server Coordinator
    coord_completion_times = [cs.INFINITY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False]       * cs.COORD_SERVERS_MAX
    coord_server_ptype     = [None]        * cs.COORD_SERVERS_MAX  # "P1"|"P2"|"P3"|"P4"

    # Accumulatori capacità (server·tempo) per utilizzo medio su tutto l’intervallo
    cap_time_edge = 0.0
    cap_time_coord = 0.0

    # Helper assegnazioni
    def edge_assign_if_possible(sidx):
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E" o "C"
            service = GetServiceEdgeE() if job == "E" else GetServiceEdgeC()
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job
            stats.area_edge.service += service
            if job == "E":
                stats.area_E.service += service
            return True
        return False

    def coord_assign_if_possible(sidx):
        if not coord_server_busy[sidx]:
            # priorità: coda high (P3/P4) poi low (P1/P2)
            if stats.queue_coord_high:
                cls = stats.queue_coord_high.pop(0)  # "P3"|"P4"
                service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                cls = stats.queue_coord_low.pop(0)   # "P1"|"P2"
                service = GetServiceCoordP1P2()
            else:
                return False
            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
            coord_server_ptype[sidx] = cls
            stats.area_coord.service += service
            return True
        return False

    def kick_assign_all():
        # Edge
        for i in range(cs.EDGE_SERVERS):
            if not stats.queue_edge:
                break
            edge_assign_if_possible(i)
        # Coordinator
        for i in range(cs.COORD_EDGE_SERVERS):
            if not stats.queue_coord_high and not stats.queue_coord_low:
                break
            coord_assign_if_possible(i)

    # Main loop
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        next_completion_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else cs.INFINITY
        next_completion_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else cs.INFINITY

        stats.t.next = Min(stats.t.arrival, next_completion_edge, stats.t.completion_cloud, next_completion_coord)

        # Anti-stallo
        if (not math.isfinite(stats.t.next)) or (stats.t.next <= stats.t.current):
            lam = max(1e-12, float(forced_lambda) if forced_lambda is not None else GetLambda(stats.t.current))
            selectStream(0)
            stats.t.arrival = stats.t.current + Exponential(1.0 / lam)
            stats.t.next = Min(stats.t.arrival, next_completion_edge, stats.t.completion_cloud, next_completion_coord)
            if (not math.isfinite(stats.t.next)) or (stats.t.next <= stats.t.current):
                break

        # Avanza il tempo + aree (N(t)) + capacità server·tempo
        delta = stats.t.next - stats.t.current
        if stats.number_edge  > 0: stats.area_edge.node  += delta * stats.number_edge
        if stats.number_cloud > 0: stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0: stats.area_coord.node += delta * stats.number_coord
        # capacità (server attivi * tempo)
        if cs.EDGE_SERVERS > 0:
            cap_time_edge  += delta * cs.EDGE_SERVERS
        if cs.COORD_EDGE_SERVERS > 0:
            cap_time_coord += delta * cs.COORD_EDGE_SERVERS

        stats.t.current = stats.t.next

        # Autoscaling EDGE
        if stats.t.current - last_checkpoint_edge >= decision_interval:
            obs_time = max(1e-12, (stats.t.current - last_checkpoint_edge) * max(1, cs.EDGE_SERVERS))
            utilization = stats.area_edge.service / obs_time
            edge_scal_trace.append((stats.t.current, cs.EDGE_SERVERS, utilization))
            if utilization > cs.UTILIZATION_UPPER and cs.EDGE_SERVERS < cs.EDGE_SERVERS_MAX:
                cs.EDGE_SERVERS += 1
            elif utilization < cs.UTILIZATION_LOWER and cs.EDGE_SERVERS > 1:
                cs.EDGE_SERVERS -= 1
                highest_busy = max([i for i, b in enumerate(edge_server_busy) if b], default=-1)
                cs.EDGE_SERVERS = max(1, max(cs.EDGE_SERVERS, highest_busy + 1))
            stats.area_edge.service = 0.0
            last_checkpoint_edge = stats.t.current
            kick_assign_all()

        # Autoscaling COORD
        if stats.t.current - last_checkpoint_coord >= decision_interval:
            obs_time_c = max(1e-12, (stats.t.current - last_checkpoint_coord) * max(1, cs.COORD_EDGE_SERVERS))
            utilization_c = stats.area_coord.service / obs_time_c

            # 👇 Debug print qui
            print(f"[DEBUG][t={stats.t.current:.1f}] Utilizzazione COORD finestra = {utilization_c:.3f} "
                  f"(server={cs.COORD_EDGE_SERVERS}, obs_time_c={obs_time_c:.3f}, "
                  f"service_area={stats.area_coord.service:.3f})")

            coord_scal_trace.append((stats.t.current, cs.COORD_EDGE_SERVERS, utilization_c))
            if utilization_c > cs.UTILIZATION_UPPER and cs.COORD_EDGE_SERVERS < cs.COORD_SERVERS_MAX:
                cs.COORD_EDGE_SERVERS += 1
            elif utilization_c < cs.UTILIZATION_LOWER and cs.COORD_EDGE_SERVERS > 1:
                cs.COORD_EDGE_SERVERS -= 1
                highest_busy_c = max([i for i, b in enumerate(coord_server_busy) if b], default=-1)
                cs.COORD_EDGE_SERVERS = max(1, max(cs.COORD_EDGE_SERVERS, highest_busy_c + 1))
            stats.area_coord.service = 0.0
            last_checkpoint_coord = stats.t.current
            kick_assign_all()

        # Evento: ARRIVO all'Edge (job "E")
        if stats.t.current == stats.t.arrival and stats.t.current < stop:
            stats.job_arrived += 1
            stats.queue_edge.append("E")
            stats.count_E += 1
            # FIX: contatore numero in sistema all'Edge
            stats.number_edge += 1
            # prossimo arrivo
            if forced_lambda is not None:
                lam = max(1e-12, float(forced_lambda))
                selectStream(0)
                stats.t.arrival = stats.t.current + Exponential(1.0 / lam)
            else:
                stats.t.arrival = GetArrival(stats.t.current)
            kick_assign_all()
            continue

        # Evento: COMPLETAMENTO EDGE
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                completed_type = edge_server_jobtype[i]
                # libera il server
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1

                if completed_type == "E":
                    # routing: Cloud oppure Coordinator
                    selectStream(3)
                    r = rng_random()
                    if r < cs.P_C:
                        # Cloud
                        stats.number_cloud += 1
                        if stats.number_cloud == 1:
                            service = GetServiceCloud()
                            stats.t.completion_cloud = stats.t.current + service
                            stats.area_cloud.service += service
                            stats.area_C.service += service
                    else:
                        # Coordinator: classi P1..P4
                        stats.number_coord += 1
                        coord_r = (r - cs.P_C) / max(1e-12, (1.0 - cs.P_C))
                        if coord_r < cs.P1_PROB:
                            stats.queue_coord_low.append("P1"); stats.count_E_P1 += 1
                        elif coord_r < cs.P1_PROB + cs.P2_PROB:
                            stats.queue_coord_low.append("P2"); stats.count_E_P2 += 1
                        elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                            stats.queue_coord_high.append("P3"); stats.count_E_P3 += 1
                        else:
                            stats.queue_coord_high.append("P4"); stats.count_E_P4 += 1
                        # prova assegnazione coord
                        for j in range(cs.COORD_EDGE_SERVERS):
                            if coord_assign_if_possible(j):
                                break
                # Se era "C" (ritorno dal Cloud) il job termina qui

                kick_assign_all()
                break  # gestito un completamento

        # Evento: COMPLETAMENTO CLOUD (∞-server serializzato)
        if stats.t.current == stats.t.completion_cloud:
            stats.index_cloud += 1
            stats.number_cloud -= 1
            if stats.number_cloud > 0:
                service = GetServiceCloud()
                stats.t.completion_cloud = stats.t.current + service
                stats.area_cloud.service += service
                stats.area_C.service += service
            else:
                stats.t.completion_cloud = cs.INFINITY

            # ritorno all'Edge come job "C"
            stats.queue_edge.append("C")
            # FIX: incrementa il numero in sistema all'Edge per il ritorno dal Cloud
            stats.number_edge += 1
            kick_assign_all()

        # Evento: COMPLETAMENTO COORD
        for i in range(cs.COORD_EDGE_SERVERS):
            if stats.t.current == coord_completion_times[i]:
                coord_server_busy[i] = False
                coord_completion_times[i] = cs.INFINITY
                coord_server_ptype[i] = None

                stats.index_coord += 1
                stats.number_coord -= 1

                # tenta di assegnare nuovo lavoro
                if not coord_assign_if_possible(i):
                    kick_assign_all()
                break

    # Risultati
    sim_time = max(1e-12, stats.t.current - cs.START)

    # Utilizzazioni su tutto l’intervallo (tempo occupato totale / capacità totale)
    edge_util = (stats.area_edge.service / max(1e-12, cap_time_edge)) if cap_time_edge > 0 else 0.0
    coord_util = (stats.area_coord.service / max(1e-12, cap_time_coord)) if cap_time_coord > 0 else 0.0
    cloud_avg_busy = stats.area_cloud.service / sim_time  # ∞-server: media server occupati

    results = {
        "seed": seed,
        "lambda": (forced_lambda if forced_lambda is not None else GetLambda(stats.t.current)),
        "slot": slot_index,

        # Edge
        "edge_server_number": cs.EDGE_SERVERS,
        "edge_avg_wait": (stats.area_edge.node / max(1, stats.index_edge)) if stats.index_edge > 0 else 0.0,
        "edge_avg_delay": (stats.area_edge.node / max(1, stats.job_arrived)) if stats.job_arrived > 0 else 0.0,
        "edge_L": stats.area_edge.node / sim_time,
        "edge_Lq": None,
        "edge_service_time_mean": (stats.area_edge.service / max(1, stats.index_edge)) if stats.index_edge > 0 else 0.0,
        "edge_utilization": edge_util,
        "edge_throughput": stats.index_edge / sim_time,

        'edge_E_avg_delay': (stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0,
        'edge_E_avg_response': ((stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0) \
                               + cs.EDGE_SERVICE_E,

        # Cloud
        "cloud_avg_wait": (stats.area_cloud.node / max(1, stats.index_cloud)) if stats.index_cloud > 0 else 0.0,
        "cloud_avg_delay": (stats.area_cloud.node / max(1, stats.job_arrived)) if stats.job_arrived > 0 else 0.0,
        "cloud_L": stats.area_cloud.node / sim_time,
        "cloud_Lq": None,
        "cloud_service_time_mean": (stats.area_cloud.service / max(1, stats.index_cloud)) if stats.index_cloud > 0 else 0.0,
        "cloud_avg_busy_servers": cloud_avg_busy,
        "cloud_throughput": stats.index_cloud / sim_time,

        # Coordinator
        "coord_server_number": cs.COORD_EDGE_SERVERS,
        "coord_avg_wait": (stats.area_coord.node / max(1, stats.index_coord)) if stats.index_coord > 0 else 0.0,
        "coord_avg_delay": (stats.area_coord.node / max(1, stats.job_arrived)) if stats.job_arrived > 0 else 0.0,
        "coord_L": stats.area_coord.node / sim_time,
        "coord_Lq": None,
        "coord_service_time_mean": (stats.area_coord.service / max(1, stats.index_coord)) if stats.index_coord > 0 else 0.0,
        "coord_utilization": coord_util,
        "coord_throughput": stats.index_coord / sim_time,

        # Tracce autoscaling
        "edge_scal_trace": edge_scal_trace,
        "coord_scal_trace": coord_scal_trace,
    }
    return results
