import heapq
import math

import utils.constants as cs
from libraries.rngs import getSeed, selectStream, random as rng_random
from utils.sim_utils import (
    GetArrival, Min, reset_arrival_temp,
    GetServiceEdgeE, GetServiceEdgeC, GetServiceCloud,
    GetServiceCoordP1P2, GetServiceCoordP3P4
)
from utils.simulation_stats import SimulationStats


def edge_coord_scalability_simulation(stop, forced_lambda=None, slot_index=None):
    """
    Simulatore unificato Edge + Coordinator con anti-stallo e statistiche consistenti.
    - Arrivi: Poisson non omogeneo su 24h (λ(t) per fasce in constants) oppure λ forzato.
    - Edge: pool scalabile (1..EDGE_SERVERS_MAX) per job "E" (nuovi) e "C" (ritorno dal Cloud).
    - Cloud: ∞-server (min-heap dei completion times).
    - Coordinator: pool scalabile (1..COORD_SERVERS_MAX) per P1..P4 (P3/P4 prioritarie).
    """
    # === helper: λ(t) modulare sulle 24h, coprendo tutte le fasce definite ===
    def day_lambda(t):
        if forced_lambda is not None:
            return max(1e-12, float(forced_lambda))
        # t modulo 24h
        tt = float(t) % 86400.0
        # gestisci fasce con eventuale wrap oltre la mezzanotte
        for start, end, lam in cs.LAMBDA_SLOTS:
            s = float(start); e = float(end); l = float(lam)
            if e > 86400.0:
                # spezza in [s, 86400) ∪ [0, e-86400)
                if (s <= tt < 86400.0) or (0.0 <= tt < (e - 86400.0)):
                    return max(1e-12, l)
            else:
                if s <= tt < e:
                    return max(1e-12, l)
        # eventuali “buchi” non coperti dalle fasce: fallback
        return max(1e-12, float(getattr(cs, "LAMBDA", 1.0)))  # fallback prudente

    # === seed corrente (solo per logging/CSV) ===
    seed = getSeed()

    # === reset arrivi e stato ===
    reset_arrival_temp()  # usa stream 0 come in sim_utils.GetArrival
    # Forza orizzonte a 24h quando non si sta testando un λ fisso
    if forced_lambda is None:
        stop = cs.START + 86400.0  # 24h

    # Clamp limiti server
    cs.EDGE_SERVERS      = max(1, int(getattr(cs, "EDGE_SERVERS", 1)))
    cs.EDGE_SERVERS_MAX  = max(cs.EDGE_SERVERS, int(getattr(cs, "EDGE_SERVERS_MAX", 6)))
    cs.COORD_EDGE_SERVERS = max(1, int(getattr(cs, "COORD_EDGE_SERVERS", 1)))
    cs.COORD_SERVERS_MAX  = max(cs.COORD_EDGE_SERVERS, int(getattr(cs, "COORD_SERVERS_MAX", 6)))

    # === stato statistico ===
    stats = SimulationStats()
    stats.reset(cs.START)
    stats.cloud_heap = []                 # ∞-server: min-heap completamenti Cloud
    stats.t.completion_cloud = cs.INFINITY

    # Primo arrivo secondo λ(t) su 24h (o λ forzato)
    stats.t.arrival = GetArrival(stats.t.current, day_lambda(stats.t.current))

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

    # Accumulatori capacità (server·tempo) per utilizzo medio
    cap_time_edge = 0.0
    cap_time_coord = 0.0

    # Busy-time totali (su tutto l’orizzonte)
    total_edge_busy = 0.0
    total_coord_busy = 0.0
    total_edge_busy_E = 0.0
    total_edge_busy_C = 0.0

    # === helper assegnazioni ===
    def edge_assign_if_possible(sidx):
        nonlocal total_edge_busy, total_edge_busy_E, total_edge_busy_C
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E" o "C"

            if job == "E":
                service = GetServiceEdgeE()   # stream 1 in sim_utils
                stats.area_E.service += service
                total_edge_busy_E += service
            else:
                service = GetServiceEdgeC()   # stream 4 in sim_utils
                stats.area_C.service += service
                total_edge_busy_C += service

            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job

            stats.area_edge.service += service
            total_edge_busy += service
            return True
        return False

    def coord_assign_if_possible(sidx):
        nonlocal total_coord_busy
        if not coord_server_busy[sidx]:
            if stats.queue_coord_high:
                cls = stats.queue_coord_high.pop(0)  # "P3"/"P4"
                service = GetServiceCoordP3P4()      # stream 6
            elif stats.queue_coord_low:
                cls = stats.queue_coord_low.pop(0)   # "P1"/"P2"
                service = GetServiceCoordP1P2()      # stream 5
            else:
                return False
            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
            coord_server_ptype[sidx] = cls
            stats.area_coord.service += service
            total_coord_busy += service
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
            if not (stats.queue_coord_high or stats.queue_coord_low):
                break
            coord_assign_if_possible(i)

    # === Loop principale ===
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        next_completion_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else cs.INFINITY
        next_completion_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else cs.INFINITY
        next_cloud = stats.cloud_heap[0] if stats.cloud_heap else cs.INFINITY  # ∞-server

        stats.t.next = Min(stats.t.arrival, next_completion_edge, next_cloud, next_completion_coord)

        # Anti-stallo: rigenera un arrivo coerente con λ(t) se servisse
        if (not math.isfinite(stats.t.next)) or (stats.t.next <= stats.t.current):
            stats.t.arrival = GetArrival(stats.t.current, day_lambda(stats.t.current))
            next_cloud = stats.cloud_heap[0] if stats.cloud_heap else cs.INFINITY
            stats.t.next = Min(stats.t.arrival, next_completion_edge, next_cloud, next_completion_coord)
            if (not math.isfinite(stats.t.next)) or (stats.t.next <= stats.t.current):
                break

        # Avanza tempo + aree N(t) + capacità server·tempo
        delta = stats.t.next - stats.t.current
        if delta < 0:
            delta = 0.0

        if stats.number_edge  > 0: stats.area_edge.node  += delta * stats.number_edge
        if stats.number_cloud > 0: stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0: stats.area_coord.node += delta * stats.number_coord

        if stats.number_E > 0: stats.area_E.node += delta * stats.number_E
        if stats.number_C > 0: stats.area_C.node += delta * stats.number_C

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

        # === Evento: ARRIVO esterno (job "E") ===
        if stats.t.current == stats.t.arrival and stats.t.current < stop:
            stats.job_arrived += 1
            stats.queue_edge.append("E")
            stats.count_E += 1
            stats.number_edge += 1
            stats.number_E += 1

            # Prossimo arrivo coerente con λ(t)
            stats.t.arrival = GetArrival(stats.t.current, day_lambda(stats.t.current))
            kick_assign_all()
            continue

        # === Evento: COMPLETAMENTO EDGE ===
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                completed_type = edge_server_jobtype[i]

                # libera server i
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1

                if completed_type == "E":
                    # routing verso Cloud (P_C) o Coordinator
                    stats.index_edge_E += 1
                    stats.number_E -= 1
                    selectStream(3)  # routing
                    r = rng_random()
                    if r < cs.P_C:
                        # → Cloud (∞-server): programma completamento individuale (heap)
                        stats.number_cloud += 1
                        service = GetServiceCloud()  # stream 2
                        comp = stats.t.current + service
                        heapq.heappush(stats.cloud_heap, comp)
                        stats.area_cloud.service += service
                        # aggiorna “prossimo completamento Cloud”
                    else:
                        # → Coordinator, classi P1..P4 (split con condizionata su 1-P_C)
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
                        # prova ad assegnare subito
                        for j in range(cs.COORD_EDGE_SERVERS):
                            if coord_assign_if_possible(j):
                                break

                else:
                    # completamento di un job "C" all'Edge
                    stats.index_edge_C += 1
                    stats.number_C -= 1
                    # (stats.count_C conteggia i completamenti C all'Edge se vuoi usarlo)

                kick_assign_all()
                break  # ha gestito un completamento

        # === Evento: COMPLETAMENTO CLOUD (∞-server) ===
        next_cloud = stats.cloud_heap[0] if stats.cloud_heap else cs.INFINITY
        if stats.t.current == next_cloud:
            # consuma tutti i completamenti coincidenti (tolleranza numerica)
            while stats.cloud_heap and stats.cloud_heap[0] <= stats.t.current + 1e-12:
                heapq.heappop(stats.cloud_heap)
                stats.index_cloud += 1
                stats.number_cloud -= 1
                # ritorno all'Edge come job "C"
                stats.queue_edge.append("C")
                stats.number_edge += 1
                stats.number_C += 1
                # se Edge era idle, parte subito
                if not any(edge_server_busy[:cs.EDGE_SERVERS]):
                    # prova a schedulare sul primo server libero
                    for j in range(cs.EDGE_SERVERS):
                        if not edge_server_busy[j]:
                            service = GetServiceEdgeC()  # stream 4
                            edge_completion_times[j] = stats.t.current + service
                            edge_server_busy[j] = True
                            edge_server_jobtype[j] = "C"
                            stats.area_edge.service += service
                            stats.area_C.service += service
                            break

            kick_assign_all()

        # === Evento: COMPLETAMENTO COORD ===
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

    # Fine simulazione: calcolo metriche
    stats.calculate_area_queue()
    sim_time = max(1e-12, stats.t.current - cs.START)

    # --- Edge (totali) ---
    edge_W = (stats.area_edge.node / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_Wq = (stats.area_edge.queue / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_L = stats.area_edge.node / sim_time
    edge_Lq = stats.area_edge.queue / sim_time
    edge_S = (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_X = stats.index_edge / sim_time
    edge_busy_avg = stats.area_edge.service / sim_time
    edge_util = (stats.area_edge.service / (cap_time_edge)) if cap_time_edge > 0 else 0.0  # ∈ [0,1]

    # --- Edge per classe ---
    edge_E_W  = (stats.area_E.node / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq = (stats.area_E.queue / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_L  = stats.area_E.node / sim_time
    edge_E_Lq = stats.area_E.queue / sim_time
    edge_E_S  = (stats.area_E.service / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_util = (stats.area_E.service / cap_time_edge) if cap_time_edge > 0 else 0.0

    edge_C_W  = (stats.area_C.node / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq = (stats.area_C.queue / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_L  = stats.area_C.node / sim_time
    edge_C_Lq = stats.area_C.queue / sim_time
    edge_C_S  = (stats.area_C.service / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_util = (stats.area_C.service / cap_time_edge) if cap_time_edge > 0 else 0.0

    # --- Cloud (∞-server: queue≈0, W dalle aree) ---
    cloud_W  = (stats.area_cloud.node / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    cloud_Wq = (stats.area_cloud.queue / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    cloud_L  = stats.area_cloud.node / sim_time
    cloud_Lq = stats.area_cloud.queue / sim_time
    cloud_S  = (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    cloud_X  = stats.index_cloud / sim_time
    cloud_busy_avg = stats.area_cloud.service / sim_time  # = L per ∞-server

    # --- Coordinator ---
    coord_W = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_Wq = (stats.area_coord.queue / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_L = stats.area_coord.node / sim_time
    coord_Lq = stats.area_coord.queue / sim_time
    coord_S = (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_X = stats.index_coord / sim_time
    coord_util = (stats.area_coord.service / cap_time_coord) if cap_time_coord > 0 else 0.0

    results = {
        "seed": seed,
        # Per la scalabilità “giornaliera” non ha senso un unico λ,
        # lasciamo None (il CSV writer gestisce i default).
        "lambda": (forced_lambda if forced_lambda is not None else None),
        "slot": slot_index,

        # Edge (totali + per classe)
        "edge_server_number": cs.EDGE_SERVERS,
        "edge_avg_wait": edge_W, "edge_avg_delay": edge_Wq,
        "edge_L": edge_L, "edge_Lq": edge_Lq,
        "edge_service_time_mean": edge_S,
        "edge_avg_busy_servers": edge_busy_avg,
        "edge_throughput": edge_X,
        "edge_utilization": edge_util,
        "edge_E_utilization": edge_E_util,
        "edge_C_utilization": edge_C_util,

        "edge_E_avg_delay": edge_E_Wq,
        "edge_E_avg_response": edge_E_W,
        "edge_C_avg_delay": edge_C_Wq,
        "edge_C_avg_response": edge_C_W,
        "edge_E_L": edge_E_L, "edge_E_Lq": edge_E_Lq,
        "edge_C_L": edge_C_L, "edge_C_Lq": edge_C_Lq,

        # Cloud
        "cloud_avg_wait": cloud_W, "cloud_avg_delay": cloud_Wq,
        "cloud_L": cloud_L, "cloud_Lq": cloud_Lq,
        "cloud_service_time_mean": cloud_S,
        "cloud_avg_busy_servers": cloud_busy_avg,
        "cloud_throughput": cloud_X,

        # Coordinator
        "coord_server_number": cs.COORD_EDGE_SERVERS,
        "coord_avg_wait": coord_W, "coord_avg_delay": coord_Wq,
        "coord_L": coord_L, "coord_Lq": coord_Lq,
        "coord_service_time_mean": coord_S,
        "coord_utilization": coord_util,
        "coord_throughput": coord_X,

        # Meta
        "pc": cs.P_C, "p1": cs.P1_PROB, "p2": cs.P2_PROB, "p3": cs.P3_PROB, "p4": cs.P4_PROB,

        # Tracce autoscaling
        "edge_scal_trace": edge_scal_trace,
        "coord_scal_trace": coord_scal_trace,
    }
    return results


