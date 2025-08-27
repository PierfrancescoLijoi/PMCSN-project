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

    # --- NUOVO: busy-time totali sull'intero orizzonte (non azzerati tra finestre) ---
    total_edge_busy = 0.0
    total_coord_busy = 0.0
    total_edge_busy_E = 0.0
    total_edge_busy_C = 0.0

    # Helper assegnazioni
    def edge_assign_if_possible(sidx):
        nonlocal total_edge_busy, total_edge_busy_E, total_edge_busy_C
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E" o "C"

            # servizio per classe
            if job == "E":
                service = GetServiceEdgeE()
                stats.area_E.service += service
                total_edge_busy_E += service
            else:
                service = GetServiceEdgeC()
                stats.area_C.service += service
                total_edge_busy_C += service

            # pianifica completamento su quel server Edge
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job

            # totale Edge
            stats.area_edge.service += service
            total_edge_busy += service
            return True
        return False

    def coord_assign_if_possible(sidx):
        nonlocal total_coord_busy
        if not coord_server_busy[sidx]:
            # priorità: high poi low
            if stats.queue_coord_high:
                cls = stats.queue_coord_high.pop(0)  # "P3"|"P4"
                service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                cls = stats.queue_coord_low.pop(0)  # "P1"|"P2"
                service = GetServiceCoordP1P2()
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

        if stats.number_E > 0: stats.area_E.node += delta * stats.number_E  # <-- UNA volta
        if stats.number_C > 0: stats.area_C.node += delta * stats.number_C  # <-- C già ok


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

            stats.number_E += 1


            # prossimo arrivo
            if forced_lambda is not None:
                lam = max(1e-12, float(forced_lambda))
                selectStream(0)
                stats.t.arrival = stats.t.current + Exponential(1.0 / lam)
            else:
                stats.t.arrival = GetArrival(stats.t.current)
            kick_assign_all()
            continue

        #
        # Evento: COMPLETAMENTO EDGE
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                completed_type = edge_server_jobtype[i]

                # libera il server i
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                # contabilità Edge (totale)
                stats.index_edge += 1
                stats.number_edge -= 1

                if completed_type == "E":
                    # completamento di un job E
                    stats.index_edge_E += 1
                    stats.number_E -= 1

                    # routing: Cloud oppure Coordinator
                    selectStream(3)
                    r = rng_random()
                    if r < cs.P_C:
                        # → Cloud (∞-server serializzato)
                        stats.number_cloud += 1
                        if stats.number_cloud == 1:
                            service = GetServiceCloud()
                            stats.t.completion_cloud = stats.t.current + service
                            stats.area_cloud.service += service
                    else:
                        # → Coordinator: classi P1..P4
                        stats.number_coord += 1
                        coord_r = (r - cs.P_C) / max(1e-12, (1.0 - cs.P_C))
                        if coord_r < cs.P1_PROB:
                            stats.queue_coord_low.append("P1");
                            stats.count_E_P1 += 1
                        elif coord_r < cs.P1_PROB + cs.P2_PROB:
                            stats.queue_coord_low.append("P2");
                            stats.count_E_P2 += 1
                        elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                            stats.queue_coord_high.append("P3");
                            stats.count_E_P3 += 1
                        else:
                            stats.queue_coord_high.append("P4");
                            stats.count_E_P4 += 1

                        # prova ad assegnare subito al Coordinator
                        for j in range(cs.COORD_EDGE_SERVERS):
                            if coord_assign_if_possible(j):
                                break

                else:
                    # completed_type == "C": termina all'Edge
                    stats.index_edge_C += 1
                    stats.number_C -= 1
                    # (facoltativo) stats.count_C += 1

                # dopo un completamento prova ad assegnare nuovi job
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

            else:
                stats.t.completion_cloud = cs.INFINITY

            # ritorno all'Edge come job "C"
            stats.queue_edge.append("C")
            # FIX: incrementa il numero in sistema all'Edge per il ritorno dal Cloud
            stats.number_edge += 1
            stats.number_C += 1
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

    stats.calculate_area_queue()  # mantiene area_x.node; Lq lo calcoliamo sotto con i busy totali
    sim_time = max(1e-12, stats.t.current - cs.START)

    # --- Edge (totali) ---
    edge_W = (stats.area_edge.node / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_Wq = ((stats.area_edge.node - total_edge_busy) / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_L = stats.area_edge.node / sim_time
    edge_Lq = (stats.area_edge.node - total_edge_busy) / sim_time
    edge_S = (total_edge_busy / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_X = stats.index_edge / sim_time
    edge_busy_avg = total_edge_busy / sim_time
    edge_util = (total_edge_busy / cap_time_edge) if cap_time_edge > 0 else 0.0  # ∈ [0,1]

    # --- Edge (per classe) ---
    edge_E_W = (stats.area_E.node / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq = ((stats.area_E.node - total_edge_busy_E) / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_L = stats.area_E.node / sim_time
    edge_E_Lq = (stats.area_E.node - total_edge_busy_E) / sim_time
    edge_E_S = (total_edge_busy_E / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_util = (total_edge_busy_E / cap_time_edge) if cap_time_edge > 0 else 0.0

    edge_C_W = (stats.area_C.node / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq = ((stats.area_C.node - total_edge_busy_C) / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_L = stats.area_C.node / sim_time
    edge_C_Lq = (stats.area_C.node - total_edge_busy_C) / sim_time
    edge_C_S = (total_edge_busy_C / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_util = (total_edge_busy_C / cap_time_edge) if cap_time_edge > 0 else 0.0

    # --- Cloud (∞-server) ---
    cloud_W = cs.CLOUD_SERVICE
    cloud_Wq = 0.0
    cloud_Lq = 0.0
    cloud_L = stats.area_cloud.node / sim_time
    cloud_S = stats.area_cloud.service / stats.index_cloud
    cloud_X = stats.index_cloud / sim_time
    cloud_busy_avg = stats.area_cloud.service / sim_time

    # --- Coordinator ---
    coord_W = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_Wq = ((stats.area_coord.node - total_coord_busy) / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_L = stats.area_coord.node / sim_time
    coord_Lq = (stats.area_coord.node - total_coord_busy) / sim_time
    coord_S = (total_coord_busy / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_X = stats.index_coord / sim_time
    coord_util = (total_coord_busy / cap_time_coord) if cap_time_coord > 0 else 0.0

    results = {
        "seed": seed,
        "lambda": (forced_lambda if forced_lambda is not None else GetLambda(stats.t.current)),
        "slot": slot_index,

        # Edge
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
        "cloud_L":  cloud_busy_avg, "cloud_Lq": cloud_Lq,
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

        # (facoltativo) meta: proba
        "pc": cs.P_C, "p1": cs.P1_PROB, "p2": cs.P2_PROB, "p3": cs.P3_PROB, "p4": cs.P4_PROB,

        # tracce autoscaling
        "edge_scal_trace": edge_scal_trace,
        "coord_scal_trace": coord_scal_trace,
    }
    return results

