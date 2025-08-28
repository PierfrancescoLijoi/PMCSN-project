from utils.sim_utils import *
from utils.improved_simulation_stats import SimulationStats_improved
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def edge_coord_scalability_simulation_improved(stop, forced_lambda=None, slot_index=None, fixed_edge_servers=1):
    """
    Edge: coda FIFO M/M/m che serve SOLO job 'E'.
          m = numero di core. Scaling per-core: min 2, max cs.EDGE_SERVERS_MAX (es. 6).
    Coordinator: pool parallelo SCALABILE (priorità P3/P4 su P1/P2).
    Cloud: ∞-server.
    Dopo il Cloud: i job vanno in coda 'feedback' FIFO (1 server, Exp(1.0s)) e poi escono.
    """
    seed = getSeed()
    reset_arrival_temp()

    # --- Limiti core Edge: min 2, max come da costanti (es. 6) ---
    min_cores = 1
    cs.EDGE_SERVERS_MAX = max(min_cores, int(getattr(cs, "EDGE_SERVERS_MAX", 6)))
    # core iniziali (clampati tra 2 e max)
    cs.EDGE_SERVERS = max(min_cores, min(int(round(fixed_edge_servers)), cs.EDGE_SERVERS_MAX))

    # Coordinator parte con almeno 1 server
    cs.COORD_EDGE_SERVERS = max(1, int(getattr(cs, "COORD_EDGE_SERVERS", 1)))
    cs.COORD_SERVERS_MAX  = max(cs.COORD_EDGE_SERVERS, int(getattr(cs, "COORD_SERVERS_MAX", 6)))

    stats = SimulationStats_improved()
    stats.reset(cs.START)
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
    # --- Cloud ∞-server (lista dei tempi di completamento) ---
    stats.cloud_comp = []
    stats.t.completion_cloud = cs.INFINITY


    # --- Stato feedback (post-Cloud) ---
    stats.queue_feedback = []
    stats.number_feedback = 0
    stats.t.completion_feedback = cs.INFINITY

    # Finestra decisionale (usata sia per Edge che per Coordinator)
    decision_interval = float(getattr(cs, "SCALING_WINDOW", 1000.0))
    last_checkpoint_coord = stats.t.current

    # Tracce scaling
    edge_scal_trace = []
    coord_scal_trace = []

    # --- Stato Edge (dimensionato al massimo; uso i primi cs.EDGE_SERVERS) ---
    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy       = [False]       * cs.EDGE_SERVERS_MAX
    edge_server_jobtype    = [None]        * cs.EDGE_SERVERS_MAX  # "E"

    # --- Stato Coordinator (scalabile) ---
    coord_completion_times = [cs.INFINITY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False]       * cs.COORD_SERVERS_MAX
    coord_server_ptype     = [None]        * cs.COORD_SERVERS_MAX  # "P1"/"P2"/"P3"/"P4"

    # Accumulatori per utilizzazione su finestra (separati dalle statistiche totali)
    edge_service_win  = 0.0
    last_checkpoint_edge = stats.t.current

    # Per Coordinator (finestra)
    coord_service_win = 0.0

    def edge_assign_if_possible(sidx):
        """Tenta di avviare servizio su un core Edge (solo 'E')."""
        if sidx >= cs.EDGE_SERVERS:
            return False
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E"
            service = GetServiceEdgeE_im()
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job
            # Aree di servizio
            stats.area_edge.service += service        # totale Edge
            nonlocal edge_service_win
            edge_service_win += service               # finestra Edge (per scaling)
            stats.area_E.service += service           # classe E
            return True
        return False

    def coord_assign_if_possible(sidx):
        """Tenta di avviare servizio al Coordinator (priorità P3/P4)."""
        if sidx >= cs.COORD_EDGE_SERVERS:
            return False
        if not coord_server_busy[sidx]:
            if stats.queue_coord_high:
                ptype = stats.queue_coord_high.pop(0)   # "P3"|"P4"
                service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                ptype = stats.queue_coord_low.pop(0)    # "P1"|"P2"
                service = GetServiceCoordP1P2()
            else:
                return False
            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
            coord_server_ptype[sidx] = ptype
            stats.area_coord.service += service
            nonlocal coord_service_win
            coord_service_win += service               # finestra Coordinator (per scaling)
            return True
        return False


    while stats.t.arrival < stop or (stats.number_edge + stats.number_cloud + stats.number_coord + stats.number_feedback) > 0:
        next_completion_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else cs.INFINITY
        next_completion_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else cs.INFINITY

        stats.t.next = Min(
            stats.t.arrival,
            next_completion_edge,
            stats.t.completion_cloud,
            next_completion_coord,
            stats.t.completion_feedback
        )

        delta = stats.t.next - stats.t.current
        # Aree numero in nodo
        if stats.number_edge  > 0: stats.area_edge.node  += delta * stats.number_edge
        if stats.number_cloud > 0: stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0: stats.area_coord.node += delta * stats.number_coord
        if stats.number_E     > 0: stats.area_E.node     += delta * stats.number_E

        if stats.number_feedback > 0:
            stats.area_feedback.node += delta * stats.number_feedback

        stats.t.current = stats.t.next

        # ---------- SCALING EDGE: per-core, min=2, max=EDGE_SERVERS_MAX ----------
        if (stats.t.current - last_checkpoint_edge) >= decision_interval:
            obs_time = max(1e-12, (stats.t.current - last_checkpoint_edge) * max(1, cs.EDGE_SERVERS))
            utilization_e = edge_service_win / obs_time
            edge_scal_trace.append((stats.t.current, cs.EDGE_SERVERS, utilization_e))

            if utilization_e > cs.UTILIZATION_UPPER and cs.EDGE_SERVERS < cs.EDGE_SERVERS_MAX:
                cs.EDGE_SERVERS += 1  # +1 core
            elif utilization_e < cs.UTILIZATION_LOWER and cs.EDGE_SERVERS > min_cores:
                desired = cs.EDGE_SERVERS - 1          # -1 core
                # non spegnere core occupati
                highest_busy = max([i for i, b in enumerate(edge_server_busy[:cs.EDGE_SERVERS]) if b], default=-1)
                cs.EDGE_SERVERS= max(min_cores, max(desired, highest_busy + 1))

            edge_service_win = 0.0
            last_checkpoint_edge = stats.t.current

            # prova ad avviare nuovi servizi se ho aumentato i core
            for i in range(cs.EDGE_SERVERS):
                if not stats.queue_edge: break
                edge_assign_if_possible(i)

        # ---------- SCALING COORDINATOR: per-core (come prima) ----------
        if (stats.t.current - last_checkpoint_coord) >= decision_interval:
            obs_time_c = max(1e-12, (stats.t.current - last_checkpoint_coord) * max(1, cs.COORD_EDGE_SERVERS))
            utilization_c = coord_service_win / obs_time_c
            coord_scal_trace.append((stats.t.current, cs.COORD_EDGE_SERVERS, utilization_c))

            if utilization_c > cs.UTILIZATION_UPPER and cs.COORD_EDGE_SERVERS < cs.COORD_SERVERS_MAX:
                cs.COORD_EDGE_SERVERS += 1
            elif utilization_c < cs.UTILIZATION_LOWER and cs.COORD_EDGE_SERVERS > 1:
                desired_c = cs.COORD_EDGE_SERVERS - 1
                highest_busy_c = max([i for i, b in enumerate(coord_server_busy[:cs.COORD_EDGE_SERVERS]) if b], default=-1)
                cs.COORD_EDGE_SERVERS = max(1, max(desired_c, highest_busy_c + 1))

            coord_service_win = 0.0
            last_checkpoint_coord = stats.t.current

            # prova ad avviare nuovi servizi se ho aumentato i core coord
            for i in range(cs.COORD_EDGE_SERVERS):
                if not (stats.queue_coord_high or stats.queue_coord_low): break
                coord_assign_if_possible(i)

        # ---------------------- EVENTI ----------------------

        # Arrivo all'Edge (job "E")
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E    += 1
            stats.queue_edge.append("E")
            stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
            if stats.t.arrival > stop:
                stats.t.arrival = cs.INFINITY

            # prova ad avviare subito su un core libero
            for i in range(cs.EDGE_SERVERS):
                if edge_assign_if_possible(i):
                    break

        # Completamento Edge (serve solo "E")
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1
                stats.number_E    -= 1

                # Routing: Cloud oppure Coordinator
                selectStream(3)
                r = rng_random()
                if r < cs.P_C:

                    # → Cloud (∞-server): avvia SEMPRE un servizio indipendente
                    stats.number_cloud += 1
                    service = GetServiceCloud()
                    stats.area_cloud.service += service
                    stats.area_C.service += service

                    # programma il completamento di QUESTO job
                    stats.cloud_comp.append(stats.t.current + service)
                    stats.t.completion_cloud = (min(stats.cloud_comp) if stats.cloud_comp else cs.INFINITY)

                else:
                    # → Coordinator (P1..P4), priorità P3/P4
                    stats.number_coord += 1
                    coord_r = (r - cs.P_C) / max(1e-12, (1.0 - cs.P_C))
                    if   coord_r < cs.P1_PROB:                          stats.queue_coord_low.append("P1")
                    elif coord_r < cs.P1_PROB + cs.P2_PROB:            stats.queue_coord_low.append("P2")
                    elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB: stats.queue_coord_high.append("P3")
                    else:                                               stats.queue_coord_high.append("P4")
                    # prova assegnazione immediata
                    for j in range(cs.COORD_EDGE_SERVERS):
                        if coord_assign_if_possible(j):
                            break

                # prova a riempire il core appena liberato
                edge_assign_if_possible(i)
                break

        # Completamento Cloud → FEEDBACK (∞-server: rimuovi l'istanza completata)
        if stats.t.current == stats.t.completion_cloud:
            stats.index_cloud += 1
            stats.number_cloud -= 1

            # rimuovi il completamento corrente dalla lista (tollerante al floating point)
            try:
                stats.cloud_comp.remove(stats.t.current)
            except ValueError:
                if stats.cloud_comp:
                    # rimuovi l'elemento più vicino a t.current
                    m = min(stats.cloud_comp, key=lambda x: abs(x - stats.t.current))
                    stats.cloud_comp.remove(m)

            # aggiorna il prossimo completamento Cloud
            stats.t.completion_cloud = (min(stats.cloud_comp) if stats.cloud_comp else cs.INFINITY)

            # → coda feedback (come prima)
            stats.number_feedback += 1
            stats.queue_feedback.append("FB")
            if stats.number_feedback == 1:
                service = GetServiceFeedback_improved()
                stats.t.completion_feedback = stats.t.current + service
                stats.area_feedback.service += service

        # Completamento Coordinator
        for j in range(cs.COORD_EDGE_SERVERS):
            if stats.t.current == coord_completion_times[j]:
                coord_server_busy[j] = False
                coord_completion_times[j] = cs.INFINITY
                finished_ptype = coord_server_ptype[j]
                coord_server_ptype[j] = None

                stats.index_coord += 1
                stats.number_coord -= 1
                # conteggi classe E al completamento Coordinator
                stats.count_E += 1
                if   finished_ptype == "P1": stats.count_E_P1 += 1
                elif finished_ptype == "P2": stats.count_E_P2 += 1
                elif finished_ptype == "P3": stats.count_E_P3 += 1
                elif finished_ptype == "P4": stats.count_E_P4 += 1

                # prova riassegnazione
                coord_assign_if_possible(j)
                break

        # Completamento Feedback → uscita
        if stats.t.current == stats.t.completion_feedback:
            if stats.queue_feedback:
                stats.queue_feedback.pop(0)
            stats.number_feedback -= 1
            # NEW: conta i completamenti feedback per X_fb e medie per job
            stats.index_feedback = getattr(stats, 'index_feedback', 0) + 1
            if stats.number_feedback > 0:
                service = GetServiceFeedback_improved()
                stats.t.completion_feedback = stats.t.current + service
                stats.area_feedback.service += service  # NEW: accumula servizio FB
            else:
                stats.t.completion_feedback = cs.INFINITY

    # Fine simulazione: aree di coda
    stats.calculate_area_queue()

    T = max(1e-12, stats.t.current - cs.START)

    # Medie esistenti...
    edge_W = stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0.0
    edge_Wq = stats.area_edge.queue / stats.index_edge if stats.index_edge > 0 else 0.0
    cloud_W = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
    cloud_Wq = stats.area_cloud.queue / stats.index_cloud if stats.index_cloud > 0 else 0.0
    coord_W = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
    coord_Wq = stats.area_coord.queue / stats.index_coord if stats.index_coord > 0 else 0.0

    # NEW: Feedback (per job e per tempo)
    fb_compl = getattr(stats, 'index_feedback', 0)
    W_fb = (stats.area_feedback.node / fb_compl) if fb_compl > 0 else 0.0
    Wq_fb = (stats.area_feedback.queue / fb_compl) if fb_compl > 0 else 0.0
    L_fb = stats.area_feedback.node / T
    Lq_fb = stats.area_feedback.queue / T
    s_fb = (stats.area_feedback.service / fb_compl) if fb_compl > 0 else 0.0
    rho_fb = (stats.area_feedback.service / T)  # single-server
    X_fb = (fb_compl / T) if T > 0 else 0.0

    # NEW: Edge utilization normalizzata per #core attivi (m = cs.EDGE_SERVERS)
    edge_util_norm = (stats.area_edge.service / (T * max(1, cs.EDGE_SERVERS)))

    # --- NEW: numero medio "nel servente" (Ls) ---
    ENA_Ls = (stats.area_edge.service / T) if T > 0 else 0.0  # Edge_NuoviArrivi (solo E)
    FB_Ls = (stats.area_feedback.service / T) if T > 0 else 0.0  # Edge_Feedback (solo C)

    results = {
        'seed': seed,
        'lambda': forced_lambda,
        'slot': slot_index,

        # --- Edge_NuoviArrivi (ex-Edge) ---
        'edge_server_number': cs.EDGE_SERVERS,
        'edge_NuoviArrivi_avg_wait': edge_W,
        'edge_NuoviArrivi_avg_delay': edge_Wq,
        'edge_NuoviArrivi_L': stats.area_edge.node / T,
        'edge_NuoviArrivi_Lq': stats.area_edge.queue / T,
        'edge_NuoviArrivi_Ls': ENA_Ls,  # << NEW (qui)
        'edge_NuoviArrivi_service_time_mean': (
                    stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_NuoviArrivi_utilization': (stats.area_edge.service / (T * max(1, cs.EDGE_SERVERS))),
        'edge_NuoviArrivi_throughput': (stats.index_edge / T),
        'Edge_NuoviArrivi_E_Ts': (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0,

        # --- Edge_Feedback ---
        'edge_Feedback_avg_wait': (stats.area_feedback.node / max(1, getattr(stats, 'index_feedback', 0))) if getattr(
            stats, 'index_feedback', 0) > 0 else 0.0,
        'edge_Feedback_avg_delay': (stats.area_feedback.queue / max(1, getattr(stats, 'index_feedback', 0))) if getattr(
            stats, 'index_feedback', 0) > 0 else 0.0,
        'edge_Feedback_L': stats.area_feedback.node / T,
        'edge_Feedback_Lq': stats.area_feedback.queue / T,
        'edge_Feedback_Ls': FB_Ls,
        'edge_Feedback_service_time_mean': (
                    stats.area_feedback.service / max(1, getattr(stats, 'index_feedback', 0))) if getattr(stats,
                                                                                                          'index_feedback',
                                                                                                          0) > 0 else 0.0,
        'edge_Feedback_utilization': (stats.area_feedback.service / T),
        'edge_Feedback_throughput': (getattr(stats, 'index_feedback', 0) / T) if T > 0 else 0.0,
        'Edge_Feedback_E_Ts': (stats.area_feedback.service / max(1, getattr(stats, 'index_feedback', 0))) if getattr(
            stats, 'index_feedback', 0) > 0 else 0.0,

        # --- Cloud ---
        'cloud_avg_wait': cloud_W,
        'cloud_avg_delay': cloud_Wq,
        'cloud_L': stats.area_cloud.node / T,
        'cloud_Lq': stats.area_cloud.queue / T,
        'cloud_service_time_mean': (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0,
        'cloud_avg_busy_servers': stats.area_cloud.service / T,
        'cloud_throughput': (stats.index_cloud / T),

        # --- Coordinator ---
        'coord_server_number': cs.COORD_EDGE_SERVERS,
        'coord_avg_wait': coord_W,
        'coord_avg_delay': coord_Wq,
        'coord_L': stats.area_coord.node / T,
        'coord_Lq': stats.area_coord.queue / T,
        'coord_service_time_mean': (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0,
        'coord_utilization': stats.area_coord.service / T,
        'coord_throughput': (stats.index_coord / T),

        # Tracce scaling (debug/plot)
        'edge_scal_trace': edge_scal_trace,
        'coord_scal_trace': coord_scal_trace,
    }
    return results

