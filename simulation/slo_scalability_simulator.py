# simulation/slo_scalability_simulator.py
# ---------------------------------------------------------
# Simulatore EVENT-DRIVEN a orizzonte FINITO (24h) con autoscaling SLO-driven (solo Edge).
# Flusso allineato al modello standard; cambiano solo le condizioni di scalabilità.
# Metriche per il grafico: tempo medio di risposta *cumulato* all’Edge (W_bar).
# ---------------------------------------------------------

import heapq
import math
from typing import List, Tuple, Dict

import utils.constants as cs
from libraries.rngs import getSeed, selectStream, random as rng_random
from utils.sim_utils import (
    GetArrival, Min, reset_arrival_temp,
    GetServiceEdgeE, GetServiceEdgeC, GetServiceCloud,
    GetServiceCoordP1P2, GetServiceCoordP3P4,
    lehmer_replica_seed
)
from utils.simulation_stats import SimulationStats

INFTY = cs.INFINITY

# -------------------------
# Parametri controller SLO-driven (configurabili anche da constants.py)
# -------------------------
SLO_EDGE       = 3.0
W_TARGET       = 2.4  # target < SLO (≈20% margine)
SCALING_WINDOW = float(getattr(cs, "SCALING_WINDOW", 900.0))
SCALE_COOLDOWN = float(getattr(cs, "SCALE_COOLDOWN", 1800.0))
DOWNSCALE_HOLD = float(getattr(cs, "DOWNSCALE_HOLD", 1800.0))

# Trigger asimmetrici
W_UP,   W_DOWN   = 2.8, 2.0
LQ_UP,  LQ_DOWN  = 10.0, 2.0
RHO_UP, RHO_DOWN = 0.80, 0.55

# Anti-stickiness
C_DRAIN_THRESH   = 1
FORCE_DOWN_AFTER = 3

# -------------------------
# Utility Erlang-C per dimensionamento m*
# -------------------------
def mmc_wait(lam: float, mu: float, m: int) -> Tuple[float, float]:
    """Erlang-C M/M/m: (W, rho)."""
    if m <= 0 or mu <= 0.0:
        return float("inf"), 1.0
    rho = lam / (m * mu)
    if rho >= 1.0:
        return float("inf"), rho
    a = lam / mu
    s = 0.0
    for k in range(m):
        s += (a**k) / math.factorial(k)
    p0 = 1.0 / (s + (a**m) / (math.factorial(m) * (1.0 - rho)))
    Pq = ((a**m) * p0) / (math.factorial(m) * (1.0 - rho))
    Wq = Pq / (m*mu - lam)
    W  = Wq + 1.0/mu
    return W, rho

def day_lambda(t: float, forced_lambda=None) -> float:
    """λ(t) su 24h usando cs.LAMBDA_SLOTS con wrap; se forced_lambda è dato, restituisce quello."""
    if forced_lambda is not None:
        return max(1e-12, float(forced_lambda))
    tt = float(t) % 86400.0
    for start, end, lam in cs.LAMBDA_SLOTS:
        s = float(start); e = float(end); l = float(lam)
        if e > 86400.0:
            if (s <= tt < 86400.0) or (0.0 <= tt < (e - 86400.0)):
                return max(1e-12, l)
        else:
            if s <= tt < e:
                return max(1e-12, l)
    return max(1e-12, float(getattr(cs, "LAMBDA", 1.0)))

# -------------------------
# Singola replica (24h) — i semi RNG sono PIANTATI all’esterno
# -------------------------
def run_single_replication_24h(forced_lambda=None, slot_index=None) -> Dict:
    """
    Replica a orizzonte FINITO (24h con λ(t) se forced_lambda=None).
    Controller snello:
      - a ogni finestra T: stima λ^ (arrivi E+C) e μ^ (1/tempo di servizio medio osservato),
        scegli minimo m* con Erlang-C s.t. W(m*) <= W_TARGET (2.4 s),
      - isteresi (UP aggressivo, DOWN conservativo), hold-time per downscale, cooldown tra decisioni,
      - coda unica Edge; "drain C": niente down se open_C sopra soglia,
      - kicker: se m*=1 per N finestre e segnali bassi → forza 2→1.
    """
    reset_arrival_temp()

    stats = SimulationStats()
    stats.reset(cs.START)

    # --- Parametri controller (da constants.py) ---
    SLO_EDGE       = float(getattr(cs, "SLO_EDGE", 3.0))       # QoS da rispettare: 3 s
    W_TARGET       = float(getattr(cs, "W_TARGET", 2.4))       # margine 20% sotto SLO

    SCALING_WINDOW = float(getattr(cs, "SCALING_WINDOW", 600.0))   # T: 10 min
    decision_interval = SCALING_WINDOW

    SCALE_COOLDOWN_UP   = float(getattr(cs, "SCALE_COOLDOWN_UP", 600.0))
    SCALE_COOLDOWN_DOWN = float(getattr(cs, "SCALE_COOLDOWN_DOWN", 1800.0))
    DOWNSCALE_HOLD      = float(getattr(cs, "DOWNSCALE_HOLD", 1200.0))
    FREEZE_AT_END       = float(getattr(cs, "FREEZE_AT_END", 0.0))  # opzionale (0 = disattivo)

    # Isteresi (UP aggressivo, DOWN conservativo)
    W_UP   = float(getattr(cs, "W_UP",   2.6))
    W_DOWN = float(getattr(cs, "W_DOWN", 2.1))
    LQ_UP,  LQ_DOWN  = float(getattr(cs, "LQ_UP",  4.0)),  float(getattr(cs, "LQ_DOWN", 1.2))
    RHO_UP, RHO_DOWN = float(getattr(cs, "RHO_UP", 0.78)), float(getattr(cs, "RHO_DOWN", 0.55))

    # Drain "C" e kicker
    C_DRAIN_THRESH   = int(getattr(cs, "C_DRAIN_THRESH", 2))   # non scendere se open_C > soglia
    FORCE_DOWN_AFTER = int(getattr(cs, "FORCE_DOWN_AFTER", 2)) # m*=1 per N finestre → forza 2→1

    # Orizzonte 24h (se non forzi λ a singolo valore)
    stop = 86400.0 if forced_lambda is None else float(getattr(cs, "STOP", 86400.0))

    # Pool server
    cs.EDGE_SERVERS       = int(getattr(cs, "EDGE_SERVERS_INIT", getattr(cs, "EDGE_SERVERS", 1)))
    cs.COORD_EDGE_SERVERS = int(getattr(cs, "COORD_EDGE_SERVERS_INIT", getattr(cs, "COORD_SERVERS", 1)))
    cs.EDGE_SERVERS_MAX   = max(cs.EDGE_SERVERS,       int(getattr(cs, "EDGE_SERVERS_MAX", 6)))
    cs.COORD_SERVERS_MAX  = max(cs.COORD_EDGE_SERVERS, int(getattr(cs, "COORD_SERVERS_MAX", 6)))

    # Stato server
    edge_completion_times = [INFTY] * cs.EDGE_SERVERS_MAX
    edge_server_busy      = [False]  * cs.EDGE_SERVERS_MAX
    edge_server_jobtype   = [None]   * cs.EDGE_SERVERS_MAX
    edge_server_arrival   = [None]   * cs.EDGE_SERVERS_MAX  # per sojourn

    coord_completion_times = [INFTY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False]  * cs.COORD_SERVERS_MAX
    coord_server_ptype     = [None]   * cs.COORD_SERVERS_MAX

    # Code (coda unica per Edge)
    stats.queue_edge = []           # ("E"|"C", t_arr_edge)
    stats.queue_coord_high = []
    stats.queue_coord_low  = []
    stats.cloud_heap = []

    # Primo arrivo
    stats.t.arrival = GetArrival(stats.t.current, day_lambda(stats.t.current, forced_lambda))
    if stats.t.arrival > stop:
        stats.t.arrival = INFTY

    slo_scal_trace: List[Tuple[float,int,dict]] = []

    # Bookkeeping finestra SLO
    last_cp_edge    = stats.t.current
    last_node_edge  = stats.area_edge.node
    last_index_edge = stats.index_edge

    last_up_time    = None
    last_down_time  = None
    downsafe_since  = None
    mstar1_streak   = 0  # per il kicker 2→1

    edge_arrivals_since_cp = 0
    svc_sum_since_cp = 0.0
    svc_cnt_since_cp = 0

    # cumulativi risposta Edge
    cum_edge_sojourn = 0.0
    cum_edge_compl   = 0

    # Helper: assegnazioni
    def edge_assign_if_possible(sidx: int):
        nonlocal svc_sum_since_cp, svc_cnt_since_cp
        if not edge_server_busy[sidx] and stats.queue_edge:
            job, t_arr_edge = stats.queue_edge.pop(0)
            if job == "E":
                service = GetServiceEdgeE(); stats.area_E.service += service
            else:
                service = GetServiceEdgeC(); stats.area_C.service += service
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx]    = True
            edge_server_jobtype[sidx] = job
            edge_server_arrival[sidx] = t_arr_edge
            stats.area_edge.service  += service
            svc_sum_since_cp += service; svc_cnt_since_cp += 1
            return True
        return False

    def coord_assign_if_possible(sidx: int):
        if not coord_server_busy[sidx]:
            if stats.queue_coord_high:
                _ = stats.queue_coord_high.pop(0); service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                _ = stats.queue_coord_low.pop(0);  service = GetServiceCoordP1P2()
            else:
                return False
            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
            stats.area_coord.service += service
            return True
        return False

    def kick_assign_all():
        for i in range(cs.EDGE_SERVERS):
            if not stats.queue_edge: break
            edge_assign_if_possible(i)
        for i in range(cs.COORD_EDGE_SERVERS):
            if not (stats.queue_coord_high or stats.queue_coord_low): break
            coord_assign_if_possible(i)

    # --- Loop principale ---
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        next_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else INFTY
        next_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else INFTY
        next_cloud = stats.cloud_heap[0] if stats.cloud_heap else INFTY
        stats.t.next = Min(stats.t.arrival, next_edge, next_cloud, next_coord)

        # Aree
        dt = stats.t.next - stats.t.current
        if dt < 0: dt = 0.0
        if stats.number_edge  > 0: stats.area_edge.node  += dt * stats.number_edge
        if stats.number_cloud > 0: stats.area_cloud.node += dt * stats.number_cloud
        if stats.number_coord > 0: stats.area_coord.node += dt * stats.number_coord
        if stats.number_E > 0:     stats.area_E.node     += dt * stats.number_E
        if stats.number_C > 0:     stats.area_C.node     += dt * stats.number_C

        stats.t.current = stats.t.next

        # --- Checkpoint SLO-driven (ogni T) ---
        if stats.t.current - last_cp_edge >= decision_interval:
            win = max(1e-12, stats.t.current - last_cp_edge)

            # stime finestra
            delta_node = stats.area_edge.node - last_node_edge
            L_avg  = delta_node / win
            lam_hat = edge_arrivals_since_cp / win           # arrivi a Edge (E + C) nella finestra
            sbar = (svc_sum_since_cp / svc_cnt_since_cp) if svc_cnt_since_cp > 0 else 0.0
            mu_hat = (1.0 / sbar) if sbar > 0 else 0.0
            m_cur = max(1, cs.EDGE_SERVERS)
            rho_hat = (lam_hat / (m_cur * mu_hat)) if mu_hat > 0 else 1.0

            delta_idx = stats.index_edge - last_index_edge
            W_ma = (delta_node / delta_idx) if delta_idx > 0 else (L_avg / max(lam_hat, 1e-12))

            # Lq media approssimata
            Lq_avg = max(0.0, L_avg - ((stats.index_edge / max(1.0, win)) / (mu_hat * m_cur)) if mu_hat > 0 else L_avg)

            # --- m* da Erlang-C verso W_TARGET (salto diretto a m*)
            def mmc_wait(lam, mu, m):
                # usa la definizione globale se presente; fallback locale
                return globals()["mmc_wait"](lam, mu, m)

            m_star = cs.EDGE_SERVERS_MAX
            for m in range(1, cs.EDGE_SERVERS_MAX + 1):
                Wm, _ = mmc_wait(lam_hat, mu_hat, m)
                if Wm <= W_TARGET:
                    m_star = m
                    break

            # Trigger isteresi
            need_up = (W_ma > W_UP) or (Lq_avg > LQ_UP) or (rho_hat > RHO_UP) or (m_star > m_cur)
            low_now = (W_ma < W_DOWN) and (Lq_avg < LQ_DOWN) and (rho_hat < RHO_DOWN)

            # Drain C: non scendere se i C aperti superano soglia
            drain_ok = (stats.number_C <= C_DRAIN_THRESH)

            # Hold per il downscale (segnali bassi persistenti)
            if low_now and drain_ok:
                if downsafe_since is None:
                    downsafe_since = stats.t.current
            else:
                downsafe_since = None
            can_down = (downsafe_since is not None) and (stats.t.current - downsafe_since >= DOWNSCALE_HOLD)

            # Kicker 2→1: se m*=1 per N finestre e segnali bassi, forza 2→1 (anche con un minimo residuo di C)
            if m_star == 1:
                mstar1_streak += 1
            else:
                mstar1_streak = 0
            kicker_ready = (m_cur == 2) and (m_star == 1) and (mstar1_streak >= FORCE_DOWN_AFTER) and low_now

            # Cooldown
            up_cooldown_ok   = (last_up_time   is None) or (stats.t.current - last_up_time   >= SCALE_COOLDOWN_UP)
            down_cooldown_ok = (last_down_time is None) or (stats.t.current - last_down_time >= SCALE_COOLDOWN_DOWN)

            # Freeze eventuale a fine orizzonte (opzionale)
            freeze_end = (FREEZE_AT_END > 0.0) and ((stop - stats.t.current) <= FREEZE_AT_END)

            # --- Azioni ---
            if need_up and (m_cur < cs.EDGE_SERVERS_MAX) and up_cooldown_ok:
                # salto diretto a m*
                cs.EDGE_SERVERS = min(cs.EDGE_SERVERS_MAX, m_star)
                last_up_time = stats.t.current

            elif not freeze_end:
                # DOWN conservativo 1-step, solo se hold e cooldown rispettati e drain_ok
                if (m_cur > 1) and (can_down and down_cooldown_ok and drain_ok):
                    m_new = m_cur - 1
                    if not any(edge_server_busy[m_new:m_cur]):  # graceful
                        cs.EDGE_SERVERS = m_new
                        last_down_time = stats.t.current
                # Kicker (prioritario sul drain): forza 2→1
                elif kicker_ready and down_cooldown_ok:
                    if not any(edge_server_busy[1:2]):  # server indice 1 (2°) libero
                        cs.EDGE_SERVERS = 1
                        last_down_time = stats.t.current

            # Trace di debug/plot
            slo_scal_trace.append((
                stats.t.current,
                cs.EDGE_SERVERS,
                {
                    "L_avg": L_avg, "Lq_avg": Lq_avg,
                    "lam_hat": lam_hat, "mu_hat": mu_hat, "rho_hat": rho_hat,
                    "W_ma": W_ma, "m_star": m_star, "open_C": stats.number_C
                }
            ))

            # reset finestra locale
            last_cp_edge    = stats.t.current
            last_node_edge  = stats.area_edge.node
            last_index_edge = stats.index_edge
            edge_arrivals_since_cp = 0
            svc_sum_since_cp = 0.0
            svc_cnt_since_cp = 0

            # riempi subito tutti i liberi
            for j in range(cs.EDGE_SERVERS):
                edge_assign_if_possible(j)

        # --- DUMP METRICHE a intervalli uniformi (per grafici) ---
        interval_dump = min(600.0, SCALING_WINDOW)  # allinea al controllo (≤10')
        if not hasattr(stats, "_next_dump"):
            stats._next_dump = 0.0
            try:
                stats._slot_bounds = [float(e) for (_s, e, _lam) in getattr(cs, "LAMBDA_SLOTS", [])]
            except Exception:
                stats._slot_bounds = []
            stats._next_slot_mark_idx = 0

        while stats.t.current >= stats._next_dump:
            avg_edge  = stats.area_edge.node  / stats.index_edge if stats.index_edge  > 0 else 0.0
            avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
            avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
            stats.edge_wait_times.append((stats._next_dump, avg_edge))
            stats.cloud_wait_times.append((stats._next_dump, avg_cloud))
            stats.coord_wait_times.append((stats._next_dump, avg_coord))
            avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
            avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
            stats.edge_E_wait_times_interval.append((stats._next_dump, avg_edge_E))
            stats.edge_C_wait_times_interval.append((stats._next_dump, avg_edge_C))
            stats._next_dump += interval_dump

        # Marker a fine slot (solo per avere un punto esatto ai confini λ, nessuna logica di scala qui)
        while getattr(stats, "_slot_bounds", None) and stats._next_slot_mark_idx < len(stats._slot_bounds) and stats.t.current >= stats._slot_bounds[stats._next_slot_mark_idx]:
            xmark = stats._slot_bounds[stats._next_slot_mark_idx]
            avg_edge  = stats.area_edge.node  / stats.index_edge if stats.index_edge  > 0 else 0.0
            avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
            avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
            stats.edge_wait_times.append((xmark, avg_edge))
            stats.cloud_wait_times.append((xmark, avg_cloud))
            stats.coord_wait_times.append((xmark, avg_coord))
            avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
            avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
            stats.edge_E_wait_times_interval.append((xmark, avg_edge_E))
            stats.edge_C_wait_times_interval.append((xmark, avg_edge_C))
            stats._next_slot_mark_idx += 1

        # ----------------- EVENTI -----------------

        # ARRIVO esterno (E)
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E    += 1
            stats.queue_edge.append(("E", stats.t.current))
            edge_arrivals_since_cp += 1
            stats.t.arrival = GetArrival(stats.t.current, day_lambda(stats.t.current, forced_lambda))
            if stats.t.arrival > stop:
                stats.t.arrival = INFTY
            kick_assign_all()
            continue

        # COMPLETAMENTO EDGE
        if stats.t.current == next_edge:
            for i in range(cs.EDGE_SERVERS):
                if stats.t.current == edge_completion_times[i]:
                    edge_server_busy[i] = False
                    job_type    = edge_server_jobtype[i]
                    t_arr_edge  = edge_server_arrival[i]
                    edge_server_jobtype[i] = None
                    edge_server_arrival[i] = None
                    edge_completion_times[i] = INFTY

                    # sojourn cumulato Edge
                    if t_arr_edge is not None:
                        cum_edge_sojourn += (stats.t.current - t_arr_edge)
                        cum_edge_compl   += 1

                    stats.index_edge += 1
                    stats.number_edge -= 1

                    if job_type == "E":
                        stats.index_edge_E += 1
                        stats.number_E     -= 1
                        # routing E
                        selectStream(3)
                        r = rng_random()
                        if r < cs.P_C:
                            # → Cloud
                            stats.number_cloud += 1
                            service = GetServiceCloud()
                            heapq.heappush(stats.cloud_heap, stats.t.current + service)
                            stats.area_cloud.service += service
                        else:
                            # → Coordinator
                            stats.number_coord += 1
                            coord_r = (r - cs.P_C) / max(1e-12, (1.0 - cs.P_C))
                            if coord_r < cs.P1_PROB:                         stats.queue_coord_low.append("P1"); stats.count_E_P1 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB:          stats.queue_coord_low.append("P2"); stats.count_E_P2 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB: stats.queue_coord_high.append("P3"); stats.count_E_P3 += 1
                            else:                                            stats.queue_coord_high.append("P4"); stats.count_E_P4 += 1
                            for j in range(cs.COORD_EDGE_SERVERS):
                                if coord_assign_if_possible(j):
                                    break
                    else:
                        stats.index_edge_C += 1
                        stats.number_C     -= 1

            kick_assign_all()
            continue

        # COMPLETAMENTO CLOUD
        if stats.cloud_heap and stats.t.current == stats.cloud_heap[0]:
            while stats.cloud_heap and stats.cloud_heap[0] <= stats.t.current + 1e-12:
                heapq.heappop(stats.cloud_heap)
                stats.index_cloud  += 1
                stats.number_cloud -= 1
                # ritorno → Edge come "C"
                stats.number_edge += 1
                stats.number_C    += 1
                stats.queue_edge.append(("C", stats.t.current))
                edge_arrivals_since_cp += 1
            kick_assign_all()
            continue

        # COMPLETAMENTO COORDINATOR
        if stats.t.current == next_coord:
            for i in range(cs.COORD_EDGE_SERVERS):
                if stats.t.current == coord_completion_times[i]:
                    coord_server_busy[i] = False
                    coord_completion_times[i] = INFTY
                    coord_server_ptype[i] = None
                    stats.index_coord  += 1
                    stats.number_coord -= 1
                    if not coord_assign_if_possible(i):
                        kick_assign_all()
                    break

    # --- Fine simulazione: metriche globali ---
    stats.calculate_area_queue()
    sim_time = max(1e-12, stats.t.current - cs.START)

    edge_W  = (stats.area_edge.node  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    edge_Wq = (stats.area_edge.queue / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    edge_L  =  stats.area_edge.node / sim_time
    edge_Lq =  stats.area_edge.queue / sim_time
    edge_S  = (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_X  =  stats.index_edge / sim_time
    edge_util = (stats.area_edge.service / sim_time) if sim_time > 0 else 0.0

    edge_E_W  = (stats.area_E.node  / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq = (stats.area_E.queue / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_L  =  stats.area_E.node / sim_time
    edge_E_Lq =  stats.area_E.queue / sim_time
    edge_E_S  = (stats.area_E.service / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_util = (stats.area_E.service / sim_time) if sim_time > 0 else 0.0

    edge_C_W  = (stats.area_C.node  / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq = (stats.area_C.queue / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_L  =  stats.area_C.node / sim_time
    edge_C_Lq =  stats.area_C.queue / sim_time
    edge_C_S  = (stats.area_C.service / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_util = (stats.area_C.service / sim_time) if sim_time > 0 else 0.0

    cloud_W  = (stats.area_cloud.node  / stats.index_cloud)  if stats.index_cloud  > 0 else 0.0
    cloud_L  =  stats.area_cloud.node / sim_time

    coord_W  = (stats.area_coord.node  / stats.index_coord)  if stats.index_coord  > 0 else 0.0
    coord_L  =  stats.area_coord.node / sim_time
    coord_util = (stats.area_coord.service / sim_time) if sim_time > 0 else 0.0
    coord_X  =  stats.index_coord / sim_time

    results = {
        "seed": getSeed(),
        "lambda": forced_lambda if forced_lambda is not None else None,
        "slot": slot_index,

        "edge_server_number": cs.EDGE_SERVERS,
        "edge_avg_wait": edge_W, "edge_avg_delay": edge_Wq,
        "edge_L": edge_L, "edge_Lq": edge_Lq,
        "edge_utilization": edge_util, "edge_throughput": edge_X,

        "edge_E_avg_wait": edge_E_W, "edge_E_avg_delay": edge_E_Wq,
        "edge_E_L": edge_E_L, "edge_E_Lq": edge_E_Lq, "edge_E_utilization": edge_E_util,

        "edge_C_avg_wait": edge_C_W, "edge_C_avg_delay": edge_C_Wq,
        "edge_C_L": edge_C_L, "edge_C_Lq": edge_C_Lq, "edge_C_utilization": edge_C_util,

        "cloud_avg_wait": cloud_W, "cloud_L": cloud_L,

        "coord_avg_wait": coord_W, "coord_L": coord_L,
        "coord_utilization": coord_util, "coord_throughput": coord_X,

        "pc": cs.P_C, "p1": cs.P1_PROB, "p2": cs.P2_PROB, "p3": cs.P3_PROB, "p4": cs.P4_PROB,

        "slo_scal_trace": slo_scal_trace,
    }
    # serie per i grafici
    results["edge_wait_times_series"] = list(stats.edge_wait_times)
    results["cloud_wait_times_series"] = list(stats.cloud_wait_times)
    results["coord_wait_times_series"] = list(stats.coord_wait_times)
    results["edge_E_wait_times_interval"] = list(stats.edge_E_wait_times_interval)
    results["edge_C_wait_times_interval"] = list(stats.edge_C_wait_times_interval)
    return results

# -------------------------
# Repliche con semi indipendenti (Lehmer jump)
# -------------------------
def run_finite_day_replications(R: int = None, base_seed: int = None) -> Dict:
    """
    Esegue R repliche indipendenti: seed_r = lehmer_replica_seed(base_seed, J_REP, r);
    chiama plantSeeds(seed_r) qui e lancia la replica da 24h con λ(t).
    Ritorna:
      - 'summary'     : lista metriche per replica
      - 'trace_first' : trace della prima replica
      - 'traces_all'  : lista di tutte le trace (per moda/spaghetti)
    """
    import utils.constants as cs_local
    from libraries.rngs import plantSeeds

    if R is None:
        R = int(getattr(cs_local, "REPLICATIONS", 100))
    if base_seed is None:
        base_seed = int(getattr(cs_local, "SEED", 1))

    J_REP = 10 ** 10
    all_rows = []
    trace_first = None
    traces_all = []
    edge_series_all = []
    series_all = []

    for r in range(R):
        seed_r = lehmer_replica_seed(base_seed, J_REP, r)
        plantSeeds(seed_r)

        cs_local.EDGE_SERVERS       = int(getattr(cs_local, "EDGE_SERVERS_INIT", getattr(cs_local, "EDGE_SERVERS", 1)))
        cs_local.COORD_EDGE_SERVERS = int(getattr(cs_local, "COORD_EDGE_SERVERS_INIT", getattr(cs_local, "COORD_EDGE_SERVERS", 1)))

        res = run_single_replication_24h(forced_lambda=None, slot_index=None)
        res["seed"] = seed_r

        if trace_first is None:
            trace_first = list(res.get("slo_scal_trace", []))
        traces_all.append(list(res.get("slo_scal_trace", [])))
        series_all.append(list(res.get("edge_wait_times_series", [])))

        row = {k: v for k, v in res.items() if k != "slo_scal_trace"}
        all_rows.append(row)

        step = max(1, R // 10)
        if (r == 0) or ((r + 1) % step == 0) or (r + 1 == R):
            print(f"[SLO] Repliche completate: {r + 1}/{R}", flush=True)

    return {"summary": all_rows, "trace_first": trace_first, "traces_all": traces_all, "series_all": series_all}

__all__ = ["run_single_replication_24h", "run_finite_day_replications", "lehmer_replica_seed"]
