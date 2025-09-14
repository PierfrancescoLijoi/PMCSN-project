# simulation/slo_scalability_simulator.py
# ---------------------------------------------------------
# Simulatore EVENT-DRIVEN a orizzonte FINITO (24h) con numero FISSO di server.
# Edge: 2 server fissi, Coordinator: 1 server fisso.
# Nessun autoscaling, solo tracciamento metriche.
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

def day_lambda(t: float, forced_lambda=None) -> float:
    """
    λ(t) periodico su base giornaliera.
    - Se forced_lambda è dato, restituisce quello (clamp > 0).
    - Altrimenti usa cs.LAMBDA_SLOTS (lista di tuple: (start, end, lambda)),
      ripetendo le fasce ogni giorno (wrap su DAY_SECONDS).
    """
    if forced_lambda is not None:
        return max(1e-12, float(forced_lambda))

    DAY = float(getattr(cs, "DAY_SECONDS", 86400.0))
    slots = getattr(cs, "LAMBDA_SLOTS", [])
    if not slots:
        return max(1e-12, float(getattr(cs, "LAMBDA", 1.0)))

    # tempo interno al giorno (wrap ogni 24h)
    tt = float(t) % DAY

    for start, end, lam in slots:
        s = float(start); e = float(end); l = float(lam)

        # Slot normale: [s, e)
        if e > s and (s <= tt < e):
            return max(1e-12, l)

        # Slot che attraversa la mezzanotte: [s, DAY) ∪ [0, e)
        if e <= s and ((s <= tt < DAY) or (0.0 <= tt < e)):
            return max(1e-12, l)

    # Fallback: se nessuno slot ha fatto match, usa LAMBDA di default
    return max(1e-12, float(getattr(cs, "LAMBDA", 1.0)))

# -------------------------
# Singola replica (24h) — numero fisso di server
# -------------------------

def run_single_replication_48h(forced_lambda=None, slot_index=None) -> Dict:
    """
    Replica a orizzonte FINITO con λ(t) giornaliero (o forced_lambda).
    Versione a server FISSI (Edge=2, Coordinator=1), senza autoscaling.
    Compatibile con il resto del file e con la logica di execute() del simulatore standard:
    - transiente/dump ogni 1000s
    - routing E -> Cloud con P_C, altrimenti verso Coordinator con P1..P4 condizionate
    - Cloud come ∞-server con min-heap dei completamenti
    """
    reset_arrival_temp()

    stats = SimulationStats()
    stats.reset(cs.START)

    # --- Parametri fissi ---
    stop = cs.STOP if forced_lambda is None else float(getattr(cs, "STOP", cs.STOP))

    # Pool server fissi
    cs.EDGE_SERVERS = 2
    cs.EDGE_SERVERS_MAX = 2

    cs.COORD_EDGE_SERVERS = 1
    cs.COORD_SERVERS_MAX = 1

    # Stato server Edge
    edge_completion_times = [INFTY] * cs.EDGE_SERVERS_MAX
    edge_server_busy      = [False] * cs.EDGE_SERVERS_MAX
    edge_server_jobtype   = [None]  * cs.EDGE_SERVERS_MAX
    edge_server_arrival   = [None]  * cs.EDGE_SERVERS_MAX  # per sojourn

    # Stato server Coordinator
    coord_completion_times = [INFTY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False] * cs.COORD_SERVERS_MAX

    # Code e heap Cloud
    stats.queue_edge = []           # lista di tuple: ("E"|"C", t_arr_edge)
    stats.queue_coord_high = []     # P3-P4
    stats.queue_coord_low  = []     # P1-P2
    stats.cloud_heap = []           # min-heap completamenti Cloud

    # Primo arrivo
    lam0 = day_lambda(stats.t.current, forced_lambda)
    stats.t.arrival = GetArrival(stats.t.current, lam0)
    if stats.t.arrival > stop:
        stats.t.arrival = INFTY

    # Helper: assegnazioni locali
    def edge_assign_if_possible(sidx: int):
        if not edge_server_busy[sidx] and stats.queue_edge:
            job, t_arr_edge = stats.queue_edge.pop(0)
            if job == "E":
                service = GetServiceEdgeE()
                stats.area_E.service += service
            else:
                service = GetServiceEdgeC()
                stats.area_C.service += service
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx]    = True
            edge_server_jobtype[sidx] = job
            edge_server_arrival[sidx] = t_arr_edge
            stats.area_edge.service  += service
            return True
        return False

    def coord_assign_if_possible(sidx: int):
        if not coord_server_busy[sidx]:
            if stats.queue_coord_high:
                _ = stats.queue_coord_high.pop(0)
                service = GetServiceCoordP3P4()
            elif stats.queue_coord_low:
                _ = stats.queue_coord_low.pop(0)
                service = GetServiceCoordP1P2()
            else:
                return False
            coord_completion_times[sidx] = stats.t.current + service
            coord_server_busy[sidx] = True
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
            if not (stats.queue_coord_high or stats.queue_coord_low):
                break
            coord_assign_if_possible(i)

    # --- DUMP transiente come in execute(): ogni 1000s ---
    interval = 1000.0  # <— identico a simulator.execute
    if not hasattr(stats, "_next_dump"):
        stats._next_dump = 0.0

    def transient_dump_if_needed():
        # medie calcolate come in execute(): area.node / index
        while stats.t.current >= stats._next_dump:
            avg_edge  = stats.area_edge.node  / stats.index_edge  if stats.index_edge  > 0 else 0.0
            avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
            avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0

            stats.edge_wait_times.append((stats._next_dump, avg_edge))
            stats.cloud_wait_times.append((stats._next_dump, avg_cloud))
            stats.coord_wait_times.append((stats._next_dump, avg_coord))

            # breakdown per classi all'Edge, coerente con execute()
            avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
            avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
            stats.edge_E_wait_times_interval.append((stats._next_dump, avg_edge_E))
            stats.edge_C_wait_times_interval.append((stats._next_dump, avg_edge_C))

            stats._next_dump += interval

    # --- Loop principale eventi ---
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        next_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else INFTY
        next_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else INFTY
        next_cloud = stats.cloud_heap[0] if stats.cloud_heap else INFTY

        stats.t.next = Min(stats.t.arrival, next_edge, next_cloud, next_coord)

        # --- Aree (globali + classi Edge), come execute()
        dt = stats.t.next - stats.t.current
        if dt < 0:
            dt = 0.0
        if stats.number_edge  > 0: stats.area_edge.node  += dt * stats.number_edge
        if stats.number_cloud > 0: stats.area_cloud.node += dt * stats.number_cloud
        if stats.number_coord > 0: stats.area_coord.node += dt * stats.number_coord
        if stats.number_E     > 0: stats.area_E.node     += dt * stats.number_E
        if stats.number_C     > 0: stats.area_C.node     += dt * stats.number_C

        stats.t.current = stats.t.next

        # --- Dump transiente ogni 1000s
        transient_dump_if_needed()

        # ========== EVENTI ==========
        # ARRIVAL (arrivo esterno, sempre classe "E" all'Edge)
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E    += 1
            stats.queue_edge.append(("E", stats.t.current))

            # prossimo arrivo con day_lambda (wrap giornaliero) o forced
            lam = day_lambda(stats.t.current, forced_lambda)
            stats.t.arrival = GetArrival(stats.t.current, lam)
            if stats.t.arrival > stop:
                stats.t.arrival = INFTY

            # se ci sono server liberi, parti subito
            kick_assign_all()
            continue

        # EDGE COMPLETION (uno o più server possono completare su questo istante)
        if stats.t.current == next_edge:
            for i in range(cs.EDGE_SERVERS):
                if stats.t.current == edge_completion_times[i]:
                    edge_server_busy[i]    = False
                    job_type               = edge_server_jobtype[i]
                    t_arr_edge             = edge_server_arrival[i]
                    edge_server_jobtype[i] = None
                    edge_server_arrival[i] = None
                    edge_completion_times[i] = INFTY

                    # contatori e popolazioni
                    stats.index_edge  += 1
                    stats.number_edge -= 1

                    if job_type == "E":
                        stats.index_edge_E += 1
                        stats.number_E     -= 1

                        # routing: Cloud con P_C, altrimenti Coordinator (P1..P4 condizionate)
                        selectStream(3)
                        r = rng_random()

                        if r < cs.P_C:
                            # → Cloud (∞-server): ogni job schedula il suo completamento nel min-heap
                            stats.number_cloud += 1
                            service = GetServiceCloud()
                            heapq.heappush(stats.cloud_heap, stats.t.current + service)
                            stats.area_cloud.service += service
                        else:
                            # → Coordinator
                            stats.number_coord += 1
                            denom = max(1e-12, 1.0 - cs.P_C)
                            coord_r = (r - cs.P_C) / denom
                            if coord_r < cs.P1_PROB:
                                stats.queue_coord_low.append("P1")
                                stats.count_E_P1 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB:
                                stats.queue_coord_low.append("P2")
                                stats.count_E_P2 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                                stats.queue_coord_high.append("P3")
                                stats.count_E_P3 += 1
                            else:
                                stats.queue_coord_high.append("P4")
                                stats.count_E_P4 += 1
                            # prova ad avviare subito il Coordinator se libero
                            for j in range(cs.COORD_EDGE_SERVERS):
                                if coord_assign_if_possible(j):
                                    break
                    else:
                        # job di ritorno dal Cloud (classe "C")
                        stats.index_edge_C += 1
                        stats.number_C     -= 1

            # prova ad assegnare nuovi job agli Edge liberi
            kick_assign_all()
            continue

        # CLOUD COMPLETION (min-heap; possono completare più job in questo istante)
        if stats.cloud_heap and stats.t.current == stats.cloud_heap[0]:
            while stats.cloud_heap and stats.cloud_heap[0] <= stats.t.current + 1e-12:
                heapq.heappop(stats.cloud_heap)
                stats.index_cloud  += 1
                stats.number_cloud -= 1
                # ritorno all'Edge come classe "C"
                stats.number_edge += 1
                stats.number_C    += 1
                stats.queue_edge.append(("C", stats.t.current))
            kick_assign_all()
            continue

        # COORDINATOR COMPLETION
        if stats.t.current == next_coord:
            for i in range(cs.COORD_EDGE_SERVERS):
                if stats.t.current == coord_completion_times[i]:
                    coord_server_busy[i] = False
                    coord_completion_times[i] = INFTY
                    stats.index_coord  += 1
                    stats.number_coord -= 1
                    # se c'è altro in coda, parte subito
                    if not coord_assign_if_possible(i):
                        kick_assign_all()
                    break

    # --- Fine simulazione: metriche globali (come in versione originale a server fissi) ---
    stats.calculate_area_queue()
    sim_time = max(1e-12, stats.t.current - cs.START)

    edge_W  = (stats.area_edge.node  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    edge_Wq = (stats.area_edge.queue / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    edge_L  = stats.area_edge.node / sim_time
    edge_Lq = stats.area_edge.queue / sim_time
    edge_S  = (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0
    edge_X  = stats.index_edge / sim_time
    edge_util = (stats.area_edge.service / sim_time) if sim_time > 0 else 0.0

    edge_E_W  = (stats.area_E.node  / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq = (stats.area_E.queue / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_L  = stats.area_E.node / sim_time
    edge_E_Lq = stats.area_E.queue / sim_time
    edge_E_S  = (stats.area_E.service / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_util = (stats.area_E.service / sim_time) if sim_time > 0 else 0.0

    edge_C_W  = (stats.area_C.node  / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq = (stats.area_C.queue / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_L  = stats.area_C.node / sim_time
    edge_C_Lq = stats.area_C.queue / sim_time
    edge_C_S  = (stats.area_C.service / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_util = (stats.area_C.service / sim_time) if sim_time > 0 else 0.0

    cloud_W = (stats.area_cloud.node / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    cloud_L = stats.area_cloud.node / sim_time

    coord_W = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0
    coord_L = stats.area_coord.node / sim_time
    coord_util = (stats.area_coord.service / sim_time) if sim_time > 0 else 0.0
    coord_X = stats.index_coord / sim_time

    results = {
        "seed": getSeed(),
        "lambda": forced_lambda if forced_lambda is not None else None,
        "slot": slot_index,

        "edge_server_number": cs.EDGE_SERVERS,
        "edge_avg_wait": edge_W,
        "edge_avg_delay": edge_Wq,
        "edge_L": edge_L,
        "edge_Lq": edge_Lq,
        "edge_utilization": edge_util,
        "edge_throughput": edge_X,

        "edge_E_avg_wait": edge_E_W,
        "edge_E_avg_delay": edge_E_Wq,
        "edge_E_L": edge_E_L,
        "edge_E_Lq": edge_E_Lq,
        "edge_E_utilization": edge_E_util,

        "edge_C_avg_wait": edge_C_W,
        "edge_C_avg_delay": edge_C_Wq,
        "edge_C_L": edge_C_L,
        "edge_C_Lq": edge_C_Lq,
        "edge_C_utilization": edge_C_util,

        "cloud_avg_wait": cloud_W,
        "cloud_L": cloud_L,

        "coord_avg_wait": coord_W,
        "coord_L": coord_L,
        "coord_utilization": coord_util,
        "coord_throughput": coord_X,

        "pc": cs.P_C,
        "p1": cs.P1_PROB,
        "p2": cs.P2_PROB,
        "p3": cs.P3_PROB,
        "p4": cs.P4_PROB,

        "slo_scal_trace": [],
    }
    # Serie per i grafici (identiche all'altra versione)
    results["edge_wait_times_series"] = list(stats.edge_wait_times)
    results["cloud_wait_times_series"] = list(stats.cloud_wait_times)
    results["coord_wait_times_series"] = list(stats.coord_wait_times)
    results["edge_E_wait_times_interval"] = list(stats.edge_E_wait_times_interval)
    results["edge_C_wait_times_interval"] = list(stats.edge_C_wait_times_interval)
    return results


# -------------------------
# Repliche con semi indipendenti (Lehmer jump) - versione fixed server
# -------------------------
def run_finite_day_replications(R: int = None, base_seed: int = None) -> Dict:
    """
    Esegue R repliche indipendenti con server fissi: Edge=2, Coordinator=1
    """
    import utils.constants as cs_local
    from libraries.rngs import plantSeeds

    if R is None:
        R = int(getattr(cs_local, "REPLICATIONS", 100))
    if base_seed is None:
        base_seed = int(getattr(cs_local, "SEED", 1))

    J_REP = 10 ** 10
    all_rows = []
    series_all = []

    for r in range(R):
        seed_r = lehmer_replica_seed(base_seed, J_REP, r)
        plantSeeds(seed_r)

        res = run_single_replication_48h(forced_lambda=None, slot_index=None)
        res["seed"] = seed_r

        series_all.append(list(res.get("edge_E_wait_times_interval", [])))

        row = {k: v for k, v in res.items() if not k.endswith("_series") and k != "edge_wait_times_interval"}
        all_rows.append(row)

        step = max(1, R // 10)
        if (r == 0) or ((r + 1) % step == 0) or (r + 1 == R):
            print(f"[FIXED] Repliche completate: {r + 1}/{R}", flush=True)

    return {"summary": all_rows, "series_all": series_all}


__all__ = ["run_single_replication_48h", "run_finite_day_replications", "lehmer_replica_seed"]