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
from libraries.rngs import getSeed, selectStream, random as rng_random, plantSeeds
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
    Server fissi: Edge=2, Coord=1. Raccoglie metriche per (giorno, fascia λ)
    e le espone in 'by_day_slot_rows' per scrittura CSV.
    """
    reset_arrival_temp()
    stats = SimulationStats()
    stats.reset(cs.START)

    stop = float(getattr(cs, "STOP", cs.STOP))
    DAY  = float(getattr(cs, "DAY_SECONDS", 86400.0))
    slots = list(getattr(cs, "LAMBDA_SLOTS", []))  # [(start,end,λ)] definiti su 0..DAY

    # pool fissi
    cs.EDGE_SERVERS = 2
    cs.EDGE_SERVERS_MAX = 2
    cs.COORD_EDGE_SERVERS = 1
    cs.COORD_SERVERS_MAX = 1

    # stato server
    edge_completion_times = [INFTY] * cs.EDGE_SERVERS_MAX
    edge_server_busy      = [False] * cs.EDGE_SERVERS_MAX
    edge_server_jobtype   = [None]  * cs.EDGE_SERVERS_MAX
    edge_server_arrival   = [None]  * cs.EDGE_SERVERS_MAX

    coord_completion_times = [INFTY] * cs.COORD_SERVERS_MAX
    coord_server_busy      = [False] * cs.COORD_SERVERS_MAX

    stats.queue_edge = []
    stats.queue_coord_high = []
    stats.queue_coord_low  = []
    stats.cloud_heap = []

    # --- bounding e bucket per (giorno, slot) ---
    import math, heapq
    abs_bounds = {0.0, float(stop)}
    bucket_info = {}  # (day,slot) -> [(a,b,lam), ...] in tempo assoluto
    for day_idx in range(int(math.ceil(stop / DAY))):
        base = day_idx * DAY
        for sidx, (s, e, lam) in enumerate(slots):
            s = float(s); e = float(e); lam = float(lam)
            if e > s:
                segs = [(base + s, base + e)]
            else:
                segs = [(base + s, base + DAY), (base + 0.0, base + e)]
            for a, b in segs:
                if a >= stop:   continue
                b = min(b, stop)
                if b <= 0:      continue
                bucket_info.setdefault((day_idx, sidx), []).append((a, b, lam))
                abs_bounds.add(a); abs_bounds.add(b)
    abs_bounds = sorted(abs_bounds)

    from collections import defaultdict
    bucket_area = defaultdict(lambda: {
        "edge_node":0.0,"cloud_node":0.0,"coord_node":0.0,
        "E_node":0.0,"C_node":0.0,
        "edge_serv":0.0,"cloud_serv":0.0,"coord_serv":0.0,
        "idx_edge":0,"idx_cloud":0,"idx_coord":0,"idx_E":0,"idx_C":0
    })
    bucket_span = {}  # (day,slot,seg_idx) -> durata segmento

    def locate_bucket(t_now: float):
        for (day_idx, sidx), segs in bucket_info.items():
            for i, (a, b, lam) in enumerate(segs):
                if a <= t_now < b or (abs(t_now-b) < 1e-12 and b == stop):
                    return (day_idx, sidx, i, a, b, lam)
        return (None, None, None, None, None, None)

    for (day_idx, sidx), segs in bucket_info.items():
        for i, (a, b, _lam) in enumerate(segs):
            bucket_span[(day_idx, sidx, i)] = max(0.0, b - a)

    # primo arrivo
    lam0 = day_lambda(stats.t.current, forced_lambda)
    stats.t.arrival = GetArrival(stats.t.current, lam0)
    if stats.t.arrival > stop: stats.t.arrival = INFTY

    def edge_assign_if_possible(sidx: int):
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
            # bucket: servizio edge
            loc = locate_bucket(stats.t.current)
            if loc[0] is not None:
                bucket_area[(loc[0], loc[1], loc[2])]["edge_serv"] += service
            return True
        return False

    def coord_assign_if_possible(sidx: int):
        if coord_server_busy[sidx]: return False
        if stats.queue_coord_high:
            _ = stats.queue_coord_high.pop(0); service = GetServiceCoordP3P4()
        elif stats.queue_coord_low:
            _ = stats.queue_coord_low.pop(0);  service = GetServiceCoordP1P2()
        else:
            return False
        coord_completion_times[sidx] = stats.t.current + service
        coord_server_busy[sidx] = True
        stats.area_coord.service += service
        loc = locate_bucket(stats.t.current)
        if loc[0] is not None:
            bucket_area[(loc[0], loc[1], loc[2])]["coord_serv"] += service
        return True

    def kick_assign_all():
        for i in range(cs.EDGE_SERVERS):
            if not stats.queue_edge: break
            edge_assign_if_possible(i)
        for i in range(cs.COORD_EDGE_SERVERS):
            if not (stats.queue_coord_high or stats.queue_coord_low): break
            coord_assign_if_possible(i)

    # dump transiente (1000s)
    interval = 1000.0
    if not hasattr(stats, "_next_dump"):
        stats._next_dump = 0.0
    def transient_dump_if_needed():
        while stats.t.current >= stats._next_dump:
            avg_edge  = stats.area_edge.node  / stats.index_edge  if stats.index_edge  > 0 else 0.0
            avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
            avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
            stats.edge_wait_times.append((stats._next_dump, avg_edge))
            stats.cloud_wait_times.append((stats._next_dump, avg_cloud))
            stats.coord_wait_times.append((stats._next_dump, avg_coord))
            avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
            avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
            stats.edge_E_wait_times_interval.append((stats._next_dump, avg_edge_E))
            stats.edge_C_wait_times_interval.append((stats._next_dump, avg_edge_C))
            stats._next_dump += interval

    # loop eventi con ripartizione delle aree su confini (giorni/fasce)
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        next_edge  = min(edge_completion_times[:cs.EDGE_SERVERS]) if cs.EDGE_SERVERS > 0 else INFTY
        next_coord = min(coord_completion_times[:cs.COORD_EDGE_SERVERS]) if cs.COORD_EDGE_SERVERS > 0 else INFTY
        next_cloud = stats.cloud_heap[0] if stats.cloud_heap else INFTY
        t_next = Min(stats.t.arrival, next_edge, next_cloud, next_coord)

        # integra per step tra confini assoluti
        t_curr = stats.t.current
        while t_curr < t_next - 1e-12:
            bound_after = next((b for b in abs_bounds if b - t_curr > 1e-12), t_next)
            t_seg_end = min(bound_after, t_next)
            dt = max(0.0, t_seg_end - t_curr)

            if stats.number_edge  > 0: stats.area_edge.node  += dt * stats.number_edge
            if stats.number_cloud > 0: stats.area_cloud.node += dt * stats.number_cloud
            if stats.number_coord > 0: stats.area_coord.node += dt * stats.number_coord
            if stats.number_E     > 0: stats.area_E.node     += dt * stats.number_E
            if stats.number_C     > 0: stats.area_C.node     += dt * stats.number_C

            loc = locate_bucket(t_curr)
            if loc[0] is not None:
                key = (loc[0], loc[1], loc[2])
                if stats.number_edge  > 0: bucket_area[key]["edge_node"]  += dt * stats.number_edge
                if stats.number_cloud > 0: bucket_area[key]["cloud_node"] += dt * stats.number_cloud
                if stats.number_coord > 0: bucket_area[key]["coord_node"] += dt * stats.number_coord
                if stats.number_E     > 0: bucket_area[key]["E_node"]     += dt * stats.number_E
                if stats.number_C     > 0: bucket_area[key]["C_node"]     += dt * stats.number_C
            t_curr = t_seg_end

        stats.t.current = t_next
        transient_dump_if_needed()

        # ARRIVAL
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E    += 1
            stats.queue_edge.append(("E", stats.t.current))
            lam = day_lambda(stats.t.current, forced_lambda)
            stats.t.arrival = GetArrival(stats.t.current, lam)
            if stats.t.arrival > stop: stats.t.arrival = INFTY
            kick_assign_all()
            continue

        # EDGE COMPLETION
        if stats.t.current == next_edge:
            for i in range(cs.EDGE_SERVERS):
                if stats.t.current == edge_completion_times[i]:
                    edge_server_busy[i]    = False
                    job_type               = edge_server_jobtype[i]
                    edge_server_jobtype[i] = None
                    edge_completion_times[i] = INFTY
                    stats.index_edge  += 1
                    stats.number_edge -= 1
                    loc = locate_bucket(stats.t.current)
                    if loc[0] is not None:
                        bucket_area[(loc[0], loc[1], loc[2])]["idx_edge"] += 1

                    if job_type == "E":
                        stats.index_edge_E += 1
                        stats.number_E     -= 1
                        if loc[0] is not None:
                            bucket_area[(loc[0], loc[1], loc[2])]["idx_E"] += 1
                        from libraries.rngs import selectStream, random as rng_random
                        selectStream(3); r = rng_random()
                        if r < cs.P_C:
                            stats.number_cloud += 1
                            service = GetServiceCloud()
                            heapq.heappush(stats.cloud_heap, stats.t.current + service)
                            stats.area_cloud.service += service
                            if loc[0] is not None:
                                bucket_area[(loc[0], loc[1], loc[2])]["cloud_serv"] += service
                        else:
                            stats.number_coord += 1
                            denom = max(1e-12, 1.0 - cs.P_C)
                            coord_r = (r - cs.P_C) / denom
                            if coord_r < cs.P1_PROB:
                                stats.queue_coord_low.append("P1"); stats.count_E_P1 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB:
                                stats.queue_coord_low.append("P2"); stats.count_E_P2 += 1
                            elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                                stats.queue_coord_high.append("P3"); stats.count_E_P3 += 1
                            else:
                                stats.queue_coord_high.append("P4"); stats.count_E_P4 += 1
                            for j in range(cs.COORD_EDGE_SERVERS):
                                if coord_assign_if_possible(j):
                                    break
                    else:
                        stats.index_edge_C += 1
                        stats.number_C     -= 1
                        if loc[0] is not None:
                            bucket_area[(loc[0], loc[1], loc[2])]["idx_C"] += 1
            kick_assign_all()
            continue

        # CLOUD COMPLETION
        if stats.cloud_heap and stats.t.current == stats.cloud_heap[0]:
            while stats.cloud_heap and stats.cloud_heap[0] <= stats.t.current + 1e-12:
                heapq.heappop(stats.cloud_heap)
                stats.index_cloud  += 1
                stats.number_cloud -= 1
                loc = locate_bucket(stats.t.current)
                if loc[0] is not None:
                    bucket_area[(loc[0], loc[1], loc[2])]["idx_cloud"] += 1
                # rientra all'Edge come "C"
                stats.number_edge += 1
                stats.number_C    += 1
                stats.queue_edge.append(("C", stats.t.current))
            kick_assign_all()
            continue

        # COORD COMPLETION
        if stats.t.current == next_coord:
            for i in range(cs.COORD_EDGE_SERVERS):
                if stats.t.current == coord_completion_times[i]:
                    coord_server_busy[i] = False
                    coord_completion_times[i] = INFTY
                    stats.index_coord  += 1
                    stats.number_coord -= 1
                    loc = locate_bucket(stats.t.current)
                    if loc[0] is not None:
                        bucket_area[(loc[0], loc[1], loc[2])]["idx_coord"] += 1
                    if not coord_assign_if_possible(i):
                        kick_assign_all()
                    break

    # metriche globali (come già avevi)
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
    edge_C_W  = (stats.area_C.node  / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    cloud_W = (stats.area_cloud.node / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    coord_W = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0
    cloud_L = stats.area_cloud.node / sim_time
    coord_L = stats.area_coord.node / sim_time
    coord_util = (stats.area_coord.service / sim_time) if sim_time > 0 else 0.0
    coord_X = stats.index_coord / sim_time

    # === costruzione righe CSV per (giorno, fascia) ===
    per_day_slot = {}
    for (day_idx, sidx), segs in bucket_info.items():
        for i, (a, b, lam) in enumerate(segs):
            span = max(0.0, b - a)
            if span <= 0: continue
            key3 = (day_idx, sidx, i)
            acc = bucket_area.get(key3)
            if not acc: continue
            tgt = per_day_slot.setdefault((day_idx, sidx), {"span":0.0, **{k:0.0 for k in acc}})
            tgt["span"] += span
            for k, v in acc.items():
                tgt[k] += v

    # λ per fascia
    slot_lambda = {}
    for (day_idx, sidx), segs in bucket_info.items():
        if segs:
            slot_lambda[(day_idx, sidx)] = float(segs[0][2])

    by_day_slot_rows = []
    for (day_idx, sidx), acc in sorted(per_day_slot.items()):
        T = max(1e-12, acc["span"])
        edge_L_bs  = acc["edge_node"] / T
        edge_W_bs  = (acc["edge_node"] / acc["idx_edge"]) if acc["idx_edge"] > 0 else 0.0
        edge_E_W_bs = (acc["E_node"] / acc["idx_E"]) if acc["idx_E"] > 0 else 0.0
        edge_C_W_bs = (acc["C_node"] / acc["idx_C"]) if acc["idx_C"] > 0 else 0.0
        cloud_W_bs  = (acc["cloud_node"] / acc["idx_cloud"]) if acc["idx_cloud"] > 0 else 0.0
        coord_W_bs  = (acc["coord_node"] / acc["idx_coord"]) if acc["idx_coord"] > 0 else 0.0

        by_day_slot_rows.append({
            "seed": getSeed(),
            "day": int(day_idx + 1),
            "day_label": ("giorno1" if day_idx == 0 else "giorno2"),
            "slot": int(sidx),
            "lambda": slot_lambda.get((day_idx, sidx)),
            "pc": cs.P_C,
            "p1": cs.P1_PROB, "p2": cs.P2_PROB, "p3": cs.P3_PROB, "p4": cs.P4_PROB,
            "edge_avg_wait": edge_W_bs,
            "edge_E_avg_wait": edge_E_W_bs,
            "edge_C_avg_wait": edge_C_W_bs,
            "cloud_avg_wait": cloud_W_bs,
            "coord_avg_wait": coord_W_bs,
            "edge_L": edge_L_bs,
            "span_seconds": T,
        })

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
        "edge_C_avg_wait": edge_C_W,
        "cloud_avg_wait": cloud_W,
        "coord_avg_wait": coord_W,
        "cloud_L": cloud_L,
        "coord_L": coord_L,
        "coord_utilization": coord_util,
        "coord_throughput": coord_X,
        "pc": cs.P_C, "p1": cs.P1_PROB, "p2": cs.P2_PROB, "p3": cs.P3_PROB, "p4": cs.P4_PROB,
        "edge_wait_times_series": list(stats.edge_wait_times),
        "cloud_wait_times_series": list(stats.cloud_wait_times),
        "coord_wait_times_series": list(stats.coord_wait_times),
        "edge_E_wait_times_interval": list(stats.edge_E_wait_times_interval),
        "edge_C_wait_times_interval": list(stats.edge_C_wait_times_interval),
        "by_day_slot_rows": by_day_slot_rows,   # <-- nuovo
    }
    return results


# -------------------------
# Repliche con semi indipendenti (Lehmer jump) - versione fixed server
# -------------------------
def run_finite_day_replications(R: int = None, base_seed: int = None) -> Dict:
    """
    Esegue R repliche 48h con λ(t) giornaliero.
    Ritorna:
      - summary      : 1 riga per replica (metriche globali)
      - series_all   : serie per plot
      - by_day_slot_rows : N righe per (replica, giorno, fascia)
    """
    if R is None: R = int(getattr(cs, "REPLICATIONS", 5))
    if base_seed is None: base_seed = int(getattr(cs, "BASE_SEED", 12345))

    all_rows, series_all, by_slot_rows_all = [], [], []
    J_REP = 10 ** 10
    for r in range(R):
        seed_r = lehmer_replica_seed(base_seed, J_REP, r)
        plantSeeds(seed_r)

        res = run_single_replication_48h(forced_lambda=None, slot_index=None)

        # serie per plot
        series_all.append(list(res.get("edge_E_wait_times_interval", [])))

        # sommario per replica (compatibilità)
        row = {k: v for k, v in res.items()
               if not k.endswith("_series") and k not in ("edge_wait_times_interval", "by_day_slot_rows")}
        row["replication"] = r
        all_rows.append(row)

        # righe per CSV (giorno/fascia)
        for br in res.get("by_day_slot_rows", []):
            br2 = dict(br); br2["replication"] = r
            by_slot_rows_all.append(br2)

        step = max(1, R // 10)
        if (r == 0) or ((r + 1) % step == 0) or (r + 1 == R):
            print(f"[FIXED] Repliche completate: {r + 1}/{R}", flush=True)

    return {"summary": all_rows, "series_all": series_all, "by_day_slot_rows": by_slot_rows_all}

__all__ = ["run_single_replication_48h", "run_finite_day_replications", "lehmer_replica_seed"]