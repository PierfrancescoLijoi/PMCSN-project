"""
main.py
---------
Questo è il file principale che avvia la simulazione per il modello standard
descritto nel documento PMCSN Project (Luglio 2025).

Riferimento: Sezione "Modello computazionale" del documento.
La simulazione segue un approccio next-event-driven con orizzonte finito
(transiente), come definito a pagina 8–10 del testo allegato.
"""
import os
import pandas as pd
import utils.constants as cs
import time

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from simulation.edge_cord_merged_scalability_simulator import edge_coord_scalability_simulation
from simulation.simulator import finite_simulation, infinite_simulation
from utils.simulation_output import write_file, clear_file, print_simulation_stats, plot_multi_seed_per_lambda, \
    write_infinite_row, write_file_merged_scalability, clear_merged_scalability_file, clear_infinite_file, \
    plot_infinite_analysis, plot_edge_response_vs_pc, plot_analysis, plot_transient_with_seeds
from utils.simulation_stats import ReplicationStats
from simulation.improved_edge_cord_merged_scalability_simulator import edge_coord_scalability_simulation_improved
from simulation.improved_simulator import finite_simulation_improved, infinite_simulation_improved
from utils.improved_simulation_output import write_file_improved, clear_file_improved, print_simulation_stats_improved, \
    plot_multi_seed_per_lambda_improved, \
    write_infinite_row_improved, write_file_merged_scalability_improved, clear_merged_scalability_file_improved, \
    clear_infinite_file_improved, plot_infinite_analysis_improved
from utils.sim_utils import append_stats, calculate_confidence_interval, set_pc_and_update_probs, append_stats_improved, \
    lehmer_replica_seed
from utils.improved_simulation_stats import ReplicationStats_improved
from libraries.rngs import plantSeeds, getSeed



def start_finite_simulation():
    lam = cs.LAMBDA          # job/sec
    stop = cs.STOP           # es. 86400 sec (24h)
    replicationStats = ReplicationStats()
    print("FINITE SIMULATION - Aeroporto Ciampino")

    file_name = os.path.join("output/", "finite_statistics.csv")
    clear_file(file_name)

    J_REP = 10 ** 10  # salto per separare le repliche (sceglilo grande)

    # Esegui le repliche
    for rep in range(cs.REPLICATIONS):
        # seed indipendente per replica
        seed = lehmer_replica_seed(cs.SEED, J_REP, rep)
        plantSeeds(seed)

        # 1 replica a orizzonte finito
        results, stats = finite_simulation(stop, forced_lambda=lam)

        # traccia il seed usato (solo per CSV/riassunti; NON entra in legenda)
        results['seed'] = seed

        # salva riga CSV e accumula statistiche aggregate
        write_file(results, file_name)
        append_stats(replicationStats, results, stats)  # <-- qui vengono salvate anche le serie (t, W)

        if (rep + 1) % 10 == 0:
            print(f"  -> completate {rep + 1} repliche")

    # Stampa riassuntiva (media ± CI)
    print_simulation_stats(replicationStats, "finite_fixed_lambda")

    # === Grafici transiente: SEMPRE con wait_times (serie [(t, W)]) ===
    # Una curva per replica, x = tempo simulazione (s), y = tempo medio di risposta
    plot_analysis(replicationStats.edge_wait_interval,  replicationStats.seeds, "edge_node",               sim_type="finite_fixed_lambda")
    plot_analysis(replicationStats.edge_E_wait_interval, replicationStats.seeds, "edge_node_E",
                  sim_type="finite_fixed_lambda")
    plot_analysis(replicationStats.cloud_wait_interval, replicationStats.seeds, "cloud_node",              sim_type="finite_fixed_lambda")
    plot_analysis(replicationStats.coord_wait_interval, replicationStats.seeds, "coordinator_server_edge", sim_type="finite_fixed_lambda")

    return replicationStats


def start_transient_analysis():
    """
    Analisi del transitorio con 20 repliche (nessun CSV).
    Usa finite_simulation() per riempire le serie (t, W) e
    plotta con legenda 'Seed ...' in output/plot/transient_analysis/finite_fixed_lambda/.
    """
    lam = cs.LAMBDA
    stop = cs.STOP_ANALYSIS
    reps = int(getattr(cs, "TRANSIENT_REPLICATIONS", 20))

    print(f"\nTRANSIENT ANALYSIS — repliche={reps}, λ={lam:.5f}, stop={stop}s")

    seeds = []
    edge_series, cloud_series, coord_series = [], [], []
    edge_E_series, edge_C_series = [], []

    # salti ampi tra repliche con Lehmer per indipendenza dei seed
    J_REP = 10 ** 10  # grande per separare bene i flussi
    for r in range(reps):
        seed = lehmer_replica_seed(cs.SEED, J_REP, r)
        plantSeeds(seed)

        # singola replica a orizzonte finito (solo serie transiente)
        _results, stats = finite_simulation(stop, forced_lambda=lam)
        seeds.append(seed)

        # serie (t, media cumulata) già raccolte in execute()
        edge_series.append(list(stats.edge_wait_times))
        cloud_series.append(list(stats.cloud_wait_times))
        coord_series.append(list(stats.coord_wait_times))

        # per classi all'Edge
        edge_E_series.append(list(stats.edge_E_wait_times_interval))
        edge_C_series.append(list(stats.edge_C_wait_times_interval))

    # Grafici con legenda dei seed
    sim_label = "finite_fixed_lambda"
    plot_transient_with_seeds(edge_series, seeds, "edge_node", sim_label)
    plot_transient_with_seeds(edge_E_series, seeds, "edge_node_E", sim_label)
    plot_transient_with_seeds(edge_C_series, seeds, "edge_node_C", sim_label)
    plot_transient_with_seeds(cloud_series, seeds, "cloud_node", sim_label)
    plot_transient_with_seeds(coord_series, seeds, "coordinator_server_edge", sim_label)

    print("Grafici scritti in: output/plot/transient_analysis/finite_fixed_lambda/")





def start_infinite_single_simulation():
    print("\nINFINITE SINGLE-LAMBDA SIMULATION")

    file_name = "infinite_statistics.csv"
    clear_infinite_file(file_name)  # header già compatibile con batch-means

    # un’unica simulazione lunga, con batch-means
    plantSeeds(cs.SEED)
    base_seed = getSeed()

    batch_stats = infinite_simulation(forced_lambda=cs.LAMBDA)

    for batch_index, results in enumerate(batch_stats.results):
        results['lambda'] = cs.LAMBDA   # <— unico λ preso dalle costanti
        results['slot'] = 0             # <— unico slot “fittizio”
        results['seed'] = base_seed
        results['batch'] = batch_index
        write_infinite_row(results, file_name)

    print_simulation_stats(batch_stats, "infinite")
    plot_infinite_analysis()
    return batch_stats


def start_scalability_simulation():
    """
    Scalabilità UNIFICATA (Edge + Coordinator) con metodo a repliche:
    - Esegue tutte le repliche per ciascun p_c (in cs.PC_VALUES), poi passa al successivo.
    - Scrive CSV (una riga per replica×slot).
    - Grafico: 1 per p_c (media a gradini del numero di server sulle repliche).
    - Report cumulativo per p_c: min/max/media server, log UP/DOWN con tempo e slot, W ± CI(95%) per Edge/Cloud/Coord.
    """
    file_name = "merged_scalability_statistics.csv"
    clear_merged_scalability_file(file_name)

    output_dir = os.path.join("output", "merged_scalability")
    os.makedirs(output_dir, exist_ok=True)

    slot_duration = getattr(cs, "SLOT_DURATION", 3600.0)
    num_slots = len(cs.LAMBDA_SLOTS) if hasattr(cs, "LAMBDA_SLOTS") else 1
    horizon = slot_duration * num_slots
    decision_interval = float(getattr(cs, "SCALING_WINDOW", 1000.0))
    if decision_interval <= 0:
        decision_interval = slot_duration / 20.0

    pc_values = getattr(cs, "PC_VALUES", [0.1, 0.4, 0.5, 0.7, 0.9])

    print("MERGED (EDGE+COORD) SCALABILITY SIMULATION")

    for pc in pc_values:
        set_pc_and_update_probs(pc)  # imposta cs.P_C, cs.P_COORD e le condizionate P1..P4
        s = cs.P1_PROB + cs.P2_PROB + cs.P3_PROB + cs.P4_PROB

        print(f"\n### p_c = {pc:.2f} (P_COORD={cs.P_COORD:.2f}) | "
              f"P1={cs.P1_PROB:.3f} P2={cs.P2_PROB:.3f} P3={cs.P3_PROB:.3f} P4={cs.P4_PROB:.3f}")

        rep_traces_edge, rep_traces_coord = [], []
        rep_edge_wait_means, rep_coord_wait_means, rep_cloud_wait_means = [], [], []
        edge_servers_all, coord_servers_all = [], []
        edge_scale_events, coord_scale_events = [], []

        J_REP = 10 ** 10  # salto ampio per separare le repliche (come in finite_simulation)

        for rep in range(cs.REPLICATIONS):
            print(f"  ★ Replica {rep + 1}")
            seed = lehmer_replica_seed(cs.SEED, J_REP, rep)  # seed indipendente per replica
            plantSeeds(seed)  # imposta gli stream RNG

            edge_wait_this_rep, coord_wait_this_rep, cloud_wait_this_rep = [], [], []
            t_trace, edge_trace, coord_trace = [], [], []
            t_offset = 0.0
            last_edge_count, last_coord_count = None, None

            for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
                print(f"    ➤ Slot {slot_index} - λ = {lam:.5f} job/sec")
                stop = slot_duration

                res = edge_coord_scalability_simulation(stop=stop, forced_lambda=lam, slot_index=slot_index)
                res["seed"] = seed
                res["lambda"] = lam
                res["slot"] = slot_index
                res["pc"] = pc
                res["p1"], res["p2"], res["p3"], res["p4"] = cs.P1_PROB, cs.P2_PROB, cs.P3_PROB, cs.P4_PROB
                write_file_merged_scalability(res, file_name)

                edge_wait_this_rep.append(res["edge_avg_wait"])
                coord_wait_this_rep.append(res["coord_avg_wait"])
                cloud_wait_this_rep.append(res["cloud_avg_wait"])
                edge_servers_all.append(res["edge_server_number"])
                coord_servers_all.append(res["coord_server_number"])

                for (t_rel, s, _) in res.get("edge_scal_trace", []):
                    t_abs = t_offset + t_rel
                    t_trace.append(t_abs)
                    edge_trace.append(s)
                    if last_edge_count is None:
                        last_edge_count = s
                    elif s != last_edge_count:
                        direction = "UP" if s > last_edge_count else "DOWN"
                        edge_scale_events.append((slot_index, t_abs, direction, s))
                        last_edge_count = s

                for (t_rel, s, _) in res.get("coord_scal_trace", []):
                    t_abs = t_offset + t_rel
                    coord_trace.append(s)
                    if last_coord_count is None:
                        last_coord_count = s
                    elif s != last_coord_count:
                        direction = "UP" if s > last_coord_count else "DOWN"
                        coord_scale_events.append((slot_index, t_abs, direction, s))
                        last_coord_count = s

                t_offset += slot_duration

            if edge_wait_this_rep:
                rep_edge_wait_means.append(sum(edge_wait_this_rep) / len(edge_wait_this_rep))
            if coord_wait_this_rep:
                rep_coord_wait_means.append(sum(coord_wait_this_rep) / len(coord_wait_this_rep))
            if cloud_wait_this_rep:
                rep_cloud_wait_means.append(sum(cloud_wait_this_rep) / len(cloud_wait_this_rep))

            rep_traces_edge.append(list(zip(t_trace, edge_trace)))
            rep_traces_coord.append(list(zip(t_trace[:len(coord_trace)], coord_trace)))

        grid = [i * decision_interval for i in range(int(horizon / decision_interval) + 1)]

        def step_value(trace_tv, t):
            if not trace_tv:
                return 1
            v_last = trace_tv[0][1]
            for (tt, vv) in trace_tv:
                if tt <= t:
                    v_last = vv
                else:
                    break
            return v_last

        from collections import Counter

        def modal_value(values):
            c = Counter(values)
            m = max(c.values())
            # tie-break: tra i più frequenti scegliamo il più piccolo
            return min([v for v, cnt in c.items() if cnt == m])

        mode_edge, mode_coord = [], []
        for t in grid:
            vals_e = [step_value(tr, t) for tr in rep_traces_edge]
            vals_c = [step_value(tr, t) for tr in rep_traces_coord]
            mode_edge.append(modal_value(vals_e) if vals_e else 1)
            mode_coord.append(modal_value(vals_c) if vals_c else 1)

        plt.figure()
        plt.step(grid, mode_edge, where="post", label="Edge servers (moda repliche)")
        plt.step(grid, mode_coord, where="post", label="Coordinator servers (moda repliche)")
        plt.xlabel("Tempo")
        plt.ylabel("Numero server")
        plt.title(f"Andamento server nel tempo – pc={pc:.2f} (moda su {cs.REPLICATIONS} repliche)")
        plt.legend()
        fig_path = os.path.join(output_dir, f"servers_over_time_pc_{str(pc).replace('.', '_')}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grafico scritto: {fig_path}")

        print("\n--- REPORT CUMULATIVO ---")
        if edge_servers_all:
            print(f"Edge servers     -> min: {min(edge_servers_all)}, max: {max(edge_servers_all)}, avg: {sum(edge_servers_all)/len(edge_servers_all):.4f}")
        if coord_servers_all:
            print(f"Coordinator srv  -> min: {min(coord_servers_all)}, max: {max(coord_servers_all)}, avg: {sum(coord_servers_all)/len(coord_servers_all):.4f}")

        m, ci = calculate_confidence_interval(rep_edge_wait_means)
        print(f"Edge avg wait    -> {m:.4f} ± {ci:.4f} (95% CI)")
        m, ci = calculate_confidence_interval(rep_cloud_wait_means)
        print(f"Cloud avg wait   -> {m:.4f} ± {ci:.4f} (95% CI)")
        m, ci = calculate_confidence_interval(rep_coord_wait_means)
        print(f"Coord avg wait   -> {m:.4f} ± {ci:.4f} (95% CI)")

        print("\nEventi scalabilità EDGE (tempo assoluto; slot λ):")
        if edge_scale_events:
            for slot_idx, t_abs, direction, new_count in edge_scale_events:
                print(f"  t={t_abs:.1f}s  slot={slot_idx}  -> {direction} a {new_count} server")
        else:
            print("  Nessun cambiamento")

        print("\nEventi scalabilità COORD (tempo assoluto; slot λ):")
        if coord_scale_events:
            for slot_idx, t_abs, direction, new_count in coord_scale_events:
                print(f"  t={t_abs:.1f}s  slot={slot_idx}  -> {direction} a {new_count} server")
        else:
            print("  Nessun cambiamento")

    print(f"\nCSV scritto: output/{file_name}")


def print_csv_legend():
    print("\n=== CSV Columns Legend ===\n")

    # Notazione (ASCII semplice)
    print("NOTAZIONE:")
    print("  S            = tempo di servizio")
    print("  E(Ts_*)      = tempo di risposta (sojourn) del nodo *")
    print("  E(Tq_*)      = tempo di coda del nodo *")
    print("  E(N_*)       = numero medio nel nodo *")
    print("  E(Nq_*)      = numero medio in coda nel nodo *")
    print("  E(Ns_*)      = numero medio in servizio (server occupati) nel nodo *")
    print("  ρ_* ≈ E(Ns_*)/m_*   |   X_* = completamenti/tempo")
    print("  Suffix di classe: _E = classe E, _C = classe C\n")

    legend = {
        # Identificativi simulazione
        "seed":  "seed RNG (riproducibilità).",
        "slot":  "indice della fascia oraria (slot λ).",
        "lambda":"tasso d’arrivo (job/s).",
        "batch": "indice batch (orizzonte infinito).",

        # --- Tempi (STANDARD) -> *_avg_wait = E(Ts_*), *_avg_delay = E(Tq_*) ---
        "edge_avg_wait":   "E(Ts_Edge)",
        "edge_avg_delay":  "E(Tq_Edge)",
        "cloud_avg_wait":  "E(Ts_Cloud)",
        "cloud_avg_delay": "E(Tq_Cloud)",
        "coord_avg_wait":  "E(Ts_Coord)",
        "coord_avg_delay": "E(Tq_Coord)",

        # --- Numeri medi (STANDARD) ---
        "edge_L":   "E(N_Edge)",
        "edge_Lq":  "E(Nq_Edge)",
        "cloud_L":  "E(N_Cloud)",
        "cloud_Lq": "E(Nq_Cloud)",
        "coord_L":  "E(N_Coord)",
        "coord_Lq": "E(Nq_Coord)",

        # --- Utilizzazione / busy (STANDARD) ---
        "edge_utilization":       "ρ_Edge  (≈ E(Ns_Edge)/m_Edge)",
        "coord_utilization":      "ρ_Coord",
        "cloud_avg_busy_servers": "E(Ns_Cloud)",

        # --- Throughput (STANDARD) ---
        "edge_throughput":  "X_Edge",
        "cloud_throughput": "X_Cloud",
        "coord_throughput": "X_Coord",

        # --- Tempi di servizio (STANDARD) -> service_time_mean = S_* ---
        "edge_service_time_mean":  "S_Edge",
        "cloud_service_time_mean": "S_Cloud",
        "coord_service_time_mean": "S_Coord",

        # --- Metriche per classi (se presenti) ---
        "edge_avg_wait_E":     "E(Ts_E@Edge)",
        "edge_E_avg_response": "E(Ts_E@Edge) (alias)",
        "edge_E_avg_delay":    "E(Tq_E@Edge)",

        # --- Contatori job ---
        "count_E":    "job E completati",
        "count_E_P1": "job E (P1) completati",
        "count_E_P2": "job E (P2) completati",
        "count_E_P3": "job E (P3) completati",
        "count_E_P4": "job E (P4) completati",
        "count_C":    "job C completati",

        # --- Legacy ---
        "E_utilization": "ρ_E (solo centro di classe E).",
        "C_utilization": "ρ_C (solo centro di classe C).",

        # === IMPROVED: Edge_NuoviArrivi (E-only) ===
        "edge_NuoviArrivi_avg_wait":          "E(Ts_E@Edge_NuoviArrivi)",
        "edge_NuoviArrivi_avg_delay":         "E(Tq_E@Edge_NuoviArrivi)",
        "edge_NuoviArrivi_L":                 "E(N_E@Edge_NuoviArrivi)",
        "edge_NuoviArrivi_Lq":                "E(Nq_E@Edge_NuoviArrivi)",
        "edge_NuoviArrivi_Ls":                "E(Ns_E@Edge_NuoviArrivi)",
        "edge_NuoviArrivi_utilization":       "ρ_E@Edge_NuoviArrivi (≈ E(Ns_E)/m)",
        "edge_NuoviArrivi_throughput":        "X_E@Edge_NuoviArrivi",
        "edge_NuoviArrivi_service_time_mean": "S_E@Edge_NuoviArrivi",
        "Edge_NuoviArrivi_E_Ts":              "S_E@Edge_NuoviArrivi (alias)",

        # === IMPROVED: Edge_Feedback (C-only) ===
        "edge_Feedback_avg_wait":          "E(Ts_C@Edge_Feedback)",
        "edge_Feedback_avg_delay":         "E(Tq_C@Edge_Feedback)",
        "edge_Feedback_L":                 "E(N_C@Edge_Feedback)",
        "edge_Feedback_Lq":                "E(Nq_C@Edge_Feedback)",
        "edge_Feedback_Ls":                "E(Ns_C@Edge_Feedback)",
        "edge_Feedback_utilization":       "ρ_C@Edge_Feedback (single-server ⇒ ρ = E(Ns_C))",
        "edge_Feedback_throughput":        "X_C@Edge_Feedback",
        "edge_Feedback_service_time_mean": "S_C@Edge_Feedback",
        "Edge_Feedback_E_Ts":              "S_C@Edge_Feedback (alias)",

        # === MERGED SCALABILITY ===
        "edge_server_number":  "m_Edge (server/core attivi)",
        "coord_server_number": "m_Coord (server/core attivi)",
        "pc":  "P_C (prob. instradamento al Cloud dall’Edge)",
        "p1":  "P(P1 | not Cloud)",
        "p2":  "P(P2 | not Cloud)",
        "p3":  "P(P3 | not Cloud)",
        "p4":  "P(P4 | not Cloud)",
        "edge_scal_trace":  "traccia scaling Edge: (t, m_Edge, ρ_finestra)",
        "coord_scal_trace": "traccia scaling Coord: (t, m_Coord, ρ_finestra)",
    }

    for name, desc in legend.items():
        print(f"{name:30} -> {desc}")


def improved_start_lambda_scan_simulation():
    replicationStats = ReplicationStats_improved()
    print("LAMBDA SCAN SIMULATION - Aeroporto Ciampino")

    file_name = "finite_statistics.csv"
    clear_file_improved(file_name)  # header 'finito' OK

    # Repliche
    for rep in range(cs.REPLICATIONS):
        plantSeeds(cs.SEED + rep)
        base_seed = getSeed()
        print(f"\n★ Replica {rep+1} con seed base = {base_seed}")

        # Tutti gli slot λ
        for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec (Replica {rep+1})")

            stop = cs.SLOT_DURATION
            results, stats = finite_simulation_improved(stop, forced_lambda=lam)

            results['lambda'] = lam
            results['slot'] = lam_index
            results['seed'] = base_seed
            write_file_improved(results, file_name)

            # ⬇️ questa usa i nuovi nomi edge_NuoviArrivi_* (vedi sim_utils)
            append_stats_improved(replicationStats, results, stats)

    print_simulation_stats_improved(replicationStats, "lambda_scan")

    if cs.TRANSIENT_ANALYSIS == 1:

        # Analisi per λ
        plot_multi_seed_per_lambda_improved(
            replicationStats.edge_wait_interval,
            replicationStats.seeds,
            "edge_node", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_seed_per_lambda_improved(
            replicationStats.cloud_wait_interval,
            replicationStats.seeds,
            "cloud_server", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_seed_per_lambda_improved(
            replicationStats.coord_wait_interval,
            replicationStats.seeds,
            "coord_server_edge", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
    return replicationStats


def improved_start_infinite_lambda_scan_simulation():
    print("\nINFINITE SIMULATION - Aeroporto Ciampino")

    file_name = "infinite_statistics.csv"
    # 🔧 usa l'header 'infinite'
    clear_infinite_file_improved(file_name)  # <-- fix

    replicationStats = ReplicationStats_improved()

    plantSeeds(cs.SEED)
    base_seed = getSeed()
    print(f"\n★ Infinite horizon con seed base = {base_seed}")

    for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
        print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec")

        batch_stats = infinite_simulation_improved(forced_lambda=lam)

        for batch_index, results in enumerate(batch_stats.results):
            results['lambda'] = lam
            results['slot'] = lam_index
            results['seed'] = base_seed
            results['batch'] = batch_index
            write_infinite_row_improved(results, file_name)

            # ⬇️ questa salva Edge_NuoviArrivi_* come 'edge' negli array
            append_stats_improved(replicationStats, results, batch_stats)

    print_simulation_stats_improved(replicationStats, "lambda_scan_infinite")
    # 📊 NEW: genera i grafici dall'infinite_statistics.csv (improved)
    plot_infinite_analysis_improved()
    return replicationStats



def improved_start_scalability_simulation():
    """
    Scalabilità UNIFICATA (Edge + Coordinator).
    """
    file_name = "merged_scalability_statistics.csv"
    clear_merged_scalability_file_improved(file_name)

    output_dir = os.path.join("output_improved", "merged_scalability")
    os.makedirs(output_dir, exist_ok=True)

    slot_duration = getattr(cs, "SLOT_DURATION", 3600.0)
    num_slots = len(cs.LAMBDA_SLOTS) if hasattr(cs, "LAMBDA_SLOTS") else 1
    horizon = slot_duration * num_slots
    decision_interval = float(getattr(cs, "SCALING_WINDOW", 1000.0))
    if decision_interval <= 0:
        decision_interval = slot_duration / 20.0

    pc_values = getattr(cs, "PC_VALUES", [0.1, 0.4, 0.5, 0.7, 0.9])
    print("MERGED (EDGE+COORD) SCALABILITY SIMULATION")

    for pc in pc_values:
        set_pc_and_update_probs(pc)
        print(f"\n### p_c = {pc:.2f} (P_COORD={cs.P_COORD:.2f}) | "
              f"P1={cs.P1_PROB:.3f} P2={cs.P2_PROB:.3f} P3={cs.P3_PROB:.3f} P4={cs.P4_PROB:.3f}")

        rep_traces_edge, rep_traces_coord = [], []
        rep_edge_wait_means, rep_coord_wait_means, rep_cloud_wait_means = [], [], []
        edge_servers_all, coord_servers_all = [], []
        edge_scale_events, coord_scale_events = [], []

        for rep in range(cs.REPLICATIONS):
            print(f"  ★ Replica {rep + 1}")
            plantSeeds(cs.SEED + rep)
            seed = getSeed()

            edge_wait_this_rep, coord_wait_this_rep, cloud_wait_this_rep = [], [], []
            t_trace, edge_trace, coord_trace = [], [], []
            t_offset = 0.0
            last_edge_count, last_coord_count = None, None

            for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
                print(f"    ➤ Slot {slot_index} - λ = {lam:.5f} job/sec")
                stop = slot_duration

                res = edge_coord_scalability_simulation_improved(stop=stop, forced_lambda=lam, slot_index=slot_index)
                res["seed"] = seed
                res["lambda"] = lam
                res["slot"] = slot_index
                res["pc"] = pc
                res["p1"], res["p2"], res["p3"], res["p4"] = cs.P1_PROB, cs.P2_PROB, cs.P3_PROB, cs.P4_PROB
                write_file_merged_scalability_improved(res, file_name)

                # 🔧 campi aggiornati
                edge_wait_this_rep.append(res["edge_NuoviArrivi_avg_wait"])
                coord_wait_this_rep.append(res["coord_avg_wait"])
                cloud_wait_this_rep.append(res["cloud_avg_wait"])
                edge_servers_all.append(res["edge_server_number"])
                coord_servers_all.append(res["coord_server_number"])

                for (t_rel, s, _) in res.get("edge_scal_trace", []):
                    t_abs = t_offset + t_rel
                    t_trace.append(t_abs)
                    edge_trace.append(s)
                    if last_edge_count is None:
                        last_edge_count = s
                    elif s != last_edge_count:
                        direction = "UP" if s > last_edge_count else "DOWN"
                        edge_scale_events.append((slot_index, t_abs, direction, s))
                        last_edge_count = s

                for (t_rel, s, _) in res.get("coord_scal_trace", []):
                    t_abs = t_offset + t_rel
                    coord_trace.append(s)
                    if last_coord_count is None:
                        last_coord_count = s
                    elif s != last_coord_count:
                        direction = "UP" if s > last_coord_count else "DOWN"
                        coord_scale_events.append((slot_index, t_abs, direction, s))
                        last_coord_count = s

                t_offset += slot_duration

            if edge_wait_this_rep:
                rep_edge_wait_means.append(sum(edge_wait_this_rep) / len(edge_wait_this_rep))
            if coord_wait_this_rep:
                rep_coord_wait_means.append(sum(coord_wait_this_rep) / len(coord_wait_this_rep))
            if cloud_wait_this_rep:
                rep_cloud_wait_means.append(sum(cloud_wait_this_rep) / len(cloud_wait_this_rep))

            rep_traces_edge.append(list(zip(t_trace, edge_trace)))
            rep_traces_coord.append(list(zip(t_trace[:len(coord_trace)], coord_trace)))

        grid = [i * decision_interval for i in range(int(horizon / decision_interval) + 1)]

        def step_value(trace_tv, t):
            if not trace_tv:
                return 1
            v_last = trace_tv[0][1]
            for (tt, vv) in trace_tv:
                if tt <= t:
                    v_last = vv
                else:
                    break
            return v_last

        avg_edge, avg_coord = [], []
        for t in grid:
            vals_e = [step_value(tr, t) for tr in rep_traces_edge]
            vals_c = [step_value(tr, t) for tr in rep_traces_coord]
            avg_edge.append(sum(vals_e) / max(1, len(vals_e)))
            avg_coord.append(sum(vals_c) / max(1, len(vals_c)))

        plt.figure()
        plt.plot(grid, avg_edge, label="Edge servers (media repliche)")
        plt.plot(grid, avg_coord, label="Coordinator servers (media repliche)")
        plt.xlabel("Tempo")
        plt.ylabel("Numero server")
        plt.title(f"Andamento server nel tempo – pc={pc:.2f} (media su {cs.REPLICATIONS} repliche)")
        plt.legend()
        fig_path = os.path.join(output_dir, f"servers_over_time_pc_{str(pc).replace('.', '_')}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grafico scritto: {fig_path}")

        print("\n--- REPORT CUMULATIVO ---")
        if edge_servers_all:
            print(f"Edge servers     -> min: {min(edge_servers_all)}, max: {max(edge_servers_all)}, avg: {sum(edge_servers_all)/len(edge_servers_all):.4f}")
        if coord_servers_all:
            print(f"Coordinator srv  -> min: {min(coord_servers_all)}, max: {max(coord_servers_all)}, avg: {sum(coord_servers_all)/len(coord_servers_all):.4f}")

        m, ci = calculate_confidence_interval(rep_edge_wait_means)
        print(f"Edge avg wait    -> {m:.4f} ± {ci:.4f} (95% CI)")
        m, ci = calculate_confidence_interval(rep_cloud_wait_means)
        print(f"Cloud avg wait   -> {m:.4f} ± {ci:.4f} (95% CI)")
        m, ci = calculate_confidence_interval(rep_coord_wait_means)
        print(f"Coord avg wait   -> {m:.4f} ± {ci:.4f} (95% CI)")

        print("\nEventi scalabilità EDGE (tempo assoluto; slot λ):")
        if edge_scale_events:
            for slot_idx, t_abs, direction, new_count in edge_scale_events:
                print(f"  t={t_abs:.1f}s  slot={slot_idx}  -> {direction} a {new_count} server")
        else:
            print("  Nessun cambiamento")

        print("\nEventi scalabilità COORD (tempo assoluto; slot λ):")
        if coord_scale_events:
            for slot_idx, t_abs, direction, new_count in coord_scale_events:
                print(f"  t={t_abs:.1f}s  slot={slot_idx}  -> {direction} a {new_count} server")
        else:
            print("  Nessun cambiamento")

    print(f"\nCSV scritto: output_improved/{file_name}")


def _mean_ci_95(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    n = len(s)
    if n == 0:
        return (None, None, 0)
    mean = float(s.mean())
    if n < 2:
        return (mean, 0.0, n)
    std = float(s.std(ddof=1))
    margin = 1.96 * (std / (n ** 0.5))
    return (mean, margin, n)

def summarize_by_lambda(input_csv: str,
                        output_name: str | None = None,
                        exclude_cols=None,
                        output_dir: str | None = None) -> str:
    """
    Crea un report di medie ± CI(95%).
    - Se il CSV contiene la colonna 'lambda' → raggruppa per λ (comportamento classico).
    - Se il CSV NON contiene 'lambda' → assume un singolo λ (letto da constants.LAMBDA se disponibile)
      e produce un unico blocco di riepilogo.

    Parametri:
      - input_csv    : percorso al CSV di input.
      - output_name  : nome del file di output (con o senza estensione).
                       Se None -> "summary_by_lambda_Global_Table.txt" nella cartella dell'input.
      - exclude_cols : iterable opzionale di colonne da escludere (si sommano alle default).
      - output_dir   : cartella di output; se None -> stessa cartella dell'input.

    Ritorna:
      Percorso del file di testo generato.
    """
    import os
    import pandas as pd
    from datetime import datetime

    # per recuperare LAMBDA se serve (senza imporre il path esatto del modulo)
    lambda_from_constants = None
    for mod_name in ("utils.constants", "constants"):
        try:
            cs_mod = __import__(mod_name, fromlist=["LAMBDA"])
            if hasattr(cs_mod, "LAMBDA"):
                lambda_from_constants = float(getattr(cs_mod, "LAMBDA"))
                break
        except Exception:
            pass

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"No data in {input_csv}")

    # normalizza intestazioni
    df.columns = [str(c).strip() for c in df.columns]

    # colonne da escludere (default + eventuali custom)
    exclude = set(exclude_cols or [])
    exclude |= {
        'seed','slot','lambda','batch',
        'pc','p1','p2','p3','p4',
        'edge_scal_trace','coord_scal_trace',
        'server_utilization_by_count'
    }

    # individua dinamicamente colonne numeriche presenti (ed escluse)
    metric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # prova a capire se è numerica
        try:
            pd.to_numeric(df[c], errors='raise')
            metric_cols.append(c)
        except Exception:
            pass

    if not metric_cols:
        raise ValueError("Nessuna metrica numerica trovata nel CSV dopo le esclusioni.")

    # utility per mean ± CI(95%)
    def _mean_ci_95(s: pd.Series):
        s = pd.to_numeric(s, errors='coerce').dropna()
        n = s.size
        if n == 0:
            return None, None, 0
        m = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else 0.0
        z = 1.96  # 95%
        margin = z * (sd / (n ** 0.5)) if n > 1 else 0.0
        return m, margin, n

    lines = []
    lines.append(f"# Summary report generated on {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Source CSV: {input_csv}")
    lines.append("Note: mean ± 95% CI (z=1.96).")
    lines.append("")

    def bucket_key(col: str):
        if col.startswith('edge_NuoviArrivi_'): return (0, col)
        if col.startswith('edge_Feedback_'):    return (1, col)
        if col.startswith('cloud_'):            return (2, col)
        if col.startswith('coord_'):            return (3, col)
        if col.startswith('edge_'):             return (4, col)
        return (5, col)

    if 'lambda' in df.columns:
        # ---- Caso classico: raggruppa per λ ----
        grouped = df.groupby('lambda', dropna=False)
        for lam, g in grouped:
            lam_val = float(lam) if pd.notna(lam) else (lambda_from_constants if lambda_from_constants is not None else float('nan'))
            header = f"--- λ = {lam_val:.6f}" if pd.notna(lam_val) else "--- λ = (from CSV)"
            lines.append(f"{header}  (rows={len(g)})")
            for col in sorted(metric_cols, key=bucket_key):
                mean, margin, n = _mean_ci_95(g[col])
                if mean is None:
                    continue
                lines.append(f"{col}: {mean:.6f} ± {margin:.6f}  [n={n}]")
            lines.append("")
    else:
        # ---- Nuovo comportamento: singolo λ implicito ----
        if lambda_from_constants is not None:
            lines.append(f"--- λ = {lambda_from_constants:.6f} (single-run)  (rows={len(df)})")
        else:
            lines.append(f"--- λ = single-run  (rows={len(df)})")

        for col in sorted(metric_cols, key=bucket_key):
            mean, margin, n = _mean_ci_95(df[col])
            if mean is None:
                continue
            lines.append(f"{col}: {mean:.6f} ± {margin:.6f}  [n={n}]")
        lines.append("")

    # risoluzione percorso output
    base_dir = output_dir if output_dir is not None else os.path.dirname(input_csv) or '.'
    os.makedirs(base_dir, exist_ok=True)

    if output_name is None or not str(output_name).strip():
        output_txt = os.path.join(base_dir, "summary_by_lambda_Global_Table.txt")
    else:
        # se manca estensione, aggiungi .txt
        name = str(output_name)
        output_txt = os.path.join(base_dir, name if os.path.splitext(name)[1] else f"{name}.txt")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Summary written to: {output_txt}")
    return output_txt


def _fmt_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

if __name__ == "__main__":
    """
    Avvio della simulazione quando il file viene eseguito direttamente.
    Misura anche i tempi di esecuzione per Standard, Improved e Totale.
    """
    t0 = time.perf_counter()
    print(f"\n▶ START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print_csv_legend()

    # ===================== STANDARD =====================
    print("INIZIO---- STANDARD MODEL SIMULTIONS.\n")
    t_std = time.perf_counter()

    #stats_finite = start_finite_simulation()

    #summarize_by_lambda("output/finite_statistics.csv",
    #                    output_name="FINITE_statistics_Global.txt",
     #                   output_dir="reports_Standard_Model")

   # start_transient_analysis()

   # stats_infinite = start_infinite_single_simulation()
   # summarize_by_lambda("output/infinite_statistics.csv",
      #    output_name="INFINITE_statistics_Global.txt",
      #    output_dir="reports_Standard_Model")

    start_scalability_simulation()
    summarize_by_lambda("output/merged_scalability_statistics.csv",
          output_name="SCALABILITY_by_lambda_report.txt",
                               output_dir="reports_Standard_Model")

    #  dt_std = time.perf_counter() - t_std
    # print(f"\n⏱ Tempo STANDARD: {_fmt_hms(dt_std)}")
    # print("FINE---- STANDARD MODEL SIMULTIONS.\n")

    # ===================== IMPROVED =====================
    # print("INIZIO---- IMPROVED MODEL SIMULTIONS.\n")
    #  t_imp = time.perf_counter()

    # improved_stats_finite = improved_start_lambda_scan_simulation()

    #  summarize_by_lambda("output_improved/finite_statistics.csv",
    #      output_name="FINITE_statistics_Global.txt",
    #      output_dir="reports_Improved_Model")
    #
    # improved_stats_infinite = improved_start_infinite_lambda_scan_simulation()
    #  summarize_by_lambda("output_improved/infinite_statistics.csv",
    #                    output_name="INFINITE_statistics_Global.txt",
    #                   output_dir="reports_Improved_Model")

    # improved_start_scalability_simulation()
    #summarize_by_lambda("output_improved/merged_scalability_statistics.csv",
    #   output_name="SCALABILITY_statistics_Global.txt",
    #    output_dir="reports_Improved_Model")

    # dt_imp = time.perf_counter() - t_imp
    #print("\nStatsitche FINITE comulative per Migliorativo.\n")
    #print(f"⏱ Tempo IMPROVED: {_fmt_hms(dt_imp)}")
    #print("FINE---- IMPROVED MODEL SIMULTIONS.\n")

    # ===================== TOTALE =====================
    dt_total = time.perf_counter() - t0
    print(f"⏱ Tempo TOTALE esecuzione: {_fmt_hms(dt_total)}")
    print(f"▶ END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


