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

from matplotlib import pyplot as plt

import utils.constants as cs
from simulation.edge_ccord_scalability_simulator import edge_coord_scalability_simulation
from simulation.simulator import finite_simulation, infinite_simulation
from utils.simulation_output import write_file, clear_file, print_simulation_stats, \
    plot_multi_lambda_per_seed, plot_multi_seed_per_lambda, \
    write_infinite_row,  write_file_merged_scalability, clear_merged_scalability_file
from simulation.edge_scalability_simulator import edge_scalability_simulation
from utils.simulation_stats import ReplicationStats
from simulation.coordinator_scalability_simulator import coordinator_scalability_simulation   # ← NEW
from utils.simulation_output import (
    write_file_edge_scalability, clear_edge_scalability_file,
    write_file_coord_scalability, clear_coord_scalability_file      # ← NEW
)
from simulation.improved_edge_ccord_scalability_simulator import edge_coord_scalability_simulation_improved
from simulation.improved_simulator import finite_simulation_improved, infinite_simulation_improved
from utils.improved_simulation_output import write_file_improved, clear_file_improved, print_simulation_stats_improved, \
    plot_multi_lambda_per_seed_improved, plot_multi_seed_per_lambda_improved, \
    write_infinite_row_improved,  write_file_merged_scalability_improved, clear_merged_scalability_file_improved
from utils.sim_utils import append_stats, calculate_confidence_interval, set_pc_and_update_probs
from simulation.improved_edge_scalability_simulator import edge_scalability_simulation_improved
from utils.improved_simulation_stats import ReplicationStats_improved
from simulation.improved_coordinator_scalability_simulator import coordinator_scalability_simulation_improved   # ← NEW
from utils.improved_simulation_output import (
    write_file_edge_scalability_improved, clear_edge_scalability_file_improved,
    write_file_coord_scalability_improved, clear_coord_scalability_file_improved      # ← NEW
)

from utils.sim_utils import append_edge_scalability_stats, append_coord_scalability_stats     # ← NEW


from libraries.rngs import plantSeeds, getSeed

def start_lambda_scan_simulation():
    replicationStats = ReplicationStats()
    print("LAMBDA SCAN SIMULATION - Aeroporto Ciampino")

    file_name = "finite_statistics.csv"
    clear_file(file_name)

    # Ciclo sulle repliche
    for rep in range(cs.REPLICATIONS):
        # Inizializza il seed per questa replica
        plantSeeds(cs.SEED + rep)
        base_seed = getSeed()
        print(f"\n★ Replica {rep+1} con seed base = {base_seed}")

        # Ciclo su tutti i λ
        for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec (Replica {rep+1})")

            stop = cs.SLOT_DURATION
            results, stats = finite_simulation(stop, forced_lambda=lam)

            results['lambda'] = lam
            results['slot'] = lam_index
            results['seed'] = base_seed  # forza stesso seed nel CSV
            write_file(results, file_name)

            append_stats(replicationStats, results, stats)

    print_simulation_stats(replicationStats, "lambda_scan")

    if cs.TRANSIENT_ANALYSIS == 1:
        # Analisi per seed (già implementata)
        plot_multi_lambda_per_seed(
            replicationStats.edge_wait_interval,
            replicationStats.seeds,
            "edge_node", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_lambda_per_seed(
            replicationStats.cloud_wait_interval,
            replicationStats.seeds,
            "cloud_server", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_lambda_per_seed(
            replicationStats.coord_wait_interval,
            replicationStats.seeds,
            "coord_server_edge", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )

        # Analisi per λ
        plot_multi_seed_per_lambda(
            replicationStats.edge_wait_interval,
            replicationStats.seeds,
            "edge_node", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_seed_per_lambda(
            replicationStats.cloud_wait_interval,
            replicationStats.seeds,
            "cloud_server", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )
        plot_multi_seed_per_lambda(
            replicationStats.coord_wait_interval,
            replicationStats.seeds,
            "coord_server_edge", "lambda_scan",
            replicationStats.lambdas,
            replicationStats.slots
        )

    return replicationStats

def start_infinite_lambda_scan_simulation():
    print("\nINFINITE SIMULATION - Aeroporto Ciampino")

    file_name = "infinite_statistics.csv"
    clear_file(file_name)

    replicationStats = ReplicationStats()

    # un solo seed di base per l’infinite horizon
    plantSeeds(cs.SEED)
    base_seed = getSeed()
    print(f"\n★ Infinite horizon con seed base = {base_seed}")

    for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
        print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec")

        batch_stats = infinite_simulation(forced_lambda=lam)

        for batch_index, results in enumerate(batch_stats.results):
            results['lambda'] = lam
            results['slot'] = lam_index
            results['seed'] = base_seed
            results['batch'] = batch_index
            write_infinite_row(results, file_name)
            append_stats(replicationStats, results, batch_stats)

    print_simulation_stats(replicationStats, "lambda_scan_infinite")
    return replicationStats

def start_coord_scalability_simulation():
    """
    Coordinator Scalability:
    - Legge output/edge_scalability_statistics.csv, calcola per ciascun λ la MEDIA dei server Edge usati (per slot).
    - Usa quella media (arrotondata, min 1) come N fisso di server Edge.
    - Scala dinamicamente i server del Coordinator con le stesse soglie di utilizzo degli Edge.
    - Esporta CSV/JSON/TXT come fatto per Edge.
    """
    import os
    import pandas as pd
    import utils.constants as cs
    from libraries.rngs import plantSeeds, getSeed

    replicationStats = ReplicationStats()
    file_name = "coord_scalability_statistics.csv"
    clear_coord_scalability_file(file_name)

    # carica CSV Edge già generato
    path_edge_csv = os.path.join("output", "edge_scalability_statistics.csv")
    if not os.path.isfile(path_edge_csv):
        raise FileNotFoundError("Non trovo output/edge_scalability_statistics.csv. Esegui prima la scalabilità Edge.")

    df = pd.read_csv(path_edge_csv)

    # media server per slot di λ (robusto a floating di λ)
    slot_to_avg = df.groupby('slot')['edge_server_number'].mean().to_dict()

    print("COORDINATOR SCALABILITY SIMULATION")

    for rep in range(cs.REPLICATIONS):
        print(f"\n★ Replica {rep + 1}")
        plantSeeds(cs.SEED + rep)
        seed = getSeed()

        for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f" ➔ Slot {slot_index} - λ = {lam:.5f} job/sec")
            fixed_edge_servers = max(1, min(int(round(slot_to_avg.get(slot_index, 1))), cs.EDGE_SERVERS_MAX))

            stop = cs.SLOT_DURATION
            results, stats = coordinator_scalability_simulation(
                stop, forced_lambda=lam, slot_index=slot_index, fixed_edge_servers=fixed_edge_servers
            )

            results['seed'] = seed
            results['lambda'] = lam
            results['slot'] = slot_index

            write_file_coord_scalability(results, file_name)
            append_coord_scalability_stats(replicationStats, results, stats)

    from utils.simulation_output import print_simulation_stats
    print_simulation_stats(replicationStats, "coord_scalability")

    return replicationStats


def start_edge_scalability_simulation():
    replicationStats = ReplicationStats()
    file_name = "edge_scalability_statistics.csv"
    clear_edge_scalability_file(file_name)

    print("EDGE SCALABILITY SIMULATION")

    for rep in range(cs.REPLICATIONS):
        print(f"\n★ Replica {rep + 1}")
        plantSeeds(cs.SEED + rep)
        seed = getSeed()

        for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f" ➔ Slot {slot_index} - λ = {lam:.5f} job/sec")

            stop = cs.SLOT_DURATION
            results, stats = edge_scalability_simulation(stop, forced_lambda=lam, slot_index=slot_index)

            results['seed'] = seed
            results['lambda'] = lam
            results['slot'] = slot_index

            write_file_edge_scalability(results, file_name)


            append_edge_scalability_stats(replicationStats, results, stats)

    print_simulation_stats(replicationStats, "edge_scalability")

    return replicationStats

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

    print(f"\nCSV scritto: output/{file_name}")


def print_csv_legend():
    print("\n=== CSV Columns Legend ===\n")

    legend = {
        # Identificativi simulazione
        "seed": "Identificativo del seme RNG usato per la replica (per riproducibilità).",
        "slot": "Indice della fascia oraria (utile per scenari con λ variabile per slot).",
        "lambda": "Tasso medio di arrivi impostato in quella replica (job/unità tempo).",

        # Tempi
        "edge_avg_wait": "Tempo medio di risposta W del nodo Edge (attesa + servizio).",
        "cloud_avg_wait": "Tempo medio di risposta W del nodo Cloud.",
        "coord_avg_wait": "Tempo medio di risposta W del nodo Coordinator.",
        "edge_avg_delay": "Tempo medio di attesa in coda Wq del nodo Edge (senza servizio).",
        "cloud_avg_delay": "Tempo medio di attesa in coda Wq del nodo Cloud.",
        "coord_avg_delay": "Tempo medio di attesa in coda Wq del nodo Coordinator.",

        # Numero medio di job
        "edge_L": "Numero medio di job presenti nel nodo Edge (in coda + in servizio).",
        "edge_Lq": "Numero medio di job in coda nel nodo Edge.",
        "cloud_L": "Numero medio di job presenti nel nodo Cloud.",
        "cloud_Lq": "Numero medio di job in coda nel nodo Cloud.",
        "coord_L": "Numero medio di job presenti nel nodo Coordinator.",
        "coord_Lq": "Numero medio di job in coda nel nodo Coordinator.",

        # Utilizzazione
        "edge_utilization": "Utilizzazione media aggregata del nodo Edge (area.service/T).",
        "coord_utilization": "Utilizzazione media aggregata del nodo Coordinator.",
        "cloud_avg_busy_servers": "Numero medio di server occupati nel Cloud (ρ in multi-server).",

        # Throughput
        "edge_throughput": "Tasso medio di completamenti X nel nodo Edge (job/unità tempo).",
        "cloud_throughput": "Tasso medio di completamenti X nel nodo Cloud.",
        "coord_throughput": "Tasso medio di completamenti X nel nodo Coordinator.",

        # Tempi di servizio medi osservati
        "edge_service_time_mean": "Tempo medio di servizio realizzato nel nodo Edge (1/μ osservato).",
        "cloud_service_time_mean": "Tempo medio di servizio realizzato nel nodo Cloud.",
        "coord_service_time_mean": "Tempo medio di servizio realizzato nel nodo Coordinator.",

        # Contatori job completati
        "count_E": "Numero di job di classe E completati.",
        "count_E_P1": "Numero di job di classe E_P1 completati.",
        "count_E_P2": "Numero di job di classe E_P2 completati.",
        "count_E_P3": "Numero di job di classe E_P3 completati.",
        "count_E_P4": "Numero di job di classe E_P4 completati.",
        "count_C": "Numero di job di classe C completati.",

        # Legacy (vecchie metriche)
        "E_utilization": (
            "Utilizzazione calcolata SOLO sul centro di servizio associato alla CLASSE E "
            "(cioè la singola coda/server che gestisce i job di tipo E nel modello). "
            "Non include eventuali altri centri/server presenti nello stesso nodo Edge "
            "che servono job di altre classi (es. P1, P2...). "
            "Per questo motivo, se il nodo Edge ospita più code/server per classi diverse, "
            "questo valore può essere significativamente più basso rispetto a 'edge_utilization', "
            "che invece considera l'uso complessivo di tutti i server del nodo Edge."
        ),
        "C_utilization": (
            "Utilizzazione calcolata SOLO sul centro di servizio associato alla CLASSE C "
            "(cioè la singola coda/server che gestisce i job di tipo C nel modello). "
            "Non include eventuali altri centri/server presenti nello stesso nodo Coordinator "
            "che servono job di altre classi o funzioni. "
            "Se il Coordinator ospita più code o server per tipi di job diversi, "
            "questo valore può essere più basso rispetto a 'coord_utilization', "
            "che tiene conto di tutti i server presenti nel nodo Coordinator."
        )

    }
    for name, desc in legend.items():
        print(f"{name:30} -> {desc}")


def improved_start_lambda_scan_simulation():
    replicationStats = ReplicationStats_improved()
    print("LAMBDA SCAN SIMULATION - Aeroporto Ciampino")

    file_name = "finite_statistics.csv"
    clear_file_improved(file_name)

    # Ciclo sulle repliche
    for rep in range(cs.REPLICATIONS):
        # Inizializza il seed per questa replica
        plantSeeds(cs.SEED + rep)
        base_seed = getSeed()
        print(f"\n★ Replica {rep+1} con seed base = {base_seed}")

        # Ciclo su tutti i λ
        for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec (Replica {rep+1})")

            stop = cs.SLOT_DURATION
            results, stats = finite_simulation_improved(stop, forced_lambda=lam)

            results['lambda'] = lam
            results['slot'] = lam_index
            results['seed'] = base_seed  # forza stesso seed nel CSV
            write_file_improved(results, file_name)

            append_stats(replicationStats, results, stats)

    print_simulation_stats_improved(replicationStats, "lambda_scan")

    if cs.TRANSIENT_ANALYSIS == 1:
        # Analisi per seed (già implementata)
        plot_multi_lambda_per_seed_improved(replicationStats.edge_wait_interval, replicationStats.seeds, "edge_node",
                                            "lambda_scan", replicationStats.lambdas, replicationStats.slots)
        plot_multi_lambda_per_seed_improved(replicationStats.cloud_wait_interval, replicationStats.seeds,
                                            "cloud_server", "lambda_scan", replicationStats.lambdas,
                                            replicationStats.slots)
        plot_multi_lambda_per_seed_improved(replicationStats.coord_wait_interval, replicationStats.seeds,
                                            "coord_server_edge", "lambda_scan", replicationStats.lambdas,
                                            replicationStats.slots)

        # Analisi per λ
        plot_multi_seed_per_lambda_improved(replicationStats.edge_wait_interval, replicationStats.seeds, "edge_node",
                                            "lambda_scan", replicationStats.lambdas, replicationStats.slots)
        plot_multi_seed_per_lambda_improved(replicationStats.cloud_wait_interval, replicationStats.seeds,
                                            "cloud_server", "lambda_scan", replicationStats.lambdas,
                                            replicationStats.slots)
        plot_multi_seed_per_lambda_improved(replicationStats.coord_wait_interval, replicationStats.seeds,
                                            "coord_server_edge", "lambda_scan", replicationStats.lambdas,
                                            replicationStats.slots)

    return replicationStats

def improved_start_infinite_lambda_scan_simulation():
    print("\nINFINITE SIMULATION - Aeroporto Ciampino")

    file_name = "infinite_statistics.csv"
    clear_file_improved(file_name)

    replicationStats = ReplicationStats_improved()

    # un solo seed di base per l’infinite horizon
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
            append_stats(replicationStats, results, batch_stats)

    print_simulation_stats_improved(replicationStats, "lambda_scan_infinite")
    return replicationStats

def improved_start_coord_scalability_simulation():
    """
    Coordinator Scalability:
    - Legge output/edge_scalability_statistics.csv, calcola per ciascun λ la MEDIA dei server Edge usati (per slot).
    - Usa quella media (arrotondata, min 1) come N fisso di server Edge.
    - Scala dinamicamente i server del Coordinator con le stesse soglie di utilizzo degli Edge.
    - Esporta CSV/JSON/TXT come fatto per Edge.
    """
    import os
    import pandas as pd
    import utils.constants as cs
    from libraries.rngs import plantSeeds, getSeed

    replicationStats = ReplicationStats_improved()
    file_name = "coord_scalability_statistics.csv"
    clear_coord_scalability_file_improved(file_name)

    # carica CSV Edge già generato
    path_edge_csv = os.path.join("output_improved", "edge_scalability_statistics.csv")
    if not os.path.isfile(path_edge_csv):
        raise FileNotFoundError("Non trovo output_improved/edge_scalability_statistics.csv. Esegui prima la scalabilità Edge.")

    df = pd.read_csv(path_edge_csv)

    # media server per slot di λ (robusto a floating di λ)
    slot_to_avg = df.groupby('slot')['edge_server_number'].mean().to_dict()

    print("COORDINATOR SCALABILITY SIMULATION")

    for rep in range(cs.REPLICATIONS):
        print(f"\n★ Replica {rep + 1}")
        plantSeeds(cs.SEED + rep)
        seed = getSeed()

        for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f" ➔ Slot {slot_index} - λ = {lam:.5f} job/sec")
            fixed_edge_servers = max(1, min(int(round(slot_to_avg.get(slot_index, 1))), cs.EDGE_SERVERS_MAX))

            stop = cs.SLOT_DURATION
            results, stats = coordinator_scalability_simulation_improved(stop, forced_lambda=lam, slot_index=slot_index,
                                                                         fixed_edge_servers=fixed_edge_servers)

            results['seed'] = seed
            results['lambda'] = lam
            results['slot'] = slot_index

            write_file_coord_scalability_improved(results, file_name)
            append_coord_scalability_stats(replicationStats, results, stats)

    from utils.improved_simulation_output import print_simulation_stats_improved
    print_simulation_stats_improved(replicationStats, "coord_scalability")

    return replicationStats


def improved_start_edge_scalability_simulation():
    replicationStats = ReplicationStats_improved()
    file_name = "edge_scalability_statistics.csv"
    clear_edge_scalability_file_improved(file_name)

    print("EDGE SCALABILITY SIMULATION")

    for rep in range(cs.REPLICATIONS):
        print(f"\n★ Replica {rep + 1}")
        plantSeeds(cs.SEED + rep)
        seed = getSeed()

        for slot_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f" ➔ Slot {slot_index} - λ = {lam:.5f} job/sec")

            stop = cs.SLOT_DURATION
            results, stats = edge_scalability_simulation_improved(stop, forced_lambda=lam, slot_index=slot_index)

            results['seed'] = seed
            results['lambda'] = lam
            results['slot'] = slot_index

            write_file_edge_scalability_improved(results, file_name)


            append_edge_scalability_stats(replicationStats, results, stats)

    print_simulation_stats_improved(replicationStats, "edge_scalability")

    return replicationStats

def improved_start_scalability_simulation():
    """
    Scalabilità UNIFICATA (Edge + Coordinator) con metodo a repliche:
    - Esegue tutte le repliche per ciascun p_c (in cs.PC_VALUES), poi passa al successivo.
    - Scrive CSV (una riga per replica×slot).
    - Grafico: 1 per p_c (media a gradini del numero di server sulle repliche).
    - Report cumulativo per p_c: min/max/media server, log UP/DOWN con tempo e slot, W ± CI(95%) per Edge/Cloud/Coord.
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
        set_pc_and_update_probs(pc)  # imposta cs.P_C, cs.P_COORD e le condizionate P1..P4
        s = cs.P1_PROB + cs.P2_PROB + cs.P3_PROB + cs.P4_PROB

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






if __name__ == "__main__":
    """
    Avvio della simulazione quando il file viene eseguito direttamente.
    """
    print_csv_legend()
    stats_finite = start_lambda_scan_simulation()
    stats_infinite = start_infinite_lambda_scan_simulation()
    start_edge_scalability_simulation()
    start_coord_scalability_simulation()
    start_scalability_simulation()
    print("Simulation completed successfully.\n")

    print("INIZIO---- IMPROVED MODEL SIMULTIONS.\n")
    improved_stats_finite = improved_start_lambda_scan_simulation()
    improved_stats_infinite = improved_start_infinite_lambda_scan_simulation()
    improved_start_edge_scalability_simulation()
    improved_start_coord_scalability_simulation()
    improved_start_scalability_simulation()
    print("FINE---- IMPROVED MODEL SIMULTIONS.\n")


