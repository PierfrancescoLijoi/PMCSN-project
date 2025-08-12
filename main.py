"""
main.py
---------
Questo è il file principale che avvia la simulazione per il modello standard
descritto nel documento PMCSN Project (Luglio 2025).

Riferimento: Sezione "Modello computazionale" del documento.
La simulazione segue un approccio next-event-driven con orizzonte finito
(transiente), come definito a pagina 8–10 del testo allegato.
"""

import utils.constants as cs
from simulation.simulator import finite_simulation, infinite_simulation
from utils.simulation_output import write_file, clear_file, print_simulation_stats, plot_analysis, \
    plot_multi_lambda_per_seed, plot_multi_seed_per_lambda, write_scalability_trace, clear_infinite_file, \
    write_infinite_row, plot_infinite_analysis
from utils.simulation_stats import ReplicationStats
from utils.sim_utils import append_stats
from simulation.edge_scalability_simulator import edge_scalability_simulation
from utils.simulation_output import write_file_edge_scalability, clear_edge_scalability_file
from utils.sim_utils import append_edge_scalability_stats
from utils.simulation_stats import ReplicationStats
from utils.simulation_stats import ReplicationStats
from simulation.coordinator_scalability_simulator import coordinator_scalability_simulation   # ← NEW
from utils.simulation_output import (
    write_file_edge_scalability, clear_edge_scalability_file,
    write_file_coord_scalability, clear_coord_scalability_file      # ← NEW
)
from utils.sim_utils import append_edge_scalability_stats, append_coord_scalability_stats     # ← NEW


from libraries.rngs import plantSeeds, selectStream, getSeed

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
    clear_infinite_file(file_name)

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
    plot_infinite_analysis()
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

if __name__ == "__main__":
    """
    Avvio della simulazione quando il file viene eseguito direttamente.
    """
    print_csv_legend()
    #stats_finite = start_lambda_scan_simulation()
    stats_infinite = start_infinite_lambda_scan_simulation()
    #start_edge_scalability_simulation()
    #start_coord_scalability_simulation()

