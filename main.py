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
    plot_multi_lambda_per_seed, plot_multi_seed_per_lambda, write_scalability_trace
from utils.simulation_stats import ReplicationStats
from utils.sim_utils import append_stats
from simulation.edge_scalability_simulator import edge_scalability_simulation
from utils.simulation_output import write_file_edge_scalability, clear_edge_scalability_file
from utils.sim_utils import append_edge_scalability_stats
from utils.simulation_stats import ReplicationStats

from libraries.rngs import plantSeeds, selectStream, getSeed

def start_lambda_scan_simulation():
    replicationStats = ReplicationStats()
    print("LAMBDA SCAN SIMULATION - Aeroporto Ciampino")

    file_name = "lambda_scan_statistics.csv"
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
    print_simulation_stats(replicationStats, "replications")

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

    for rep in range(cs.REPLICATIONS):
        plantSeeds(cs.SEED + rep)
        base_seed = getSeed()
        print(f"\n★ Replica {rep+1} con seed base = {base_seed}")

        for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):
            print(f"\n➤ Slot λ[{lam_index}] = {lam:.5f} job/sec (Replica {rep+1})")

            batch_stats = infinite_simulation(forced_lambda=lam)

            # arricchisci i risultati di ogni batch
            for batch_index, results in enumerate(batch_stats.results):
                results['lambda'] = lam
                results['slot'] = lam_index
                results['seed'] = base_seed
                write_file(results, file_name)

                append_stats(replicationStats, results, batch_stats)

    print_simulation_stats(replicationStats, "lambda_scan_infinite")
    print_simulation_stats(replicationStats, "replications")

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

if __name__ == "__main__":
    """
    Avvio della simulazione quando il file viene eseguito direttamente.
    """
    stats_finite = start_lambda_scan_simulation()

    start_edge_scalability_simulation()

    stats_infinite = start_infinite_lambda_scan_simulation()