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
from simulation.simulator import finite_simulation
from utils.simulation_output import write_file, clear_file, print_simulation_stats, plot_analysis
from utils.simulation_stats import ReplicationStats
from utils.sim_utils import append_stats


def start_lambda_scan_simulation():
    """
    Esegue una simulazione per ciascun valore di λ definito in LAMBDA_SLOTS.

    - Per ogni λ, esegue N repliche con lo stesso seed pattern.
    - Ogni simulazione forza il valore di λ, ignorando l’orario corrente.
    - Salva i risultati su CSV e stampa statistiche aggregate.
    """

    replicationStats = ReplicationStats()  # raccoglie i dati di tutte le simulazioni

    print("LAMBDA SCAN SIMULATION - Aeroporto Ciampino")

    # Nome del file CSV per salvare i risultati
    file_name = "lambda_scan_statistics.csv"

    # Pulizia file di output e scrittura intestazione
    clear_file(file_name)

    # Ciclo su ogni valore di λ definito nello slot
    for lam_index, (_, _, lam) in enumerate(cs.LAMBDA_SLOTS):

        print(f"\n➤ Simulazione slot λ[{lam_index}] = {lam:.5f} job/sec")

        for rep in range(cs.REPLICATIONS):
            # Durata simulazione fissa per ogni λ
            stop = cs.SLOT_DURATION

            # Esecuzione simulazione con λ forzato
            results, stats = finite_simulation(stop, forced_lambda=lam)

            # Salvataggio dei risultati su CSV (con info slot/λ)
            results['lambda'] = lam
            results['slot'] = lam_index
            write_file(results, file_name)

            # Aggiunta dei risultati alla collezione cumulativa
            append_stats(replicationStats, results, stats)

    # Stampa delle statistiche aggregate finali
    print_simulation_stats(replicationStats, "lambda_scan")
    # Stampa delle statistiche aggregate
    print_simulation_stats(replicationStats, "replications")

    # Se attiva analisi transiente → genera grafici temporali
    if cs.TRANSIENT_ANALYSIS == 1:
        plot_analysis(replicationStats.edge_wait_interval,
                      replicationStats.seeds,
                      "edge_node", "standard")
        plot_analysis(replicationStats.cloud_wait_interval,
                      replicationStats.seeds,
                      "cloud_server", "standard")
        plot_analysis(replicationStats.coord_wait_interval,
                      replicationStats.seeds,
                      "coord_server_edge", "standard")

    return replicationStats



if __name__ == "__main__":
    """
    Avvio della simulazione quando il file viene eseguito direttamente.
    """
    stats = start_lambda_scan_simulation()
