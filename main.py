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


def start_finite_simulation():
    """
    Avvia la simulazione finita e transiente per il modello standard.

    - Riferimento: Sezione 5.1 e 5.2 del documento.
    - Orizzonte: 24 ore (STOP = 86400 secondi).
    - Metodo: next-event-driven, esecuzione fino a STOP o fino a svuotare il sistema.
    """

    replicationStats = ReplicationStats()  # raccoglie i dati di tutte le repliche

    print("FINITE STANDARD SIMULATION - Aeroporto Ciampino")

    # Nome del file CSV per salvare i risultati
    file_name = "finite_standard_statistics.csv"

    # Pulizia file di output e scrittura intestazione
    clear_file(file_name)

    # Determinazione del tempo di stop
    if cs.TRANSIENT_ANALYSIS == 1:
        # Se attiva l’analisi transiente → STOP_ANALYSIS più corto
        stop = cs.STOP_ANALYSIS
    else:
        # Altrimenti simulazione su 24 ore intere
        stop = cs.STOP

    # Esecuzione delle repliche
    for i in range(cs.REPLICATIONS):
        # Lancio della simulazione
        results, stats = finite_simulation(stop)

        # Salvataggio dei risultati su CSV
        write_file(results, file_name)

        # Aggiunta dei risultati alla collezione cumulativa
        append_stats(replicationStats, results, stats)

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
    stats = start_finite_simulation()
