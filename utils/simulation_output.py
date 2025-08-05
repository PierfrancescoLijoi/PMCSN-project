"""
utils/simulation_output.py
--------------------------
Gestione dell'output della simulazione:
- Scrittura file CSV
- Stampa delle statistiche aggregate
- Generazione di grafici per analisi transiente

Riferimento: Sezioni "Scopi e Obiettivi" e "Modello computazionale"
del documento PMCSN Project (Luglio 2025).
"""
import csv
import statistics
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import constants as cs
from utils.sim_utils import calculate_confidence_interval

# Directory di output
file_path = "output/"

# Intestazione CSV per i risultati della simulazione
header = [
    "seed",
    "edge_avg_wait",
    "cloud_avg_wait",
    "coord_avg_wait",
    "count_E",
    "count_E_P1P2",  # Aggiungi questi
    "count_E_P3P4",  # due nuovi campi
    "count_C",
    "E_utilization",
    "C_utilization"
]


def clear_file(file_name):
    """
    Pulisce il file di output e scrive l'intestazione.

    Riferimento: per ogni nuova simulazione (Sezione 5.2),
    serve un file CSV fresco per raccogliere i dati delle repliche.
    """
    path = os.path.join(file_path, file_name)
    os.makedirs(file_path, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


def write_file(results, file_name):
    """
    Scrive i risultati di una replica nel file CSV.

    Ogni riga corrisponde a una replica con un seed diverso.
    """
    path = os.path.join(file_path, file_name)
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writerow(results)


def print_simulation_stats(stats, type):
    """
    Stampa le statistiche aggregate dopo tutte le repliche.

    - Usa la media delle repliche.
    - Calcola anche l’intervallo di confidenza al 95%.
    - Riferimento: Obiettivi 1 e 2 → garantire tempi medi di risposta < 3s
    """
    print(f"\nStats after {cs.REPLICATIONS} replications:")

    # Edge Node
    mean_edge, ci_edge = calculate_confidence_interval(stats.edge_wait_times)
    print(f"Edge Node - Average wait time: {mean_edge:.6f} ± {ci_edge:.6f}")

    # Cloud Server
    mean_cloud, ci_cloud = calculate_confidence_interval(stats.cloud_wait_times)
    print(f"Cloud Server - Average wait time: {mean_cloud:.6f} ± {ci_cloud:.6f}")

    # Coordinator Edge
    mean_coord, ci_coord = calculate_confidence_interval(stats.coord_wait_times)
    print(f"Coordinator Edge - Average wait time: {mean_coord:.6f} ± {ci_coord:.6f}")

    # Totali job elaborati (usando count veri, non lunghezza liste)
    total_E = sum(stats.total_count_E) if hasattr(stats, 'total_count_E') else len(stats.edge_wait_times)
    total_C = sum(stats.total_count_C) if hasattr(stats, 'total_count_C') else len(stats.cloud_wait_times)
    print(f"\nTotal jobs E processed: {total_E}")
    print(f"Total jobs C processed: {total_C}")


def plot_analysis(wait_times, seeds, name, sim_type):
    """
    Genera un grafico con le curve dei tempi di attesa (transiente).

    - wait_times: lista di liste, con (tempo, attesa media) per replica
    - seeds: lista di seed usati per riproducibilità
    - name: nome del centro analizzato (edge_node, cloud_server, ecc.)
    - sim_type: tipo di simulazione (standard, better, ecc.)

    Riferimento: Analisi transiente (documento, Sezione 5.1).
    """
    output_dir = os.path.join(file_path, "plot", "transient_analysis", sim_type)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    found = False
    for run_index, response_times in enumerate(wait_times):
        if not response_times:
            continue
        times = [point[0] for point in response_times]
        avg_response_times = [point[1] for point in response_times]
        plt.plot(times, avg_response_times, label=f'Seed {seeds[run_index]}')
        found = True

    plt.xlabel('Time (s)')
    plt.ylabel('Average wait time (s)')
    plt.title(f'Transient Analysis - {name}')
    if found:
        plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, f'{name}.png')
    plt.savefig(output_path)
    plt.close()
