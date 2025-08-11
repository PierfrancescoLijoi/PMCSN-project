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
    "slot",          # nuovo campo
    "lambda",        # nuovo campo
    "edge_avg_wait",
    "cloud_avg_wait",
    "coord_avg_wait",
    "count_E",
    "count_E_P1",
    "count_E_P2",
    "count_E_P3",
    "count_E_P4",
    "count_C",
    "E_utilization",
    "C_utilization"
]
import json  # Assicurati che sia importato all'inizio del file

import json  # Assicurati che sia in cima al file

import json

header_edge_scalability = [
    "seed", "lambda", "slot",
    "edge_server_number",
    "edge_avg_wait",
    "edge_avg_delay",
    "edge_server_service",
    "edge_server_utilization",
    "edge_weight_utilization",
    "server_utilization_by_count"  # nuovo campo JSON
]

def write_file_edge_scalability(results, file_name):
    """
    Scrive i risultati principali della simulazione edge scalability nel CSV.
    Includendo il dizionario di utilizzo per server come JSON stringificato.
    """
    path = os.path.join(file_path, file_name)
    os.makedirs(file_path, exist_ok=True)

    results_serialized = results.copy()

    # Converti il dizionario in stringa JSON compatta
    if "server_utilization_by_count" in results_serialized:
        results_serialized["server_utilization_by_count"] = json.dumps(
            results_serialized["server_utilization_by_count"], separators=(',', ':')
        )

    # Rimuovi altri campi inutili dal dizionario
    for key in ["scalability_trace", "edge_servers"]:
        if key in results_serialized:
            del results_serialized[key]

    file_exists = os.path.isfile(path)

    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_edge_scalability)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_serialized)


file_path = "output/"  # assicurati che questa variabile sia definita


def clear_edge_scalability_file(file_name):
    path = os.path.join(file_path, file_name)
    os.makedirs(file_path, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_edge_scalability)
        writer.writeheader()

def write_scalability_trace(trace, seed, lam, slot):
    path = os.path.join(file_path, "edge_scalability_statistics.csv")
    os.makedirs(file_path, exist_ok=True)

    header = ["seed", "lambda", "slot", "time", "edge_servers", "E_utilization"]
    file_exists = os.path.isfile(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for time, servers, utilization in trace:
            writer.writerow([seed, lam, slot, time, servers, utilization])

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


def print_simulation_stats(stats, sim_type):
    """
    Stampa un riepilogo delle statistiche di simulazione.
    - Per l'orizzonte finito: mostra le medie e intervalli di confidenza dopo N repliche.
    - Per l'orizzonte infinito: mostra le medie e intervalli di confidenza calcolati sui batch.
    """
    is_infinite = sim_type in ("lambda_scan_infinite", "infinite", "infinite_horizon")

    if is_infinite:
        print("\n=== Infinite Horizon Simulation - Batch Means Summary ===")
    else:
        print(f"\nStats after {cs.REPLICATIONS} replications:")

    # Edge Node
    if stats.edge_wait_times:
        mean_edge, ci_edge = calculate_confidence_interval(stats.edge_wait_times)
        label = "Edge Node - Average wait time"
        print(f"{label}: {mean_edge:.6f} ± {ci_edge:.6f}")

    # Cloud Server
    if stats.cloud_wait_times:
        mean_cloud, ci_cloud = calculate_confidence_interval(stats.cloud_wait_times)
        label = "Cloud Server - Average wait time"
        print(f"{label}: {mean_cloud:.6f} ± {ci_cloud:.6f}")

    # Coordinator Edge
    if stats.coord_wait_times:
        mean_coord, ci_coord = calculate_confidence_interval(stats.coord_wait_times)
        label = "Coordinator Edge - Average wait time"
        print(f"{label}: {mean_coord:.6f} ± {ci_coord:.6f}")

    # Conteggi totali (se presenti)
    if hasattr(stats, 'total_count_E'):
        total_E = sum(stats.total_count_E)
        print(f"\nTotal jobs E processed: {total_E}")
    if hasattr(stats, 'total_count_C'):
        total_C = sum(stats.total_count_C)
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

def plot_multi_lambda_per_seed(wait_times, seeds, name, sim_type, lambdas, slots):
    """
    Genera un grafico per ogni seed, confrontando più fasce di λ.

    - wait_times: lista di liste [(tempo, attesa), ...] per ogni run
    - seeds: lista di seed usati
    - name: centro analizzato (edge_node, cloud_server, ecc.)
    - sim_type: tipo di simulazione (es. lambda_scan)
    - lambdas: lista dei λ usati (uno per ciascun run)
    - slots: lista degli indici slot (uno per ciascun run)
    """

    output_dir = os.path.join(file_path, "plot", "multi_lambda", sim_type)
    os.makedirs(output_dir, exist_ok=True)

    # Raggruppo per seed
    seed_to_runs = {}
    for idx, seed in enumerate(seeds):
        if not wait_times[idx]:
            continue
        if seed not in seed_to_runs:
            seed_to_runs[seed] = []
        seed_to_runs[seed].append((slots[idx], lambdas[idx], wait_times[idx]))

    # Creo un grafico per ogni seed
    for seed, runs in seed_to_runs.items():
        plt.figure(figsize=(10, 6))

        for slot, lam, response_times in runs:
            times = [pt[0] for pt in response_times]
            avg_response_times = [pt[1] for pt in response_times]
            plt.plot(times, avg_response_times, label=f"Slot {slot} (λ={lam:.4f})")

        plt.xlabel("Time (s)")
        plt.ylabel("Average wait time (s)")
        plt.title(f"Multi-λ Analysis - {name} | Seed {seed}")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_seed{seed}.png")
        plt.savefig(output_path)
        plt.close()
def plot_multi_seed_per_lambda(wait_times, seeds, name, sim_type, lambdas, slots):
    """
    Genera un grafico per ogni λ, confrontando repliche con seed diversi.

    - wait_times: lista di liste [(tempo, attesa), ...] per ogni run
    - seeds: lista di seed usati
    - name: centro analizzato (edge_node, cloud_server, ecc.)
    - sim_type: tipo di simulazione (es. lambda_scan)
    - lambdas: lista dei λ usati
    - slots: lista degli indici slot
    """

    output_dir = os.path.join(file_path, "plot", "multi_seed", sim_type)
    os.makedirs(output_dir, exist_ok=True)

    # Raggruppo per λ
    lambda_to_runs = {}
    for idx, lam in enumerate(lambdas):
        if not wait_times[idx]:
            continue
        if lam not in lambda_to_runs:
            lambda_to_runs[lam] = []
        lambda_to_runs[lam].append((seeds[idx], slots[idx], wait_times[idx]))

    # Creo un grafico per ogni λ
    for lam, runs in lambda_to_runs.items():
        plt.figure(figsize=(10, 6))

        for seed, slot, response_times in runs:
            times = [pt[0] for pt in response_times]
            avg_response_times = [pt[1] for pt in response_times]
            plt.plot(times, avg_response_times, label=f"Seed {seed} (Slot {slot})")

        plt.xlabel("Time (s)")
        plt.ylabel("Average wait time (s)")
        plt.title(f"Multi-seed Analysis - {name} | λ={lam:.4f}")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_lambda{lam:.4f}.png")
        plt.savefig(output_path)
        plt.close()
