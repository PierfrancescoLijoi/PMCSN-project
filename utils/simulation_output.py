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
from datetime import datetime
import json
# Directory di output
file_path = "output/"

# Intestazione CSV per i risultati della simulazione
header = [
    "seed", "slot", "lambda",

    # tempi
    "edge_avg_wait","cloud_avg_wait","coord_avg_wait",
    "edge_avg_delay","cloud_avg_delay","coord_avg_delay",

    # L, Lq
    "edge_L","edge_Lq","cloud_L","cloud_Lq","coord_L","coord_Lq",

    # utilizzazioni per centro
    "edge_utilization","coord_utilization","cloud_avg_busy_servers",

    # throughput
    "edge_throughput","cloud_throughput","coord_throughput",

    # tempi di servizio realizzati
    "edge_service_time_mean","cloud_service_time_mean","coord_service_time_mean",

    # contatori esistenti
    "count_E","count_E_P1","count_E_P2","count_E_P3","count_E_P4","count_C",

    # legacy per compatibilità
    "E_utilization","C_utilization"
]

import json  # Assicurati che sia importato all'inizio del file

import json  # Assicurati che sia in cima al file

import json

header_edge_scalability = [
    "seed","lambda","slot",

    # Edge
    "edge_server_number",
    "edge_avg_wait","edge_avg_delay",
    "edge_L","edge_Lq",
    "edge_server_service",
    "edge_server_utilization",
    "edge_weight_utilization",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay",
    "cloud_L","cloud_Lq",
    "cloud_service_time_mean",
    "cloud_avg_busy_servers",
    "cloud_throughput",

    # Coordinator
    "coord_avg_wait","coord_avg_delay",
    "coord_L","coord_Lq",
    "coord_service_time_mean",
    "coord_utilization",
    "coord_throughput",

    # dettagli scalabilità
    "server_utilization_by_count"
]
def _label_for_sim(sim_type: str) -> str:
    """
    Converte il sim_type interno in un'etichetta parlante per nomi file/cartelle.
    """
    mapping = {
        "lambda_scan": "orizzonte_finito",
        "infinite": "orizzonte_infinito",
        "infinite_horizon": "orizzonte_infinito",
        "lambda_scan_infinite": "orizzonte_infinito",
        "edge_scalability": "scalabilita_edge",
    }
    return mapping.get(sim_type, sim_type.replace(" ", "_").lower())


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


def _print_ci(label, data):
    if data:
        mean, ci = calculate_confidence_interval(data)
        print(f"{label}: {mean:.6f} ± {ci:.6f}")


def _ci_or_none(arr):
    if arr:
        mean, ci = calculate_confidence_interval(arr)
        return float(mean), float(ci)
    return None, None

def build_summary_dict(stats, sim_type):
    """Costruisce un dizionario con tutte le metriche mostrate a schermo, con mean e ±CI."""
    is_infinite = sim_type in ("lambda_scan_infinite","infinite","infinite_horizon")
    header = "Infinite Horizon Simulation - Batch Means Summary" if is_infinite \
             else f"Stats after {cs.REPLICATIONS} replications"

    # prepara tutte le coppie (mean, ci)
    summary = {
        "header": header,
        "sim_type": sim_type,
        "replications": cs.REPLICATIONS,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": []
    }

    def add(label, arr):
        mean, ci = _ci_or_none(arr)
        if mean is not None:
            summary["metrics"].append({
                "name": label,
                "mean": mean,
                "ci_95": ci
            })

    # tempi di risposta
    add("Edge Node - Average wait time",  stats.edge_wait_times)
    add("Cloud Server - Average wait time", stats.cloud_wait_times)
    add("Coordinator Edge - Average wait time", stats.coord_wait_times)

    # attese in coda
    add("Edge Node - Average delay (queue)",  getattr(stats, 'edge_delay_times', []))
    add("Cloud Server - Average delay (queue)", getattr(stats, 'cloud_delay_times', []))
    add("Coordinator Edge - Average delay (queue)", getattr(stats, 'coord_delay_times', []))

    # L e Lq
    add("Edge Node - Avg number in node (L)", getattr(stats, 'edge_L', []))
    add("Edge Node - Avg number in queue (Lq)", getattr(stats, 'edge_Lq', []))
    add("Cloud Server - Avg number in node (L)", getattr(stats, 'cloud_L', []))
    add("Cloud Server - Avg number in queue (Lq)", getattr(stats, 'cloud_Lq', []))
    add("Coordinator - Avg number in node (L)", getattr(stats, 'coord_L', []))
    add("Coordinator - Avg number in queue (Lq)", getattr(stats, 'coord_Lq', []))

    # utilizzazioni
    add("Edge Node - Utilization", getattr(stats, 'edge_utilization', []))
    add("Coordinator - Utilization", getattr(stats, 'coord_utilization', []))
    add("Cloud - Avg busy servers", getattr(stats, 'cloud_busy', []))

    # throughput
    add("Edge Node - Throughput", getattr(stats, 'edge_X', []))
    add("Cloud - Throughput", getattr(stats, 'cloud_X', []))
    add("Coordinator - Throughput", getattr(stats, 'coord_X', []))

    return summary

def save_summary_json(summary, sim_type, filename=None):
    """
    Salva il riepilogo in JSON ordinato in una cartella per tipo di analisi.
    Esempio: output/summaries/orizzonte_infinito/summary_orizzonte_infinito_20250811_153045.json
    """
    label = _label_for_sim(sim_type)
    out_dir = os.path.join(file_path, "summaries", label)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{label}_{ts}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def save_summary_txt(summary, sim_type, filename=None):
    """
    (Opzionale) Salva un .txt simile alla stampa a schermo in una cartella per tipo di analisi.
    Esempio: output/summaries/scalabilita_edge/summary_scalabilita_edge_20250811_153210.txt
    """
    label = _label_for_sim(sim_type)
    out_dir = os.path.join(file_path, "summaries", label)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{label}_{ts}.txt"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {summary['header']} ===\n")
        for m in summary["metrics"]:
            f.write(f"{m['name']}: {m['mean']:.6f} ± {m['ci_95']:.6f}\n")
    return path

def print_simulation_stats(stats, sim_type):
    # Costruisci il riepilogo per file JSON/TXT
    summary = build_summary_dict(stats, sim_type)
    is_infinite = sim_type in ("lambda_scan_infinite","infinite","infinite_horizon")
    print("\n=== Infinite Horizon Simulation - Batch Means Summary ===" if is_infinite
          else f"\nStats after {cs.REPLICATIONS} replications:")

    # tempi di risposta
    _print_ci("Edge Node - Average wait time",  stats.edge_wait_times)
    _print_ci("Cloud Server - Average wait time", stats.cloud_wait_times)
    _print_ci("Coordinator Edge - Average wait time", stats.coord_wait_times)

    # nuovi: tempi d'attesa in coda
    _print_ci("Edge Node - Average delay (queue)",  getattr(stats, 'edge_delay_times', []))
    _print_ci("Cloud Server - Average delay (queue)", getattr(stats, 'cloud_delay_times', []))
    _print_ci("Coordinator Edge - Average delay (queue)", getattr(stats, 'coord_delay_times', []))

    # L e Lq
    _print_ci("Edge Node - Avg number in node (L)", getattr(stats, 'edge_L', []))
    _print_ci("Edge Node - Avg number in queue (Lq)", getattr(stats, 'edge_Lq', []))
    _print_ci("Cloud Server - Avg number in node (L)", getattr(stats, 'cloud_L', []))
    _print_ci("Cloud Server - Avg number in queue (Lq)", getattr(stats, 'cloud_Lq', []))
    _print_ci("Coordinator - Avg number in node (L)", getattr(stats, 'coord_L', []))
    _print_ci("Coordinator - Avg number in queue (Lq)", getattr(stats, 'coord_Lq', []))

    # utilizzazioni
    _print_ci("Edge Node - Utilization", getattr(stats, 'edge_utilization', []))
    _print_ci("Coordinator - Utilization", getattr(stats, 'coord_utilization', []))
    _print_ci("Cloud - Avg busy servers", getattr(stats, 'cloud_busy', []))

    # throughput
    _print_ci("Edge Node - Throughput", getattr(stats, 'edge_X', []))
    _print_ci("Cloud - Throughput", getattr(stats, 'cloud_X', []))
    _print_ci("Coordinator - Throughput", getattr(stats, 'coord_X', []))

    # salvataggio persistente
    json_path = save_summary_json(summary, sim_type)
    txt_path = save_summary_txt(summary, sim_type)  # se non vuoi il .txt, commenta questa riga

    print(f"\n[Saved summaries] JSON: {json_path}  TXT: {txt_path}")


def plot_analysis(wait_times, seeds, name, sim_type):
    """
    Genera un grafico con le curve dei tempi di attesa (transiente),
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/transient_analysis/orizzonte_finito/<name>.png
    """
    label = _label_for_sim(sim_type)
    output_dir = os.path.join(file_path, "plot", "transient_analysis", label)
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
    Genera un grafico per seed confrontando più λ,
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/multi_lambda/orizzonte_finito/<name>_seed<seed>.png
    """
    label = _label_for_sim(sim_type)
    output_dir = os.path.join(file_path, "plot", "multi_lambda", label)
    os.makedirs(output_dir, exist_ok=True)

    # Raggruppo per seed
    seed_to_runs = {}
    for idx, seed in enumerate(seeds):
        if not wait_times[idx]:
            continue
        seed_to_runs.setdefault(seed, []).append((slots[idx], lambdas[idx], wait_times[idx]))

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
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_seed{seed}.png")
        plt.savefig(output_path)
        plt.close()


def plot_multi_seed_per_lambda(wait_times, seeds, name, sim_type, lambdas, slots):
    """
    Genera un grafico per λ confrontando seed diversi,
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/multi_seed/orizzonte_finito/<name>_lambda0.2000.png
    """
    label = _label_for_sim(sim_type)
    output_dir = os.path.join(file_path, "plot", "multi_seed", label)
    os.makedirs(output_dir, exist_ok=True)

    # Raggruppo per λ
    lambda_to_runs = {}
    for idx, lam in enumerate(lambdas):
        if not wait_times[idx]:
            continue
        lambda_to_runs.setdefault(lam, []).append((seeds[idx], slots[idx], wait_times[idx]))

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
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_lambda{lam:.4f}.png")
        plt.savefig(output_path)
        plt.close()
