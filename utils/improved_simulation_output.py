"""
utils/improved_simulation_output.py
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
file_path_improved = "output_improved/"

# Intestazione CSV per i risultati della simulazione
# utils/improved_simulation_output.py

header_improved = [
    "seed","slot","lambda",

    # Edge_NuoviArrivi (ex-Edge)
    "edge_NuoviArrivi_avg_wait","edge_NuoviArrivi_avg_delay",
    "edge_NuoviArrivi_L","edge_NuoviArrivi_Lq",
    "edge_NuoviArrivi_utilization","edge_NuoviArrivi_throughput",
    "edge_NuoviArrivi_service_time_mean","Edge_NuoviArrivi_E_Ts",

    # Edge_Feedback (post-Cloud)
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq",
    "edge_Feedback_utilization","edge_Feedback_throughput",
    "edge_Feedback_service_time_mean","Edge_Feedback_E_Ts",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay","cloud_L","cloud_Lq",
    "cloud_avg_busy_servers","cloud_throughput","cloud_service_time_mean",

    # Coordinator
    "coord_avg_wait","coord_avg_delay","coord_L","coord_Lq",
    "coord_utilization","coord_throughput","coord_service_time_mean",

    # (opzionale) classe E per report
    "edge_E_avg_delay","edge_E_avg_response",

    # contatori
    "count_E","count_E_P1","count_E_P2","count_E_P3","count_E_P4","count_C"
]


header_infinite_improved =[
    "seed", "slot", "lambda", "batch"

    # tempi
    "edge_avg_wait", "cloud_avg_wait", "coord_avg_wait",
    "edge_avg_delay", "cloud_avg_delay", "coord_avg_delay",
     "edge_E_avg_delay", "edge_E_avg_response",

    # L, Lq
    "edge_L", "edge_Lq", "cloud_L", "cloud_Lq", "coord_L", "coord_Lq",

    # utilizzazioni per centro
    "edge_utilization", "coord_utilization", "cloud_avg_busy_servers",

    # throughput
    "edge_throughput", "cloud_throughput", "coord_throughput",

    # tempi di servizio realizzati
    "edge_service_time_mean", "cloud_service_time_mean", "coord_service_time_mean",

    # contatori esistenti
    "count_E", "count_E_P1", "count_E_P2", "count_E_P3", "count_E_P4", "count_C",

    # legacy per compatibilità
    "E_utilization", "C_utilization"
]

header_edge_scalability_improved = [
    "seed","lambda","slot",

    # Edge
    "edge_server_number",
    "edge_avg_wait","edge_avg_delay",
    "edge_L","edge_Lq",
    "edge_server_service",
    "edge_server_utilization",
    "edge_weight_utilization",
    "edge_E_avg_delay", "edge_E_avg_response",

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

header_coord_scalability_improved = [
    "seed","lambda","slot",

    # Edge (repliche fisse derivate da CSV Edge)
    "edge_server_number",
    "edge_avg_wait","edge_avg_delay",
    "edge_L","edge_Lq",
    "edge_service_time_mean",
    "edge_avg_busy_servers",
    "edge_throughput",
    "edge_E_avg_delay", "edge_E_avg_response",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay",
    "cloud_L","cloud_Lq",
    "cloud_service_time_mean",
    "cloud_avg_busy_servers",
    "cloud_throughput",

    # Coordinator (SCALABILE)
    "coord_server_number",
    "coord_avg_wait","coord_avg_delay",
    "coord_L","coord_Lq",
    "coord_service_time_mean",
    "coord_utilization",
    "coord_throughput",

    # dettagli scalabilità
    "server_utilization_by_count"
]

# === Header dedicato per la simulazione ad orizzonte infinito ===
infinite_header_improved = [
    "seed", "slot", "lambda", "batch",

    # Edge_NuoviArrivi (ex-Edge)
    "edge_NuoviArrivi_avg_wait","edge_NuoviArrivi_avg_delay",
    "edge_NuoviArrivi_L","edge_NuoviArrivi_Lq",
    "edge_NuoviArrivi_utilization","edge_NuoviArrivi_throughput",
    "edge_NuoviArrivi_service_time_mean","Edge_NuoviArrivi_E_Ts",

    # Edge_Feedback
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq",
    "edge_Feedback_utilization","edge_Feedback_throughput",
    "edge_Feedback_service_time_mean","Edge_Feedback_E_Ts",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay","cloud_L","cloud_Lq",
    "cloud_avg_busy_servers","cloud_throughput","cloud_service_time_mean",

    # Coordinator
    "coord_avg_wait","coord_avg_delay","coord_L","coord_Lq",
    "coord_utilization","coord_throughput","coord_service_time_mean",

    # (opzionale) classe E per report
    "edge_E_avg_delay","edge_E_avg_response",

    # contatori
    "count_E","count_E_P1","count_E_P2","count_E_P3","count_E_P4","count_C"
]

HEADER_MERGED_SCALABILITY_improved = [
    "seed","lambda","slot",

    # Edge_NuoviArrivi
    "edge_server_number",
    "edge_NuoviArrivi_avg_wait","edge_NuoviArrivi_avg_delay",
    "edge_NuoviArrivi_L","edge_NuoviArrivi_Lq",
    "edge_NuoviArrivi_service_time_mean",
    "edge_NuoviArrivi_utilization",
    "edge_NuoviArrivi_throughput",

    # Edge_Feedback
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq",
    "edge_Feedback_service_time_mean",
    "edge_Feedback_utilization",
    "edge_Feedback_throughput",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay","cloud_L","cloud_Lq",
    "cloud_service_time_mean","cloud_avg_busy_servers","cloud_throughput",

    # Coordinator
    "coord_server_number","coord_avg_wait","coord_avg_delay","coord_L","coord_Lq",
    "coord_service_time_mean","coord_utilization","coord_throughput",

    # Meta probabilità (se le usi ancora)
    "pc","p1","p2","p3","p4",

    # Tracce serializzate
    "edge_scal_trace","coord_scal_trace"
]


def clear_merged_scalability_file_improved(file_name: str):
    os.makedirs(file_path_improved, exist_ok=True)
    path = os.path.join(file_path_improved, file_name)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_MERGED_SCALABILITY_improved)
        writer.writeheader()

def write_file_merged_scalability_improved(results: dict, file_name: str):
    os.makedirs(file_path_improved, exist_ok=True)
    path = os.path.join(file_path_improved, file_name)

    row = dict(results)  # copia
    # serializza tracce come JSON compatti
    for k in ("edge_scal_trace", "coord_scal_trace"):
        if k in row and not isinstance(row[k], str):
            row[k] = json.dumps(row[k], separators=(",", ":"))

    # garantisci che tutte le chiavi esistano
    for key in HEADER_MERGED_SCALABILITY_improved:
        row.setdefault(key, None)

    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_MERGED_SCALABILITY_improved)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in HEADER_MERGED_SCALABILITY_improved})


def clear_infinite_file_improved(file_name):
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=infinite_header_improved)
        writer.writeheader()

def write_infinite_row_improved(results, file_name):
    """Scrive una riga per la simulazione ad orizzonte infinito usando 'infinite_header_improved'."""
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)
    # Serializza solo le chiavi presenti nell'header_improved infinito
    results_serialized = {k: results.get(k, "") for k in infinite_header_improved}
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=infinite_header_improved)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_serialized)

def _label_for_sim_improved(sim_type: str) -> str:
    """
    Converte il sim_type interno in un'etichetta parlante per nomi file/cartelle.
    """
    mapping = {
        "lambda_scan": "orizzonte_finito",
        "infinite": "orizzonte_infinito",
        "infinite_horizon": "orizzonte_infinito",
        "lambda_scan_infinite": "orizzonte_infinito",
        "edge_scalability": "scalabilita_edge",
        "coord_scalability": "scalabilita_coordinator"
    }
    return mapping.get(sim_type, sim_type.replace(" ", "_").lower())


def write_file_edge_scalability_improved(results, file_name):
    """
    Scrive i risultati principali della simulazione edge scalability nel CSV.
    Includendo il dizionario di utilizzo per server come JSON stringificato.
    """
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)

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
        writer = csv.DictWriter(csvfile, fieldnames=header_edge_scalability_improved)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_serialized)




def clear_edge_scalability_file_improved(file_name):
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_edge_scalability_improved)
        writer.writeheader()

def write_scalability_trace_improved(trace, seed, lam, slot):
    path = os.path.join(file_path_improved, "edge_scalability_statistics.csv")
    os.makedirs(file_path_improved, exist_ok=True)

    header = ["seed", "lambda", "slot", "time", "edge_servers", "E_utilization"]
    file_exists = os.path.isfile(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for time, servers, utilization in trace:
            writer.writerow([seed, lam, slot, time, servers, utilization])

def clear_file_improved(file_name):
    """
    Pulisce il file di output e scrive l'intestazione.

    Riferimento: per ogni nuova simulazione (Sezione 5.2),
    serve un file CSV fresco per raccogliere i dati delle repliche.
    """
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_improved)
        writer.writeheader()


def write_file_improved(results, file_name):
    """
    Scrive i risultati di una replica nel file CSV.

    Ogni riga corrisponde a una replica con un seed diverso.
    """
    path = os.path.join(file_path_improved, file_name)
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_improved)
        writer.writerow(results)


def _print_ci_improved(label, data):
    if data:
        mean, ci = calculate_confidence_interval(data)
        print(f"{label}: {mean:.6f} ± {ci:.6f}")


def _ci_or_none_improved(arr):
    if arr:
        mean, ci = calculate_confidence_interval(arr)
        return float(mean), float(ci)
    return None, None

def build_summary_dict_improved(stats, sim_type):
    """Costruisce un dizionario con tutte le metriche mostrate a schermo, con mean e ±CI."""
    is_infinite = sim_type in ("lambda_scan_infinite","infinite","infinite_horizon")
    header = "Infinite Horizon Simulation - Batch Means Summary" if is_infinite \
             else f"Stats after {cs.REPLICATIONS} replications"

    # prepara tutte le coppie (mean, ci)
    summary = {
        "header_improved": header,
        "sim_type": sim_type,
        "replications": cs.REPLICATIONS,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": []
    }

    def add(label, arr):
        mean, ci = _ci_or_none_improved(arr)
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

    # attese in coda e rispsota classe E edge
    add("Edge Node (Class E) - Average delay", stats.edge_E_delay_times)
    add("Edge Node (Class E) - Average response", stats.edge_E_response_times)

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

def save_summary_json_improved(summary, sim_type, filename=None):
    """
    Salva il riepilogo in JSON ordinato in una cartella per tipo di analisi.
    Esempio: output/summaries/orizzonte_infinito/summary_orizzonte_infinito_20250811_153045.json
    """
    label = _label_for_sim_improved(sim_type)
    out_dir = os.path.join(file_path_improved, "summaries", label)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{label}_{ts}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def save_summary_txt_improved(summary, sim_type, filename=None):
    """
    (Opzionale) Salva un .txt simile alla stampa a schermo in una cartella per tipo di analisi.
    Esempio: output/summaries/scalabilita_edge/summary_scalabilita_edge_20250811_153210.txt
    """
    label = _label_for_sim_improved(sim_type)
    out_dir = os.path.join(file_path_improved, "summaries", label)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{label}_{ts}.txt"
    path = os.path.join(out_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {summary['header_improved']} ===\n")
        for m in summary["metrics"]:
            f.write(f"{m['name']}: {m['mean']:.6f} ± {m['ci_95']:.6f}\n")
    return path

def print_simulation_stats_improved(stats, sim_type):


    is_infinite = sim_type in ("lambda_scan_infinite","infinite","infinite_horizon")
    print("\n=== Infinite Horizon Simulation - Batch Means Summary ===" if is_infinite
          else f"\nStats after {cs.REPLICATIONS} replications:")



    _print_ci_improved("Edge_NuoviArrivi - Average wait time", stats.edge_wait_times)
    _print_ci_improved("Edge_NuoviArrivi (Class E) - Avg delay", stats.edge_E_delay_times)
    _print_ci_improved("Edge_NuoviArrivi (Class E) - Avg response", stats.edge_E_response_times)
    _print_ci_improved("Edge_NuoviArrivi - Average delay (queue)", getattr(stats, 'edge_delay_times', []))
    _print_ci_improved("Edge_NuoviArrivi - Avg number in node (L)", getattr(stats, 'edge_L', []))
    _print_ci_improved("Edge_NuoviArrivi - Avg number in queue (Lq)", getattr(stats, 'edge_Lq', []))
    _print_ci_improved("Edge_NuoviArrivi - Utilization", getattr(stats, 'edge_utilization', []))
    _print_ci_improved("Edge_NuoviArrivi - Throughput", getattr(stats, 'edge_X', []))

    _print_ci_improved("Cloud Server - Average wait time", stats.cloud_wait_times)
    _print_ci_improved("Coordinator Edge - Average wait time", stats.coord_wait_times)

    # tempi di e attesa risposta per job di classe E


    # nuovi: tempi d'attesa in coda

    _print_ci_improved("Cloud Server - Average delay (queue)", getattr(stats, 'cloud_delay_times', []))
    _print_ci_improved("Coordinator Edge - Average delay (queue)", getattr(stats, 'coord_delay_times', []))

    # L e Lq

    _print_ci_improved("Cloud Server - Avg number in node (L)", getattr(stats, 'cloud_L', []))
    _print_ci_improved("Cloud Server - Avg number in queue (Lq)", getattr(stats, 'cloud_Lq', []))
    _print_ci_improved("Coordinator - Avg number in node (L)", getattr(stats, 'coord_L', []))
    _print_ci_improved("Coordinator - Avg number in queue (Lq)", getattr(stats, 'coord_Lq', []))

    # utilizzazioni

    _print_ci_improved("Coordinator - Utilization", getattr(stats, 'coord_utilization', []))
    _print_ci_improved("Cloud - Avg busy servers", getattr(stats, 'cloud_busy', []))

    # throughput

    _print_ci_improved("Cloud - Throughput", getattr(stats, 'cloud_X', []))
    _print_ci_improved("Coordinator - Throughput", getattr(stats, 'coord_X', []))



def plot_analysis_improved(wait_times, seeds, name, sim_type):
    """
    Genera un grafico con le curve dei tempi di attesa (transiente),
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/transient_analysis/orizzonte_finito/<name>.png
    """
    label = _label_for_sim_improved(sim_type)
    output_dir = os.path.join(file_path_improved, "plot", "transient_analysis", label)
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

    plt.xlabel('Time_improved (s)')
    plt.ylabel('Average wait time (s)')
    plt.title(f'Transient Analysis - {name}')
    if found:
        plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, f'{name}.png')
    plt.savefig(output_path)
    plt.close()


def plot_multi_lambda_per_seed_improved(wait_times, seeds, name, sim_type, lambdas, slots):
    """
    Genera un grafico per seed confrontando più λ,
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/multi_lambda/orizzonte_finito/<name>_seed<seed>.png
    """
    label = _label_for_sim_improved(sim_type)
    output_dir = os.path.join(file_path_improved, "plot", "multi_lambda", label)
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

        plt.xlabel("Time_improved (s)")
        plt.ylabel("Average wait time (s)")
        plt.title(f"Multi-λ Analysis - {name} | Seed {seed}")
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_seed{seed}.png")
        plt.savefig(output_path)
        plt.close()


def plot_multi_seed_per_lambda_improved(wait_times, seeds, name, sim_type, lambdas, slots):
    """
    Genera un grafico per λ confrontando seed diversi,
    salvando in una cartella per tipo di analisi.
    Esempio: output/plot/multi_seed/orizzonte_finito/<name>_lambda0.2000.png
    """
    label = _label_for_sim_improved(sim_type)
    output_dir = os.path.join(file_path_improved, "plot", "multi_seed", label)
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

        plt.xlabel("Time_improved (s)")
        plt.ylabel("Average wait time (s)")
        plt.title(f"Multi-seed Analysis - {name} | λ={lam:.4f}")
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{name}_lambda{lam:.4f}.png")
        plt.savefig(output_path)
        plt.close()


def write_file_coord_scalability_improved(results, file_name):
    """Scrive i risultati della simulazione Coordinator Scalability nel CSV."""
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)

    # Copia e serializza i campi complessi
    results_serialized = results.copy()

    # serializza il dict di utilizzi
    if "server_utilization_by_count" in results_serialized:
        results_serialized["server_utilization_by_count"] = json.dumps(
            results_serialized["server_utilization_by_count"], separators=(',', ':')
        )

    # ⚠️ rimuovi chiavi non presenti nell'header_improved (come 'scalability_trace')
    allowed = set(header_coord_scalability_improved)
    results_serialized = {k: v for k, v in results_serialized.items() if k in allowed}

    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_coord_scalability_improved)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_serialized)


def clear_coord_scalability_file_improved(file_name):
    path = os.path.join(file_path_improved, file_name)
    os.makedirs(file_path_improved, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_coord_scalability_improved)
        writer.writeheader()


# ---------------------------- PLOT FOR INFINITE SIMULATIONS -------------------------------
def plot_infinite_analysis_improved():
    """
    Grafici per l'analisi a orizzonte infinito:
    - Nodi (Edge/Cloud/Coordinator): x = batch, linee per ciascun λ reale
    - Curva Edge response time vs λ con punti per slot e linea QoS = 3s
    Salva tutto in output/plot/orizzonte infinito
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(file_path_improved)
    csv_path = out_dir / "infinite_statistics.csv"
    if not csv_path.exists():
        print(f"[plot_infinite_analysis] File non trovato: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[plot_infinite_analysis] Nessun dato in infinite_statistics.csv")
        return

    # Assicura la colonna 'batch'
    if 'batch' not in df.columns:
        df = df.sort_values(["slot", "lambda"]).copy()
        df['batch'] = df.groupby(["slot", "lambda"]).cumcount()
    else:
        df = df.sort_values(["lambda", "batch"]).copy()

    plot_dir = out_dir / "plot" / "orizzonte infinito"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Nodi: x = batch, linee per ciascun λ reale ----------
    agg_lam_batch = df.groupby(["lambda", "batch"], as_index=False).agg({
        "edge_NuoviArrivi_avg_wait": "mean",
        "cloud_avg_wait": "mean",
        "coord_avg_wait": "mean",
    })

    def plot_node_by_lambda(ycol, title, fname):
        plt.figure()
        any_line = False
        for lam_val, g in agg_lam_batch.groupby("lambda"):
            g = g.sort_values("batch")
            if not g.empty:
                plt.plot(g["batch"], g[ycol], marker="", label=f"λ={lam_val:.5f}")
                any_line = True
        plt.xlabel("Batch")
        plt.ylabel("Tempo medio di risposta (s)")
        plt.title(title)
        if any_line:
            plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()


    plot_node_by_lambda("cloud_avg_wait",
                        "Nodo Cloud: tempo di risposta per batch (linee per λ)",
                        "infinite_cloud_response_vs_batch_per_lambda.png")
    plot_node_by_lambda("coord_avg_wait",
                        "Nodo Coordinator: tempo di risposta per batch (linee per λ)",
                        "infinite_coord_response_vs_batch_per_lambda.png")

    # ---------- 2) Curva Edge response time vs λ con punti per slot + QoS ----------
    agg_lambda_global = df.groupby("lambda", as_index=False).agg({
        "edge_NuoviArrivi_avg_wait": "mean"
    }).sort_values("lambda")

    agg_lambda_slot = df.groupby(["slot", "lambda"], as_index=False).agg({
        "edge_NuoviArrivi_avg_wait": "mean"
    }).sort_values(["lambda", "slot"])

    qos_seconds = 3.0  # QoS richiesto = 3s

    plt.figure()
    # Curva media globale (λ -> W_edge)
    if not agg_lambda_global.empty:
        plt.plot(agg_lambda_global["lambda"], agg_lambda_global["edge_NuoviArrivi_avg_wait"], marker="o", color="blue")
    # Punti per slot
    for slot, g in agg_lambda_slot.groupby("slot"):
        if not g.empty:
            plt.scatter(g["lambda"], g["edge_NuoviArrivi_avg_wait"], label=f"Slot {slot}")
            for _, r in g.iterrows():
                plt.annotate(f"{r['lambda']:.5f}", (r["lambda"], r["edge_NuoviArrivi_avg_wait"]),
                             textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=8)
    # Linea QoS
    plt.axhline(y=qos_seconds, linestyle="--", color="r", linewidth=1.2,
                label=f"QoS = {qos_seconds:.0f}s")
    plt.xlabel("λ (arrivi/secondo)")
    plt.ylabel("Tempo medio di risposta Edge_NuoviArrivi (s)")
    plt.title("Tempo di risposta Edge_NuoviArrivi vs λ (con QoS = 3s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / "infinite_edge_response_vs_lambda_with_qos.png",
                dpi=150, bbox_inches="tight")
    plt.close()
