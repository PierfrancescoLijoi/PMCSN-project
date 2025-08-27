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
    "edge_E_avg_delay", "edge_E_avg_response",
    "edge_C_avg_delay","edge_C_avg_response",

    # L, Lq (totali)
    "edge_L","edge_Lq","cloud_L","cloud_Lq","coord_L","coord_Lq",
    # L, Lq per classe (Edge)  ← NEW
    "edge_E_L","edge_E_Lq","edge_C_L","edge_C_Lq",

    # utilizzazioni per centro
    "edge_utilization","coord_utilization","cloud_avg_busy_servers",
    # (se le esporti) utilizzazioni per classe (Edge)
    "edge_E_utilization","edge_C_utilization",

    # throughput
    "edge_throughput","cloud_throughput","coord_throughput",

    # tempi di servizio realizzati
    "edge_service_time_mean","cloud_service_time_mean","coord_service_time_mean",

    # contatori esistenti
    "count_E","count_E_P1","count_E_P2","count_E_P3","count_E_P4","count_C",

    # legacy (se vuoi mantenerli)
    "E_utilization","C_utilization",
]

# === Header dedicato per la simulazione ad orizzonte infinito ===
infinite_header = [
    "seed", "slot", "lambda", "batch",
    # tempi medi per nodo
    "edge_avg_wait","cloud_avg_wait","coord_avg_wait",
    "edge_avg_delay","cloud_avg_delay","coord_avg_delay",
    "edge_E_avg_delay","edge_E_avg_response",
    "edge_C_avg_delay","edge_C_avg_response",
    # L, Lq (totali + per classe Edge)
    "edge_L", "edge_Lq", "cloud_L", "cloud_Lq", "coord_L", "coord_Lq",
    "edge_E_L", "edge_E_Lq", "edge_C_L", "edge_C_Lq",
    # utilizzazioni / busy servers
    "edge_utilization", "coord_utilization", "cloud_avg_busy_servers",
    "edge_E_utilization", "edge_C_utilization",
    # throughput
    "edge_throughput","cloud_throughput","coord_throughput",
    # tempi di servizio
    "edge_service_time_mean", "cloud_service_time_mean", "coord_service_time_mean",

    # conteggi per classi
    "count_E", "count_E_P1", "count_E_P2", "count_E_P3", "count_E_P4", "count_C",

    # legacy (se li vuoi ancora)
    "E_utilization", "C_utilization",

]


HEADER_MERGED_SCALABILITY = [
    "seed","lambda","slot",

    # Edge (con per-classe)
    "edge_server_number",
    "edge_avg_wait","edge_avg_delay",
    "edge_L","edge_Lq","edge_service_time_mean",
    "edge_avg_busy_servers","edge_throughput",
    "edge_utilization","edge_E_utilization","edge_C_utilization",
    "edge_E_avg_delay","edge_E_avg_response",
    "edge_C_avg_delay","edge_C_avg_response",
    "edge_E_L","edge_E_Lq","edge_C_L","edge_C_Lq",

    # Cloud
    "cloud_avg_wait","cloud_avg_delay","cloud_L","cloud_Lq",
    "cloud_service_time_mean","cloud_avg_busy_servers","cloud_throughput",

    # Coordinator
    "coord_server_number","coord_avg_wait","coord_avg_delay",
    "coord_L","coord_Lq","coord_service_time_mean",
    "coord_utilization","coord_throughput",

    # Meta probabilità
    "pc","p1","p2","p3","p4",

    # Tracce serializzate
    "edge_scal_trace","coord_scal_trace"
]

def clear_merged_scalability_file(file_name: str):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, file_name)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_MERGED_SCALABILITY)
        writer.writeheader()

def write_file_merged_scalability(results: dict, file_name: str):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, file_name)

    row = dict(results)  # copia
    # serializza tracce come JSON compatti
    for k in ("edge_scal_trace", "coord_scal_trace"):
        if k in row and not isinstance(row[k], str):
            row[k] = json.dumps(row[k], separators=(",", ":"))

    # garantisci che tutte le chiavi esistano
    for key in HEADER_MERGED_SCALABILITY:
        row.setdefault(key, None)

    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_MERGED_SCALABILITY)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in HEADER_MERGED_SCALABILITY})


def clear_infinite_file(file_name):
    path = os.path.join(file_path, file_name)
    os.makedirs(file_path, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=infinite_header)
        writer.writeheader()

def write_infinite_row(results, file_name):
    """Scrive una riga per la simulazione ad orizzonte infinito usando 'infinite_header'."""
    path = os.path.join(file_path, file_name)
    os.makedirs(file_path, exist_ok=True)
    # Serializza solo le chiavi presenti nell'header infinito
    results_serialized = {k: results.get(k, "") for k in infinite_header}
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=infinite_header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_serialized)

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
        "coord_scalability": "scalabilita_coordinator"
    }
    return mapping.get(sim_type, sim_type.replace(" ", "_").lower())


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

    # tempi di e attesa risposta per job di classe E
    _print_ci("Edge Node (Class E) - Avg delay", stats.edge_E_delay_times)
    _print_ci("Edge Node (Class E) - Avg response", stats.edge_E_response_times)

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



def plot_multi_seed_per_lambda(
    wait_times, seeds, name, sim_type, lambdas, slots,
    edge_E_wait_times=None, edge_C_wait_times=None
):
    """
    Genera un grafico per λ confrontando seed diversi.
    Cartella: output/plot/multi_seed/<label>/
    File:     <name>_lambda<lam>.png

    Se 'name' è dell'EDGE e vengono passate anche le serie per classe,
    dopo il grafico globale crea automaticamente:
      - <name>_class_E_lambda<lam>.png  (tempo di risposta pacchetti E all'Edge)
      - <name>_class_C_lambda<lam>.png  (tempo di risposta pacchetti C all'Edge)
    """
    label = _label_for_sim(sim_type)
    output_dir = os.path.join(file_path, "plot", "multi_seed", label)
    os.makedirs(output_dir, exist_ok=True)

    def _plot_one(series_list, file_suffix="", title_suffix="", ylabel_override=None):
        # Raggruppo per λ
        lambda_to_runs = {}
        for idx, lam in enumerate(lambdas):
            if not series_list[idx]:
                continue
            lambda_to_runs.setdefault(lam, []).append((seeds[idx], slots[idx], series_list[idx]))

        # Creo un grafico per ogni λ
        for lam, runs in lambda_to_runs.items():
            plt.figure(figsize=(10, 6))
            for seed, slot, response_times in runs:
                times = [pt[0] for pt in response_times]
                values = [pt[1] for pt in response_times]
                plt.plot(times, values, label=f"Seed {seed} (Slot {slot})")

            plt.xlabel("Time (s)")
            plt.ylabel(ylabel_override if ylabel_override else "Average wait time (s)")
            plt.title(f"Multi-seed Analysis - {name}{title_suffix} | λ={lam:.4f}")
            plt.grid(True)
            # (lasciamo senza legend come nella funzione originale)

            out = os.path.join(output_dir, f"{name}{file_suffix}_lambda{lam:.4f}.png")
            plt.savefig(out)
            plt.close()

    # 1) Grafico EDGE/CLOUD/COORD "globale" (comportamento originale)
    _plot_one(wait_times)

    # 2) Se è l’EDGE e ho le serie per classe, produco anche E e C
    is_edge = name.lower().startswith("edge")
    if is_edge and (edge_E_wait_times is not None) and (edge_C_wait_times is not None):
        _plot_one(edge_E_wait_times, file_suffix="_class_E", title_suffix=" [Class E]",
                  ylabel_override="Average response time (s)")
        _plot_one(edge_C_wait_times, file_suffix="_class_C", title_suffix=" [Class C]",
                  ylabel_override="Average response time (s)")




# ---------------------------- PLOT FOR INFINITE SIMULATIONS -------------------------------
def plot_infinite_analysis():
    """
    Grafici per l'analisi a orizzonte infinito:
    - Nodi (Edge/Cloud/Coordinator): x = batch, linee per ciascun λ reale
    - Curva Edge response time vs λ con punti per slot e linea QoS = 3s
    Salva tutto in output/plot/orizzonte infinito
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(file_path)
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
        "edge_avg_wait": "mean",
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

    plot_node_by_lambda("edge_avg_wait",
                        "Nodo Edge: tempo di risposta per batch (linee per λ)",
                        "infinite_edge_response_vs_batch_per_lambda.png")
    plot_node_by_lambda("cloud_avg_wait",
                        "Nodo Cloud: tempo di risposta per batch (linee per λ)",
                        "infinite_cloud_response_vs_batch_per_lambda.png")
    plot_node_by_lambda("coord_avg_wait",
                        "Nodo Coordinator: tempo di risposta per batch (linee per λ)",
                        "infinite_coord_response_vs_batch_per_lambda.png")

    # ---------- 2) Curva Edge response time vs λ con punti per slot + QoS ----------
    agg_lambda_global = df.groupby("lambda", as_index=False).agg({
        "edge_avg_wait": "mean"
    }).sort_values("lambda")

    agg_lambda_slot = df.groupby(["slot", "lambda"], as_index=False).agg({
        "edge_avg_wait": "mean"
    }).sort_values(["lambda", "slot"])

    qos_seconds = 3.0  # QoS richiesto = 3s

    plt.figure()
    # Curva media globale (λ -> W_edge)
    if not agg_lambda_global.empty:
        plt.plot(agg_lambda_global["lambda"], agg_lambda_global["edge_avg_wait"],
                 marker="o", color="blue")
    # Punti per slot
    for slot, g in agg_lambda_slot.groupby("slot"):
        if not g.empty:
            plt.scatter(g["lambda"], g["edge_avg_wait"], label=f"Slot {slot}")
            for _, r in g.iterrows():
                plt.annotate(f"{r['lambda']:.5f}",
                             (r["lambda"], r["edge_avg_wait"]),
                             textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=8)
    # Linea QoS
    plt.axhline(y=qos_seconds, linestyle="--", color="r", linewidth=1.2,
                label=f"QoS = {qos_seconds:.0f}s")
    plt.xlabel("λ (arrivi/secondo)")
    plt.ylabel("Tempo medio di risposta Edge (s)")
    plt.title("Tempo di risposta Edge rispetto a λ (con QoS = 3s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / "infinite_edge_response_vs_lambda_with_qos.png",
                dpi=150, bbox_inches="tight")
    plt.close()
