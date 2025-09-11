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
from typing import List, Tuple, Dict, Any, Union

import matplotlib.pyplot as plt
import pandas as pd
from utils import constants as cs
from utils.sim_utils import calculate_confidence_interval
from datetime import datetime
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os

# Directory di output
file_path = "output/"

# Intestazione CSV per i risultati della simulazione
header = [
    "seed",
    # Edge (centro)
    "edge_avg_wait", "edge_avg_delay", "edge_avg_service_time", "edge_utilization",
    "edge_avg_number_node", "edge_avg_number_queue",
    # Cloud (centro)
    "cloud_avg_wait", "cloud_avg_delay", "cloud_avg_service_time", "cloud_utilization",
    "cloud_avg_number_node", "cloud_avg_number_queue",
    # Coordinator (extra)
    "coord_avg_wait", "coord_avg_delay", "coord_avg_service_time", "coord_utilization",
    "coord_avg_number_node", "coord_avg_number_queue",
    # Classe E (Edge)
    "count_E", "E_avg_wait", "E_avg_delay", "E_avg_service_time", "E_utilization",
    "E_avg_number_edge", "E_avg_number_queue_edge",
    # Classe C (Edge)
    "count_C", "C_avg_wait", "C_avg_delay", "C_avg_service_time", "C_utilization",
    "C_avg_number_edge", "C_avg_number_queue_edge",
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
# Mappa header -> chiave nel dict results della tua simulazione
CSV_KEYMAP = {
    "seed": "seed",

    "edge_avg_wait": "edge_avg_wait",
    "edge_avg_delay": "edge_avg_delay",
    "edge_avg_service_time": "edge_service_time_mean",
    "edge_utilization": "edge_utilization",
    "edge_avg_number_node": "edge_L",
    "edge_avg_number_queue": "edge_Lq",

    "cloud_avg_wait": "cloud_avg_wait",
    "cloud_avg_delay": "cloud_avg_delay",
    "cloud_avg_service_time": "cloud_service_time_mean",
    "cloud_utilization": "cloud_utilization",          # se assente, fallback su busy servers
    "cloud_avg_number_node": "cloud_L",
    "cloud_avg_number_queue": "cloud_Lq",

    "coord_avg_wait": "coord_avg_wait",
    "coord_avg_delay": "coord_avg_delay",
    "coord_avg_service_time": "coord_service_time_mean",
    "coord_utilization": "coord_utilization",
    "coord_avg_number_node": "coord_L",
    "coord_avg_number_queue": "coord_Lq",

    "count_E": "count_E",
    "E_avg_wait": "edge_E_avg_response",
    "E_avg_delay": "edge_E_avg_delay",
    "E_avg_service_time": "edge_E_service_time_mean",  # potrebbe non esserci: lascio vuoto
    "E_utilization": "E_utilization",
    "E_avg_number_edge": "edge_E_L",
    "E_avg_number_queue_edge": "edge_E_Lq",

    "count_C": "count_C",
    "C_avg_wait": "edge_C_avg_response",
    "C_avg_delay": "edge_C_avg_delay",
    "C_avg_service_time": "edge_C_service_time_mean",  # potrebbe non esserci: lascio vuoto
    "C_utilization": "C_utilization",
    "C_avg_number_edge": "edge_C_L",
    "C_avg_number_queue_edge": "edge_C_Lq",
}

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

def clear_file(file_path: str, header: Dict[str, Any] | None = None):
    """Svuota il file. Se passi una 'header' (dict), crea il file con intestazione."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", newline="") as f:
        if header:
            w = csv.DictWriter(f, fieldnames=list(header.keys()))
            w.writeheader()


def _is_pathlike(x) -> bool:
    try:
        return isinstance(x, (str, bytes, os.PathLike))
    except TypeError:
        return False

def write_file(a: Union[str, Dict[str, Any], List[Dict[str, Any]]],
               b: Union[str, Dict[str, Any], List[Dict[str, Any]]]):
    """
    Accetta sia write_file(file_path, rows) sia write_file(rows, file_path).
    'rows' può essere un dict (singola riga) o una lista di dict.
    Scrive l'intestazione se il file è nuovo/vuoto.
    """
    # --- disambiguazione degli argomenti ---
    if _is_pathlike(a) and not _is_pathlike(b):
        file_path, rows = a, b
    elif _is_pathlike(b) and not _is_pathlike(a):
        file_path, rows = b, a
    else:
        raise TypeError("write_file vuole (file_path, rows) oppure (rows, file_path).")

    # rows può essere dict o list[dict]
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return

    # campi/colonne
    fieldnames = list(rows[0].keys())

    # crea cartella se serve
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    # header se file non esiste o è vuoto
    must_write_header = (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0

    with open(file_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if must_write_header:
            w.writeheader()
        w.writerows(rows)


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



# utils/simulation_output.py
def plot_analysis(series, seeds, name, sim_type="finite_fixed_lambda"):
    """
    Traccia 1 curva per replica.
    - series: [[(t,y), ...], [(t,y), ...], ...]  (oppure una singola [(t,y), ...])
    - seeds:  mantenuto solo per compatibilità; NON usato in legenda
    - name:   nome file immagine senza estensione
    - sim_type: per scegliere la sotto-cartella di output
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    import utils.constants as cs  # per fissare l'asse X a [0, STOP]

    def _is_xy_list(obj):
        return bool(obj) and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2 and \
               all(isinstance(p, (list, tuple)) and len(p) == 2 for p in obj)

    def _is_list_of_xy_lists(obj):
        return bool(obj) and isinstance(obj[0], (list, tuple)) and _is_xy_list(obj[0])

    label = _label_for_sim(sim_type)

    out_dir = os.path.join(file_path, "plot", label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.png")

    plt.figure(figsize=(10, 6))

    if _is_list_of_xy_lists(series):
        # Serie = [ replica1[(t,y),...], replica2[(t,y),...], ... ]
        for run in series:
            if not run:
                continue
            xs = [pt[0] for pt in run]
            ys = [pt[1] for pt in run]
            plt.plot(xs, ys)  # nessuna legenda (no seed)
    elif _is_xy_list(series):
        xs = [pt[0] for pt in series]
        ys = [pt[1] for pt in series]
        plt.plot(xs, ys)
    else:
        # Fallback: niente (non abbiamo (t,y))
        plt.close()
        return out_path

    # Assi e stile
    try:
        plt.xlim(0, float(getattr(cs, "STOP", 86400.0)))  # 24h = 86400s
    except Exception:
        pass
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Average response time (s)")
    plt.title(f"{name} — {label.replace('_', ' ')}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path



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

    # --- Batch-means cumulative plots (Edge/Cloud/Coord) ---

    # Batch numerico e pulizia robusta
    df["batch"] = pd.to_numeric(df["batch"], errors="coerce")
    df = df.dropna(subset=["batch"]).copy()
    df["batch"] = df["batch"].astype(int)

    def _running_mean(y):
        s = pd.to_numeric(y, errors='coerce').fillna(0)
        return s.expanding().mean()

    def plot_node_by_lambda(ycol, title, fname):
        plt.figure()
        any_line = False
        groups = list(df.sort_values(["slot", "lambda", "batch"]).groupby(["slot", "lambda"]))
        for (slot_val, lam_val), g in groups:
            if g.empty:
                continue
            rm = _running_mean(g[ycol])
            lab = f"λ={lam_val:.5f}" if df["slot"].nunique() == 1 else f"Slot {slot_val} — λ={lam_val:.5f}"
            plt.plot(g["batch"], rm, label=lab)
            any_line = True
        plt.xlabel("Batch")
        plt.ylabel("Wait time (media cumulata) [s]")
        plt.title(title + " — media cumulata")
        if any_line and len(groups) > 1:
            plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

    plot_node_by_lambda(
        "edge_avg_wait",
        "Nodo Edge: wait time medio per batch",
        "infinite_edge_wait_vs_batch_per_lambda.png"
    )
    plot_node_by_lambda(
        "cloud_avg_wait",
        "Nodo Cloud: wait time medio per batch",
        "infinite_cloud_wait_vs_batch_per_lambda.png"
    )
    plot_node_by_lambda(
        "coord_avg_wait",
        "Nodo Coordinator: wait time medio per batch",
        "infinite_coord_wait_vs_batch_per_lambda.png"
    )

    # ---------- 2) Curva Edge wait time vs λ con punti per slot + QoS ----------
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
        plt.plot(
            agg_lambda_global["lambda"],
            agg_lambda_global["edge_avg_wait"],
            marker="o"
        )
    # Punti per slot
    for slot, g in agg_lambda_slot.groupby("slot"):
        if not g.empty:
            plt.scatter(g["lambda"], g["edge_avg_wait"], label=f"Slot {slot}")
            for _, r in g.iterrows():
                plt.annotate(
                    f"{r['lambda']:.5f}",
                    (r["lambda"], r["edge_avg_wait"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8
                )
    # Linea QoS
    plt.axhline(y=qos_seconds, linestyle="--", linewidth=1.2, label=f"QoS = {qos_seconds:.0f}s")
    plt.xlabel("λ (arrivi/secondo)")
    plt.ylabel("Tempo medio di attesa Edge (s)")
    plt.title("Wait time Edge rispetto a λ (con QoS = 3s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / "infinite_edge_wait_vs_lambda_with_qos.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_transient_with_seeds(series, seeds, name, sim_type="finite_fixed_lambda"):
    """
    Disegna una curva per replica e mette 'Seed NNN...' in legenda.
    - series: [[(t,y),...], [(t,y),...], ...]  (una lista per ogni replica)
    - seeds:  lista dei seed (stessa lunghezza di series)
    - name:   nome del file PNG senza estensione
    - sim_type: sottocartella in output/plot/transient_analysis/
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import utils.constants as cs

    os.makedirs(os.path.join("output", "plot", "transient_analysis", sim_type), exist_ok=True)
    plt.figure(figsize=(10, 6))

    for i, run in enumerate(series):
        if not run:
            continue
        xs = [p[0] for p in run]
        ys = [p[1] for p in run]
        label = f"Seed {seeds[i]}" if i < len(seeds) else f"replica {i}"
        plt.plot(xs, ys, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Average wait (s)")
    plt.xlim(0, cs.STOP_ANALYSIS)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8, frameon=False)

    out_dir = os.path.join("output", "plot", "transient_analysis", sim_type)
    out_path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_edge_response_vs_pc(csv_path: str,
                             out_path: str | None = None,
                             response_col: str | None = None,
                             qos_threshold: float = 3.0) -> str:
    """
    Grafica il TEMPO DI RISPOSTA dei pacchetti E nel nodo Edge vs p_c,
    tracciando UNA linea per ciascun λ. Usa direttamente la colonna 'edge_avg_wait_E'
    già presente nel CSV (nessuna composizione).

    Parametri:
      - csv_path      : percorso del CSV (es. "output_improved/merged_scalability_statistics.csv").
      - out_path      : (opz.) percorso PNG di output; se None -> <dir csv>/plot/edge_E_response_vs_pc.png
      - response_col  : (opz.) forza un nome colonna; default autodetect 'edge_avg_wait_E'
                        (fallback: 'edge_E_avg_response' se presente).
      - qos_threshold : (opz.) soglia QoS in secondi per la linea orizzontale (default 3.0).

    Ritorna: percorso del file PNG generato.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Nessun dato in {csv_path}")

    # Normalizza nomi colonna
    df.columns = [str(c).strip() for c in df.columns]

    # Individua la colonna del tempo di risposta dei pacchetti E @ Edge
    resp = response_col
    if resp is None:
        if "edge_NuoviArrivi_avg_wait" in df.columns:
            resp = "edge_NuoviArrivi_avg_wait"
        elif "edge_avg_wait_E" in df.columns:
            resp = "edge_avg_wait_E"
        elif "edge_E_avg_response" in df.columns:
            resp = "edge_E_avg_response"
        else:
            raise ValueError('Colonna non trovata: attesa \'edge_avg_wait_E\' , \'edge_NuoviArrivi_avg_wait\' (o \'edge_E_avg_response\').')

    # Colonne richieste
    required = {"pc", "lambda", resp}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne nel CSV: {missing}")

    # Cast numerico e drop NA
    for col in ["pc", "lambda", resp]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["pc", "lambda", resp])
    if df.empty:
        raise ValueError("Dati insufficienti dopo la pulizia per 'pc', 'lambda' e risposta E.")

    # Media per (lambda, pc) se più righe/repliche
    agg = (
        df.groupby(["lambda", "pc"], as_index=False)[resp]
          .mean()
          .rename(columns={resp: "response_E"})
          .sort_values(["lambda", "pc"])
    )

    # Output path
    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        out_dir = os.path.join(base_dir, "plot")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "edge_E_response_vs_pc.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Plot: una linea per ciascun λ
    plt.figure()
    for lam, g in agg.groupby("lambda"):
        plt.plot(g["pc"], g["response_E"], label=f"λ={lam}")

    # Linea QoS a 3s (parametrizzabile)
    if qos_threshold is not None:
        plt.axhline(y=qos_threshold, linestyle="--", label=f"QoS = {qos_threshold}s")

    plt.xlabel("p_c")
    plt.ylabel("Tempo di risposta E @ Edge [s]")
    plt.title("Edge (pacchetti E): tempo di risposta vs p_c — una linea per λ")
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[plot_edge_response_vs_pc] Grafico salvato in: {out_path}")
    return out_path
