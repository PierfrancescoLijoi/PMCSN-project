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
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import matplotlib.pyplot as plt
import pandas as pd
from utils import constants as cs
from utils.sim_utils import calculate_confidence_interval, calculate_autocorrelation
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



def plot_analysis(series, name, sim_type="finite_fixed_lambda"):
    """
    Traccia 1 curva per replica e ripete le fasce λ ogni giorno.
    - series: [[(t,y),...], ...] oppure [(t,y),...]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    import utils.constants as cs  # per STOP e LAMBDA_SLOTS

    def _is_xy_list(obj):
        return bool(obj) and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2 and \
               all(isinstance(p, (list, tuple)) and len(p) == 2 for p in obj)

    def _is_list_of_xy_lists(obj):
        return bool(obj) and isinstance(obj[0], (list, tuple)) and _is_xy_list(obj[0])

    def _max_x(series):
        if _is_list_of_xy_lists(series):
            return max((pt[0] for run in series for pt in run), default=float(getattr(cs, "STOP", 86400.0)))
        elif _is_xy_list(series):
            return max((pt[0] for pt in series), default=float(getattr(cs, "STOP", 86400.0)))
        return float(getattr(cs, "STOP", 86400.0))

    label = _label_for_sim(sim_type)

    out_dir = os.path.join(file_path, "plot", label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.png")

    plt.figure(figsize=(10, 6))

    # --- disegno serie ---
    if _is_list_of_xy_lists(series):
        for run in series:
            if not run:
                continue
            xs = [pt[0] for pt in run]
            ys = [pt[1] for pt in run]
            plt.plot(xs, ys)
    elif _is_xy_list(series):
        xs = [pt[0] for pt in series]
        ys = [pt[1] for pt in series]
        plt.plot(xs, ys)
    else:
        plt.close()
        return out_path

    # --- assi e stile ---
    x_right = _max_x(series)
    plt.xlim(0, x_right)
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Average response time (s)")
    plt.title(f"{name} — {label.replace('_', ' ')}")
    plt.grid(True, alpha=0.3)

    # Forza asse Y: tick da 0.1 a 0.7
    yticks = [i / 10 for i in range(1, 8)]  # 0.1,0.2,...,0.7
    plt.yticks(yticks)
    plt.ylim(0.0, 0.7)

    # --- Delimitatori fasce di lambda ripetuti ogni giorno ---
    try:
        slots = list(getattr(cs, "LAMBDA_SLOTS", []))  # [(start,end,lam), ...] su 0..86400
        DAY = float(getattr(cs, "DAY_SECONDS", 86400.0))  # 1 giorno in secondi

        if slots:
            # ripeti i marker di slot e le etichette su tutti i giorni coperti da x_right
            n_days = int(x_right // DAY) + 1

            for k in range(n_days):
                offset = k * DAY
                # linee di fine fascia e label λ
                for (s, e, lam) in slots:
                    xe = float(e) + offset
                    if 0.0 <= xe <= x_right:
                        plt.axvline(x=xe, linestyle=":", alpha=0.35)
                    # testo al centro fascia
                    cx = (float(s) + float(e)) / 2.0 + offset
                    if 0.0 <= cx <= x_right:
                        plt.text(cx, 0.68, f"λ={lam:g}", ha="center", va="top",
                                 fontsize=8, alpha=0.6)

            # linee tratteggiate rosse ai confini di giorno: 86400, 172800, ...
            d = DAY
            while d <= x_right + 1e-9:
                plt.axvline(x=d, linestyle="--", color="red", linewidth=1.2, alpha=0.7)
                d += DAY

    except Exception:
        pass

    # --- linea tratteggiata rossa a y=0.6 ---
    plt.axhline(y=0.6, linestyle="--", color="red", linewidth=1.2, alpha=0.8)

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


def plot_infinite_analysis():
    """
    Grafici orizzonte infinito:
    - usa SEMPRE output/infinite_statistics.csv
    - mostra SOLO la media cumulata per batch
    - include anche Edge_E e Edge_C (tempo di risposta)
    - nomi file senza '_like_ref'
    """
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    # === CSV fisso corretto ===
    csv_path = Path("output") / "infinite_statistics.csv"
    if not csv_path.exists():
        print(f"[plot_infinite_analysis] File non trovato: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot_infinite_analysis] CSV vuoto: {csv_path}")
        return

    if "batch" not in df.columns:
        df = df.copy()
        df["batch"] = range(len(df))

    # Media per batch, ordinata
    per_batch = (
        df.groupby("batch", as_index=False)
          .mean(numeric_only=True)
          .sort_values("batch")
          .reset_index(drop=True)
    )

    out_dir = csv_path.parent
    plot_dir = out_dir / "plot" / "orizzonte infinito"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _cum(y):
        s = pd.Series(y).reset_index(drop=True)
        return s.expanding().mean()

    def _ensure_col_from_df(ycol: str):
        """Se ycol non è in per_batch ma esiste in df, porta la media per batch in per_batch[ycol]."""
        nonlocal per_batch
        if ycol in per_batch.columns:
            return True
        if ycol not in df.columns:
            return False
        tmp = df.groupby("batch", as_index=False)[ycol].mean().rename(columns={ycol: f"__tmp_{ycol}"})
        per_batch = per_batch.merge(tmp, on="batch", how="left")
        per_batch[ycol] = per_batch[f"__tmp_{ycol}"]
        per_batch.drop(columns=[f"__tmp_{ycol}"], inplace=True)
        return True

    def _plot_cum(ycol: str, title: str, fname: str, ylabel: str = "valore medio (media cumulata)"):
        # porta la colonna in per_batch se serve
        if ycol not in per_batch.columns:
            if not _ensure_col_from_df(ycol):
                print(f"[plot_infinite_analysis] Colonna assente: {ycol}")
                return

        # media cumulata
        y_mean = _cum(per_batch[ycol])  # len = N

        # --- PICCO VISIBILE DA ZERO ---
        # Prependiamo un punto iniziale (x=0, y=0) e poi i punti (1..N, media cumulata)
        x = [0] + list(range(1, len(y_mean) + 1))
        y = [0.0] + y_mean.tolist()

        # plot SOLO cumulata
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, linewidth=1.6)

        # stile come da richiesta
        plt.grid(True, alpha=0.3)
        plt.xlabel("Batch")
        plt.ylabel(ylabel)
        plt.title(title)

        # spazio a sinistra + clamp a 0
        plt.xlim(left=-1)  # un po' di spazio per far vedere il salto
        plt.ylim(bottom=0)

        plt.legend()
        plt.tight_layout()
        out_path = plot_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot_infinite_analysis] Grafico salvato: {out_path}")

    # ---------- grafici principali (media cumulata) ----------
    _plot_cum("edge_avg_wait",  "Edge node",        "infinite_edge_wait.png",  "Tempo di attesa [s]")
    _plot_cum("cloud_avg_wait", "Cloud node ",       "infinite_cloud_wait.png", "Tempo di attesa [s]")
    _plot_cum("coord_avg_wait", "Coordinator node", "infinite_coord_wait.png", "Tempo di attesa [s]")

    # ---------- Edge_E / Edge_C: tempo di risposta ----------
    # Preferiamo *_avg_response; se non c'è, costruiamo response = avg_delay + edge_service_time_mean
    def _ensure_response(col_resp: str, col_delay: str, svc_candidates: list[str]):
        # già presente?
        if col_resp in per_batch.columns or col_resp in df.columns:
            _ensure_col_from_df(col_resp)
            return col_resp
        # ricostruzione da delay + service
        if _ensure_col_from_df(col_delay):
            svc_col = None
            for c in svc_candidates:
                if _ensure_col_from_df(c):
                    svc_col = c
                    break
            if svc_col:
                per_batch[col_resp] = per_batch[col_delay] + per_batch[svc_col]
                return col_resp
        return None

    e_resp = _ensure_response(
        col_resp="edge_E_avg_response",
        col_delay="edge_E_avg_delay",
        svc_candidates=["edge_E_service_time_mean", "edge_service_time_mean"]
    )
    c_resp = _ensure_response(
        col_resp="edge_C_avg_response",
        col_delay="edge_C_avg_delay",
        svc_candidates=["edge_C_service_time_mean", "edge_service_time_mean"]
    )

    if e_resp:
        _plot_cum(e_resp, "Edge_E", "infinite_edge_E_wait.png", "Tempo di risposta [s]")
    else:
        print("[plot_infinite_analysis] impossibile plottare Edge_E (manca response o delay+service).")

    if c_resp:
        _plot_cum(c_resp, "Edge_C", "infinite_edge_C_wait.png", "Tempo di risposta [s]")
    else:
        print("[plot_infinite_analysis] impossibile plottare Edge_C (manca response o delay+service).")

    print(f"[plot_infinite_analysis] CSV usato: {csv_path.resolve()}")
    print(f"[plot_infinite_analysis] Output dir: {plot_dir.resolve()}")



def print_autocorrelation_from_csv(
    csv_path: str,
    columns: list = None,
    max_lag: int = 50,
    header: str = "\n[ACF] Autocorrelazione su medie di batch"
) -> None:
    """
    Legge un CSV (es. output/infinite_statistics.csv) e stampa ACF per le colonne richieste.
    Utile per verificare che le medie di batch siano ~indipendenti (rho(1) ~ 0).
    """
    df = pd.read_csv(csv_path)
    if columns is None:
        # default sensato per l'orizzonte infinito
        candidates = [
            "edge_avg_wait", "edge_E_avg_response", "edge_C_avg_response",
            "cloud_avg_wait", "coord_avg_wait"
        ]
        columns = [c for c in candidates if c in df.columns]

    print(header)
    print(f"  file: {csv_path}")
    for col in columns:
        s = df[col].dropna().to_numpy()
        if len(s) <= max_lag:
            print(f"  - {col}: serie troppo corta (n={len(s)}) per K_LAG={max_lag}")
            continue
        mean, stdev, ac = calculate_autocorrelation(s, K_LAG=max_lag)
        rho1 = ac[0]
        rho5 = ac[4] if len(ac) > 4 else float('nan')
        print(f"  - {col}: n={len(s)}, mean={mean:.4f}, sd={stdev:.4f}, rho(1)={rho1:.3f}, rho(5)={rho5:.3f}")




# --------------------- plot analisi transiente ----------------------------------------
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


