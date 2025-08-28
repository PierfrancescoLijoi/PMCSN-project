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
    "edge_NuoviArrivi_Ls",
    "edge_NuoviArrivi_utilization","edge_NuoviArrivi_throughput",
    "edge_NuoviArrivi_service_time_mean","Edge_NuoviArrivi_E_Ts",

    # Edge_Feedback (post-Cloud)
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq",
    "edge_Feedback_Ls",
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
    "edge_NuoviArrivi_Ls",
    "edge_NuoviArrivi_utilization","edge_NuoviArrivi_throughput",
    "edge_NuoviArrivi_service_time_mean","Edge_NuoviArrivi_E_Ts",

    # Edge_Feedback
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq","edge_Feedback_Ls",
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
    "edge_NuoviArrivi_L","edge_NuoviArrivi_Lq","edge_NuoviArrivi_Ls",  # << NEW
    "edge_NuoviArrivi_service_time_mean",
    "edge_NuoviArrivi_utilization",
    "edge_NuoviArrivi_throughput",

    # Edge_Feedback
    "edge_Feedback_avg_wait","edge_Feedback_avg_delay",
    "edge_Feedback_L","edge_Feedback_Lq","edge_Feedback_Ls",     # << NEW
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

    _print_ci_improved("Edge_NuoviArrivi - Avg number in service (Ls)", getattr(stats, 'edge_Ls', []))
    _print_ci_improved("Edge_Feedback - Avg number in service (Ls)", getattr(stats, 'feedback_Ls', []))

    _print_ci_improved("Edge_NuoviArrivi - Avg number in service (Ls)", getattr(stats, 'edge_Ls', []))   # << NEW
    _print_ci_improved("Edge_Feedback - Avg number in service (Ls)", getattr(stats, 'feedback_Ls', []))  # << NEW


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



def plot_edge_response_vs_pc(csv_path: str,
                             out_path: str | None = None,
                             response_col: str | None = None) -> str:
    """
    Crea un grafico del tempo di risposta dell'Edge (Nuovi Arrivi) al variare di p_c,
    disegnando UNA linea per ciascun λ presente nel CSV.

    Parametri:
      - csv_path    : percorso del CSV (es. "output_improved/merged_scalability_statistics.csv").
      - out_path    : (opzionale) percorso file immagine di output (.png). Se None, salva in
                      <dir del csv>/plot/edge_response_vs_pc.png
      - response_col: (opzionale) nome colonna del tempo di risposta da usare.
                      Se None, prova in ordine:
                        "edge_NuoviArrivi_avg_wait" (improved) oppure "edge_avg_wait" (standard).

    Ritorna:
      - percorso del file PNG generato.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Nessun dato in {csv_path}")

    # Normalizza i nomi colonna (spazi ecc.)
    df.columns = [str(c).strip() for c in df.columns]

    # Determina la colonna risposta da usare
    resp = response_col
    if resp is None:
        if "edge_NuoviArrivi_avg_wait" in df.columns:
            resp = "edge_NuoviArrivi_avg_wait"
        elif "edge_avg_wait" in df.columns:
            resp = "edge_avg_wait"
        else:
            raise ValueError(
                "Colonna tempo di risposta non trovata. Attese: "
                "'edge_NuoviArrivi_avg_wait' o 'edge_avg_wait'."
            )

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
        raise ValueError("Dati insufficienti dopo la pulizia per 'pc', 'lambda' e risposta.")

    # Aggrega per (lambda, pc) nel caso di più repliche/righe per combinazione
    agg = (
        df.groupby(["lambda", "pc"], as_index=False)[resp]
          .mean()
          .rename(columns={resp: "response"})
          .sort_values(["lambda", "pc"])
    )

    # Prepara cartella output se non fornita
    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        out_dir = os.path.join(base_dir, "plot")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "edge_response_vs_pc.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Plot: una linea per ciascun λ
    plt.figure()
    for lam, g in agg.groupby("lambda"):
        plt.plot(g["pc"], g["response"], label=f"λ={lam}")

    plt.xlabel("p_c")
    plt.ylabel("Tempo di risposta Edge (W)")
    plt.title("Edge (Nuovi Arrivi): tempo di risposta vs p_c — una linea per λ")
    plt.legend(title="λ")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[plot_edge_response_vs_pc] Grafico salvato in: {out_path}")
    return out_path
