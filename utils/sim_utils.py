# utils/sim_utils.py
import math
import random
import statistics
from libraries.rngs import selectStream, random as rng_random
from libraries import rvms
import utils.constants as cs
import sys

from utils.simulation_stats import Track

arrival_temp = cs.START  # variabile globale per il tempo di arrivo corrente
import statistics
from math import sqrt
from statistics import StatisticsError
from decimal import Decimal
from fractions import Fraction

# -------------------------------------------------------------------------------
# helper function per ricalcolare le probabilità dei pacchetti a partire da p_c
# -------------------------------------------------------------------------------

# utils/sim_utils.py
BASE_P = getattr(cs, "P_BASES", (0.20, 0.25, 0.10, 0.05))

def set_pc_and_update_probs(pc: float):
    pc = max(0.0, min(1.0, float(pc)))
    cs.P_C = pc
    cs.P_COORD = 1.0 - pc

    b1,b2,b3,b4 = BASE_P
    base_sum = max(1e-12, b1+b2+b3+b4)  # 0.60

    # condizionate (somma=1)
    p1c = b1/base_sum; p2c = b2/base_sum; p3c = b3/base_sum; p4c = b4/base_sum
    # E* (somma=1 - pc) -> queste finiscono in cs.*
    cs.P1_PROB = cs.P_COORD * p1c
    cs.P2_PROB = cs.P_COORD * p2c
    cs.P3_PROB = cs.P_COORD * p3c
    cs.P4_PROB = cs.P_COORD * p4c

    # debug
    s_eff = cs.P1_PROB + cs.P2_PROB + cs.P3_PROB + cs.P4_PROB
    print(f"[DEBUG] pc={pc:.2f} | P5={cs.P_C:.3f} | sum(E*)={s_eff:.3f} (atteso={cs.P_COORD:.3f})")


#---------------------------------------------------------------------------------

def safe_stdev(data, xbar=None):
    """Calcolo della deviazione standard campionaria senza usare statistics.stdev"""
    data = list(map(float, data))  # forza valori float
    n = len(data)
    if n < 2:
        raise StatisticsError("safe_stdev requires at least two data points")

    mean = xbar if xbar is not None else sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return sqrt(variance)

# -------------------------------
# ARRIVI - processo di Poisson non omogeneo (in secondi)
# -------------------------------
def GetLambda(current_time):
    """
    Restituisce il tasso λ (in job/secondo) in base alla fascia oraria corrente.

    Nota: i valori di λ nel file constants.py sono già espressi in secondi,
    quindi non è necessaria alcuna conversione da minuti a secondi.

    Riferimento: Tabella \ref{tab:lambda} nel documento LaTeX.
    """
    for start, end, lam in cs.LAMBDA_SLOTS:
        if start <= current_time < end:
            return lam
    return cs.LAMBDA_SLOTS[-1][2]  # default ultima fascia

def GetServiceFeedback_improved():
    return Exponential(cs.FEEDBACK_SERVICE)

def GetServiceEdgeE_im():
    """Servizio per job di classe E all'Edge (usa la costante improved)."""
    return Exponential(cs.EDGE_SERVICE_E_im)

def Exponential(mean):
    """
    Genera un tempo esponenziale con media 'mean'.

    Riferimento: Sezione 'Scelta delle distribuzioni' nel documento.
    """
    if mean <= 0:
        raise ValueError("La media deve essere positiva")
    return -mean * math.log(1.0 - rng_random())


def LogNormal(mean, sigma=0.25):
    """
    Genera un tempo lognormale con media target 'mean' e deviazione sigma.

    Riferimento: Scelta motivata per job P3/P4 nella Sezione 'Coordinator Server Edge'.
    """
    mu = math.log(mean) - 0.5 * sigma**2
    return math.exp(mu + sigma * random.gauss(0, 1))


def GetArrival(current_time, forced_lambda=None):
    """
    Genera il prossimo tempo di arrivo.

    - Se forced_lambda è specificato, usa sempre quel valore come λ.
    - Altrimenti calcola λ in base alla fascia oraria corrente.

    Riferimento: Sezione 'Modello delle specifiche'.
    """
    global arrival_temp
    lam = forced_lambda if forced_lambda is not None else GetLambda(current_time)
    arrival_temp += Exponential(1 / lam)
    return arrival_temp


def reset_arrival_temp():
    """
    Reset del tempo di arrivo.

    Utile per ripartire la simulazione (es. nuova replica).
    """
    global arrival_temp
    arrival_temp = cs.START


# -------------------------------
# TEMPI DI SERVIZIO (in secondi)
# -------------------------------
def GetServiceEdgeE():
    """Servizio per job di classe E all'Edge (Exp(0.5s))"""
    return Exponential(cs.EDGE_SERVICE_E)


def GetServiceEdgeC():
    """Servizio per job di classe C all'Edge (Exp(0.1s))"""
    return Exponential(cs.EDGE_SERVICE_C)


def GetServiceCloud():
    """Servizio al Cloud (Exp(0.8s), infinite-server)"""
    return Exponential(cs.CLOUD_SERVICE)


def GetServiceCoordP1P2():
    """Servizio Coordinator Edge per P1/P2 (Exp(0.25s))"""
    return Exponential(cs.COORD_SERVICE_P1P2)


def GetServiceCoordP3P4():
    """Servizio Coordinator Edge per P3/P4 (LogNorm(0.4s))"""
    return LogNormal(cs.COORD_SERVICE_P3P4)


# -------------------------------
# SUPPORTO EVENT-DRIVEN
# -------------------------------
def Min(*args):
    finite_values = [a for a in args if a != float("inf")]
    return min(finite_values) if finite_values else float("inf")



def remove_batch(stats, n, renumber_batches=True):
    if n <= 0: return
    for name, value in vars(stats).items():
        if isinstance(value, list):
            setattr(stats, name, value[n:])
    if hasattr(stats, "results") and isinstance(stats.results, list):
        stats.results = stats.results[n:]
        if renumber_batches:
            for i, row in enumerate(stats.results):
                row["batch"] = i


def reset_infinite(self):
        """
        Reset per la simulazione a orizzonte infinito (batch-means).
        Azzera SOLO contatori e aree del batch corrente.
        NON svuota code, NON azzera numeri in sistema, NON cambia i completion in corso.
        """

        # --- Contatori batch (arrivi/completamenti) ---
        self.job_arrived = 0

        self.index_edge = 0
        self.index_cloud = 0
        self.index_coord = 0

        # Breakdown Edge per classi (E/C) – solo contatori del batch
        self.index_edge_E = 0
        self.index_edge_C = 0

        self.count_E = 0
        self.count_C = 0
        self.count_E_P1 = 0
        self.count_E_P2 = 0
        self.count_E_P3 = 0
        self.count_E_P4 = 0

        # --- Aree accumulate nel batch ---
        self.area_edge = Track()
        self.area_cloud = Track()
        self.area_coord = Track()
        self.area_E = Track()
        self.area_C = Track()

        # code: svuotiamo per garantire indipendenza tra batch
        self.queue_edge = []
        self.queue_coord_low = []
        self.queue_coord_high = []

        # numeri di job in servizio/attesa
        self.number_edge = 0
        self.number_cloud = 0
        self.number_coord = 0

        self.number_E = 0  #
        self.number_C = 0

        self.index_edge_E = 0
        self.index_edge_C = 0

        # completamenti a ∞
        self.t.completion_edge = cs.INFINITY
        self.t.completion_cloud = cs.INFINITY
        self.t.completion_coord = cs.INFINITY

# -------------------------------
# INTERVALLI DI CONFIDENZA
# -------------------------------
import numpy as np
from scipy import stats
def calculate_confidence_interval(data, confidence=0.95):
    data = list(map(float, data))  # ← forza i float
    if len(data) < 2:
        return 0.0, 0.0  # oppure raise ValueError

    mean = statistics.mean(data)
    stdev = safe_stdev(data)
    margin = 1.96 * (stdev / (len(data) ** 0.5))  # z=1.96 per 95%
    return mean, margin


# -------------------------------
# GESTIONE DELLE STATISTICHE
# -------------------------------
def append_stats(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))

    # tempi di risposta (già presenti)
    replicationStats.edge_wait_times.append(results['edge_avg_wait'])
    replicationStats.cloud_wait_times.append(results['cloud_avg_wait'])
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])


    # tempi di attesa e risposta  per job di classe E
    replicationStats.edge_E_delay_times.append(results['edge_E_avg_delay'])
    replicationStats.edge_E_response_times.append(results['edge_E_avg_response'])

    # NEW: C
    replicationStats.edge_C_delay_times.append(results['edge_C_avg_delay'])
    replicationStats.edge_C_response_times.append(results['edge_C_avg_response'])
    replicationStats.edge_E_util_times.append(results['edge_E_utilization'])
    replicationStats.edge_C_util_times.append(results['edge_C_utilization'])


    # nuovi: code
    replicationStats.edge_delay_times.append(results['edge_avg_delay'])
    replicationStats.cloud_delay_times.append(results['cloud_avg_delay'])
    replicationStats.coord_delay_times.append(results['coord_avg_delay'])

    # L e Lq
    replicationStats.edge_L.append(results['edge_L'])
    replicationStats.edge_Lq.append(results['edge_Lq'])
    replicationStats.cloud_L.append(results['cloud_L'])
    replicationStats.cloud_Lq.append(results['cloud_Lq'])
    replicationStats.coord_L.append(results['coord_L'])
    replicationStats.coord_Lq.append(results['coord_Lq'])
    replicationStats.edge_E_L.append(results['edge_E_L'])
    replicationStats.edge_E_Lq.append(results['edge_E_Lq'])
    replicationStats.edge_C_L.append(results['edge_C_L'])
    replicationStats.edge_C_Lq.append(results['edge_C_Lq'])

    # utilizzazioni
    replicationStats.edge_utilization.append(results['edge_utilization'])
    replicationStats.coord_utilization.append(results['coord_utilization'])
    replicationStats.cloud_busy.append(results['cloud_avg_busy_servers'])

    # throughput
    replicationStats.edge_X.append(results['edge_throughput'])
    replicationStats.cloud_X.append(results['cloud_throughput'])
    replicationStats.coord_X.append(results['coord_throughput'])

    replicationStats.edge_E_wait_interval.append(getattr(stats, 'edge_E_wait_times_interval', []))
    replicationStats.edge_C_wait_interval.append(getattr(stats, 'edge_C_wait_times_interval', []))

    # transiente (safe anche qui)
    replicationStats.edge_wait_interval.append(getattr(stats, 'edge_wait_times', []))
    replicationStats.cloud_wait_interval.append(getattr(stats, 'cloud_wait_times', []))
    replicationStats.coord_wait_interval.append(getattr(stats, 'coord_wait_times', []))


def append_stats_improved(replicationStats, results, stats):
    # meta
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))
    replicationStats.feedback_wait_times.append(stats.feedback_wait_times)
    # --- Tempi di risposta (usa Edge_NuoviArrivi come 'edge') ---
    replicationStats.edge_wait_times.append(results['edge_NuoviArrivi_avg_wait'])
    replicationStats.cloud_wait_times.append(results['cloud_avg_wait'])
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])
    # NEW: aggiungi anche le serie del feedback
    replicationStats.feedback_wait_times.append(stats.feedback_wait_times)
    # --- Tempi di attesa/risposta per job di classe E (legacy dal tuo modello) ---
    replicationStats.edge_E_delay_times.append(results['edge_E_avg_delay'])
    replicationStats.edge_E_response_times.append(results['edge_E_avg_response'])

    # --- Tempi di coda medi (usa Edge_NuoviArrivi come 'edge') ---
    replicationStats.edge_delay_times.append(results['edge_NuoviArrivi_avg_delay'])
    replicationStats.cloud_delay_times.append(results['cloud_avg_delay'])
    replicationStats.coord_delay_times.append(results['coord_avg_delay'])

    # --- L e Lq (usa Edge_NuoviArrivi come 'edge') ---
    replicationStats.edge_L.append(results['edge_NuoviArrivi_L'])
    replicationStats.edge_Lq.append(results['edge_NuoviArrivi_Lq'])
    replicationStats.cloud_L.append(results['cloud_L'])
    replicationStats.cloud_Lq.append(results['cloud_Lq'])
    replicationStats.coord_L.append(results['coord_L'])
    replicationStats.coord_Lq.append(results['coord_Lq'])

    # --- Ls (numero medio in servizio) ---              # << NEW BLOCK
    if not hasattr(replicationStats, 'edge_Ls'):
        replicationStats.edge_Ls = []
    if not hasattr(replicationStats, 'feedback_Ls'):
        replicationStats.feedback_Ls = []
    replicationStats.edge_Ls.append(results.get('edge_NuoviArrivi_Ls', 0.0))
    replicationStats.feedback_Ls.append(results.get('edge_Feedback_Ls', 0.0))

    # --- Utilizzazioni / busy (usa Edge_NuoviArrivi come 'edge') ---
    replicationStats.edge_utilization.append(results['edge_NuoviArrivi_utilization'])
    replicationStats.coord_utilization.append(results['coord_utilization'])
    replicationStats.cloud_busy.append(results['cloud_avg_busy_servers'])

    # --- Throughput (usa Edge_NuoviArrivi come 'edge') ---
    replicationStats.edge_X.append(results['edge_NuoviArrivi_throughput'])
    replicationStats.cloud_X.append(results['cloud_throughput'])
    replicationStats.coord_X.append(results['coord_throughput'])



    # --- Serie transiente (già pronte nello stats runtime) ---
    replicationStats.edge_wait_interval.append(stats.edge_wait_times)
    replicationStats.cloud_wait_interval.append(stats.cloud_wait_times)
    replicationStats.coord_wait_interval.append(stats.coord_wait_times)

def append_edge_scalability_stats_improved(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))

    # Edge scalability: considera Edge_NuoviArrivi come nodo Edge
    replicationStats.edge_wait_times.append(results['edge_NuoviArrivi_avg_wait'])
    replicationStats.edge_wait_interval.append(stats.edge_wait_times)

def append_coord_scalability_stats_improved(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))

    # Focus sul Coordinator (immutato)
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])
    replicationStats.coord_wait_interval.append(stats.coord_wait_times)


def append_edge_scalability_stats(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))
    replicationStats.edge_wait_times.append(results['edge_avg_wait'])
    replicationStats.edge_wait_interval.append(stats.edge_wait_times)

def append_coord_scalability_stats(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))
    # tempi del Coordinator (focus)
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])
    replicationStats.coord_wait_interval.append(stats.coord_wait_times)
