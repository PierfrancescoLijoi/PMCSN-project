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



def remove_batch(stats, n):
    if n < 0:
        raise ValueError()
    for attr in dir(stats):
        value = getattr(stats, attr)
        if isinstance(value, list):
            setattr(stats, attr, value[n:])

def reset_infinite(self):
    """
    Reset delle statistiche per la simulazione a orizzonte infinito (batch-means).
    Non resetta il tempo globale, ma azzera contatori e aree accumulate nel batch.
    """
    # contatori arrivi/completamenti
    self.job_arrived = 0

    self.index_edge = 0
    self.index_cloud = 0
    self.index_coord = 0

    self.count_E = 0
    self.count_C = 0

    self.count_E_P1 = 0
    self.count_E_P2 = 0
    self.count_E_P3 = 0
    self.count_E_P4 = 0

    self.index_E = 0
    self.index_C = 0

    # aree accumulate (batch corrente)
    self.area_edge = Track()
    self.area_cloud = Track()
    self.area_coord = Track()
    self.area_E = Track()
    self.area_C = Track()

    # code: svuotiamo per garantire indipendenza tra batch
    self.queue_edge = []
    self.queue_coord_low = []
    self.queue_coord_high = []

    # numero di job in servizio/attesa
    self.number_edge = 0
    self.number_cloud = 0
    self.number_coord = 0

    # tempi di completamento (nessun job in corso)
    self.t.completion_edge = cs.INFINITY
    self.t.completion_cloud = cs.INFINITY
    self.t.completion_coord = cs.INFINITY

# -------------------------------
# INTERVALLI DI CONFIDENZA
# -------------------------------
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
    replicationStats.lambdas.append(results.get('lambda'))  # nuovo
    replicationStats.slots.append(results.get('slot'))      # nuovo

    replicationStats.edge_wait_times.append(results['edge_avg_wait'])
    replicationStats.cloud_wait_times.append(results['cloud_avg_wait'])
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])

    replicationStats.edge_wait_interval.append(stats.edge_wait_times)
    replicationStats.cloud_wait_interval.append(stats.cloud_wait_times)
    replicationStats.coord_wait_interval.append(stats.coord_wait_times)

def append_edge_scalability_stats(replicationStats, results, stats):
    replicationStats.seeds.append(results['seed'])
    replicationStats.lambdas.append(results.get('lambda'))
    replicationStats.slots.append(results.get('slot'))
    replicationStats.edge_wait_times.append(results['edge_avg_wait'])
    replicationStats.edge_wait_interval.append(stats.edge_wait_times)