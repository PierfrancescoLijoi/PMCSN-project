# utils/sim_utils.py
import math
import random
import statistics
from libraries.rngs import selectStream, random as rng_random
from libraries import rvms
import utils.constants as cs
import sys
arrival_temp = cs.START  # variabile globale per il tempo di arrivo corrente


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

def GetArrival(current_time):
    """
    Genera il prossimo tempo di arrivo usando il λ(t) corrente (già in secondi).

    Riferimento: Sezione 'Modello delle specifiche'.
    """
    global arrival_temp
    lam = GetLambda(current_time)  # λ già in job/secondo
    arrival_temp += Exponential(1 / lam)
    #print(f"[λ={lam:.6f}] Arrivo generato a {arrival_temp:.7f}")
    #sys.stdout.flush()

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
    """Trova il minimo ignorando gli infiniti (usato per decidere il prossimo evento)."""
    return min(a for a in args if a != float("inf"))


# -------------------------------
# INTERVALLI DI CONFIDENZA
# -------------------------------
def calculate_confidence_interval(data):
    """
    Calcola il margine di errore (±) per l'intervallo di confidenza al 95%.

    Riferimento: Sezione 'Modello computazionale'.
    """
    n = len(data)
    if n <= 1:
        return statistics.mean(data), 0.0

    mean_val = statistics.mean(data)
    stdev = statistics.stdev(data)
    t_star = rvms.idfStudent(n - 1, 1 - cs.ALPHA / 2)
    margin_of_error = t_star * stdev / math.sqrt(n)
    return mean_val, margin_of_error


# -------------------------------
# GESTIONE DELLE STATISTICHE
# -------------------------------
def append_stats(replicationStats, results, stats):
    """
    Aggiunge i risultati di una replica agli array cumulativi.

    Riferimento: gestione dei risultati replicati (analisi statistica).
    """
    replicationStats.seeds.append(results['seed'])
    replicationStats.edge_wait_times.append(results['edge_avg_wait'])
    replicationStats.cloud_wait_times.append(results['cloud_avg_wait'])
    replicationStats.coord_wait_times.append(results['coord_avg_wait'])
    # Aggiunta per analisi transiente
    replicationStats.edge_wait_interval.append(stats.edge_wait_times)
    replicationStats.cloud_wait_interval.append(stats.cloud_wait_times)
    replicationStats.coord_wait_interval.append(stats.coord_wait_times)