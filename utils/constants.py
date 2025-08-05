"""
utils/constants.py
-------------------
Definizione delle costanti globali usate nella simulazione.

Riferimento: Sezione "Modello delle specifiche" e "Equazioni di traffico"
nel documento PMCSN Project (Luglio 2025).
"""

# -------------------------
# Parametri di randomizzazione
# -------------------------
SEED = 123456789
# Numero di repliche per stimare le metriche (Sezione 5.2 - batch means e replicazioni)
REPLICATIONS = 10


# -------------------------
# Orizzonte temporale
# -------------------------
# La simulazione parte a t=0 secondi
START = 0.0
# Orizzonte finito: 24 ore = 86400 secondi (Sezione 5.1 del documento)
STOP = 86400
# Stop ridotto per analisi transiente (documento: Sezione "Analisi Transiente")
STOP_ANALYSIS = 60000
# Caso infinito (non usato qui, ma previsto come compatibilità col framework)
STOP_INFINITE = float("inf")
# Valore infinito per confronti temporali
INFINITY = float("inf")

# Flag per attivare analisi transiente
TRANSIENT_ANALYSIS = 1


# -------------------------
# Routing e probabilità
# -------------------------
# p_c = probabilità che un pacchetto di classe E (P1..P4)
# venga classificato come sconosciuto (P5) e inviato al Cloud
# Riferimento: Tabella di routing e testo "Equazioni di traffico"
P_C = 0.4


# -------------------------
# Configurazione dei server
# -------------------------
# Un singolo nodo Edge (FIFO infinita, Sezione "Modello dei centri - Edge")
EDGE_SERVERS = 1
# Cloud trattato come infinite server (Sezione "Modello dei centri - Cloud")
CLOUD_SERVERS = float("inf")
# Un singolo Coordinator Server Edge (Sezione "Coordinator Server Edge")
COORD_EDGE_SERVERS = 1


# -------------------------
# Parametri di arrivo (Poisson non omogeneo)
# -------------------------
# λ stimati per fasce orarie sulla base dei dati ADR (Tabella \ref{tab:lambda})
# Ogni tupla = (inizio_fascia_sec, fine_fascia_sec, lambda [job/min])
LAMBDA_SLOTS = [
    (21600, 36000, 0.26861),   # 06:00 - 09:59
    (36000, 50400, 0.19202),  # 10:00 - 13:59
    (50400, 64800, 0.15348),    # 14:00 - 17:59
    (64800, 79200, 0.09208),    # 18:00 - 21:59
    (79200, 90000, 0.06138)     # 22:00 - 01:00
]


# -------------------------
# Tempi medi di servizio
# -------------------------
# Edge node: distribuzione esponenziale
# (Sezione "Modello dei centri - Edge")
EDGE_SERVICE_E = 0.5   # per pacchetti di classe E (P1..P4)
EDGE_SERVICE_C = 0.1   # per pacchetti di classe C (P5 update)

# Cloud server: esponenziale con media 0.8s (Sezione "Modello dei centri - Cloud")
CLOUD_SERVICE = 0.8

# Coordinator Edge:
# - P1/P2: servizio rapido e regolare → distribuzione esponenziale
# - P3/P4: servizio variabile e più lungo → distribuzione lognormale
# Riferimento: Sezione "Coordinator Server Edge"
COORD_SERVICE_P1P2 = 0.25
COORD_SERVICE_P3P4 = 0.40  # media lognormale


# -------------------------
# Parametri statistici
# -------------------------
# Livello di confidenza (ALPHA = 0.05 → intervallo al 95%)
# Riferimento: metodologia batch means e intervalli di confidenza
ALPHA = 0.05

# -------------------------
# Parametri per simulazione infinita (non richiesti qui ma compatibili)
# -------------------------
K = 128    # numero di batch
B = 4080   # dimensione di ciascun batch
