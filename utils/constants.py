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
REPLICATIONS = 100


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
P_COORD = 1 - P_C  # 60% → Coordinator

# Probabilità condizionate (dentro al 60% Coordinator)
P1_PROB = 0.20 / P_COORD
P2_PROB = 0.25 / P_COORD
P3_PROB = 0.10 / P_COORD
P4_PROB = 0.05 / P_COORD

# -------------------------
# Configurazione dei server
# -------------------------
# Un singolo nodo Edge (FIFO infinita, Sezione "Modello dei centri - Edge")
EDGE_SERVERS = 1
# Cloud trattato come infinite server (Sezione "Modello dei centri - Cloud")
CLOUD_SERVERS = float("inf")
# Un singolo Coordinator Server Edge (Sezione "Coordinator Server Edge")
COORD_EDGE_SERVERS = 1
# Numero massimo server scalabili
EDGE_SERVERS_MAX = 6 # Numero massimo di server Edge (per scalabilità)
COORD_SERVERS_MAX = 6 # Numero massimo di server Coordinator (per scalabilità)

# Valori soglia per scalabilità
UTILIZATION_UPPER = 0.8  # aggiunta server sopra 80% utilizzo
UTILIZATION_LOWER = 0.3  # rimozione server sotto 30% utilizzo

# -------------------------
# Parametri di arrivo (Poisson non omogeneo)
# -------------------------
# λ stimati per fasce orarie sulla base dei dati ADR (Tabella \ref{tab:lambda})
# Ogni tupla = (inizio_fascia_sec, fine_fascia_sec, lambda [job/min])
LAMBDA_SLOTS = [
    (21600, 36000, 1.880303),   # 06:00 - 09:59 con k=7.0
    (36000, 50400, 1.343074),  # 10:00 - 13:59
    (50400, 64800, 1.074459),    # 14:00 - 17:59
    (64800, 79200, 0.644675),    # 18:00 - 21:59
    (79200, 90000, 0.429783)     # 22:00 - 02:00
]
SLOT_DURATION = 14400  # 4 ore di simulazione per ogni λ


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
COORD_SERVICE_P1P2 = 0.4#con questi valori scala   #0.25 NON SCALA # media esponenziale
COORD_SERVICE_P3P4 = 0.6# con questi valori scala  #0.40  NON SCALA # media lognormale


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
