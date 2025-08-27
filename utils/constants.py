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





FEEDBACK_SERVICE = 0.1  # 0.1 classe C
# -------------------------
EDGE_SERVICE_E_im = 0.42  # 0.5 * 2
EDGE_M = 1  # M/M/1 per il nodo Edge

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
w1, w2, w3, w4 = 0.20, 0.25, 0.10, 0.05
W = w1 + w2 + w3 + w4 # = 0.60
P1 = w1 / W
P2 = w2 / W
P3 = w3 / W
P4 = w4 / W
assert abs(P1+P2+P3+P4 - 1) < 1e-12
# Se ti servono le probabilità *incondizionate* complessive:
P1_PROB = P_COORD * P1
P2_PROB = P_COORD * P2
P3_PROB = P_COORD * P3
P4_PROB = P_COORD * P4
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
COORD_SERVICE_P1P2 = 0.4   # media esponenziale
COORD_SERVICE_P3P4 = 0.6   # media lognormale


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
