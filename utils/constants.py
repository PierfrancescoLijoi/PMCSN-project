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
EDGE_SERVICE_E_im = 0.42
EDGE_M = 1  # M/M/1 per il nodo Edge

# -------------------------
# Orizzonte temporale
# -------------------------
# La simulazione parte a t=0 secondi
START = 0.0
# Orizzonte finito: 24 ore = 86400 secondi (Sezione 5.1 del documento) ,48 ore = 172800
STOP = 172800
# Stop ridotto per analisi transiente (documento: Sezione "Analisi Transiente")
STOP_ANALYSIS = 300000
# Caso infinito (non usato qui, ma previsto come compatibilità col framework)
STOP_INFINITE = float("inf")
# Valore infinito per confronti temporali
INFINITY = float("inf")


#Analisi transiente
TRANSIENT_REPLICATIONS = 10

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
EDGE_SERVERS = 2
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


EDGE_SERVERS_INIT = EDGE_SERVERS
COORD_EDGE_SERVERS_INIT = COORD_EDGE_SERVERS

# -------------------------
# Parametri di arrivo (Poisson non omogeneo)
# -------------------------
# λ stimati per fasce orarie sulla base dei dati ADR (Tabella \ref{tab:lambda})
# Ogni tupla = (inizio_fascia_sec, fine_fascia_sec, lambda [job/min])
LAMBDA_SLOTS = [
    (0,     14399, 0.83),  # 00:00–03:59  LOW
    (14400, 28799, 1.66),  # 04:00–07:59  HIGH
    (28800, 43199, 1.16),  # 08:00–11:59  MEDIUM
    (43200, 57599, 1.90),  # 12:00–15:59  VERY HIGH
    (57600, 71999, 1.49),  # 16:00–19:59  MEDIUM
    (72000, 86399, 1.24),  # 20:00–23:59  LOW-MED
]
# media = (0.95 + 1.00 + 1.98 + 1.30 + 1.95 + 1.10) / 6 = 1.38
LAMBDA = 1.38
SLOT_DURATION = 14400  # 4 ore di simulazione per ogni λ

PC_VALUES =[0.1, 0.4, 0.5, 0.7, 0.9]
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
COORD_SERVICE_P3P4 = 0.6   # media esponenziale


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

# --------------------------------
# PARAMETRI MODELLO MIGLIORATIVO
# --------------------------------
# QoS
SLO_EDGE       = 3.0
W_TARGET       = 2.4   # 20% sotto QoS

# Finestra e ritmi
SCALING_WINDOW      = 600.0    # 10 minuti
SCALE_COOLDOWN_UP   = 600.0    # 10 minuti
SCALE_COOLDOWN_DOWN = 1800.0   # 30 minuti
DOWNSCALE_HOLD      = 1800.0   # 30 minuti
FREEZE_AT_END       = 0.0      # opzionale; lascia 0 per semplicità

# Isteresi (UP aggressivo, DOWN conservativo)
W_UP,   W_DOWN   = 2.6, 2.1
LQ_UP,  LQ_DOWN  = 4.0, 1.2
RHO_UP, RHO_DOWN = 0.78, 0.55

# Drain C e kicker
C_DRAIN_THRESH   = 2    # non scendere se open_C > 2
FORCE_DOWN_AFTER = 2    # se m*=1 per 2 finestre e segnali bassi: forza 2→1

# Limiti Edge
EDGE_SERVERS_MAX = 3    # 3 basta anche per p_c alti in questo schema (aumenta se necessario)