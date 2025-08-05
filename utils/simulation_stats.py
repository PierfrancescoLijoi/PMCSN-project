"""
utils/simulation_stats.py
-------------------------
Strutture dati per registrare le statistiche della simulazione.

Riferimenti:
- Sezione "Modello computazionale" (documento PMCSN Project, Luglio 2025).
- Sezione "Eventi": gestione arrivi e completamenti a Edge, Cloud e Coordinator.
- Obiettivi: tempi medi di attesa, numero di job processati (E e C), utilizzazioni.

Le metriche seguono l’approccio classico:
- Area sotto le curve (#job * tempo) per calcolare valori medi.
- Contatori per job arrivati/completati.
- Liste per salvare tempi di attesa durante l’analisi transiente.
"""

class Track:
    """
    Classe per memorizzare le aree integrate (tempo × numero job).
    - node: tempo integrato dei job nel nodo (Edge, Cloud o Coordinator).
    - queue: tempo integrato dei job in coda.
    - service: tempo integrato dei job in servizio.

    Riferimento: approccio di area under the curve descritto a pag. 8–9.
    """
    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class Time:
    """
    Gestione dei tempi degli eventi principali:
    - arrival: prossimo arrivo (E da e-gate).
    - completion_edge: prossimo completamento al nodo Edge.
    - completion_cloud: prossimo completamento al Cloud.
    - completion_coord: prossimo completamento al Coordinator Edge.
    - current: tempo corrente della simulazione.
    - next: tempo del prossimo evento (minimo dei precedenti).
    - last: tempo dell’ultimo arrivo valido (per STOP).

    Riferimento: algoritmo next-event-driven (Sezione 5.1).
    """
    def __init__(self):
        self.arrival = -1
        self.completion_edge = float("inf")
        self.completion_cloud = float("inf")
        self.completion_coord = float("inf")
        self.current = 0.0
        self.next = 0.0
        self.last = 0.0


class SimulationStats:
    """
    Contiene tutte le statistiche di una singola esecuzione della simulazione.
    """
    def __init__(self):
        # Contatori job
        self.job_arrived = 0         # totale job entrati nel sistema
        self.index_edge = 0          # job completati all’Edge
        self.index_cloud = 0         # job completati al Cloud
        self.index_coord = 0         # job completati al Coordinator Edge
        self.count_E = 0             # job E usciti dal sistema (P1..P4)
        self.count_C = 0             # job C usciti dal sistema (P5 update)

        # Numero job attualmente nei centri
        self.number_edge = 0
        self.number_cloud = 0
        self.number_coord = 0
        self.number_E = 0            # job E in Edge
        self.number_C = 0            # job C in Edge

        # Aree integrate per calcolare metriche medie
        self.area_edge = Track()
        self.area_cloud = Track()
        self.area_coord = Track()
        self.area_E = Track()        # job di classe E
        self.area_C = Track()        # job di classe C

        # Oggetto per la gestione dei tempi
        self.t = Time()

        # Code per i vari centri
        self.queue_edge = []          # FIFO Edge
        self.queue_coord_high = []    # coda prioritaria per P3/P4
        self.queue_coord_low = []     # coda secondaria per P1/P2

        # Liste per tracciare tempi di attesa (per transiente)
        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []

    def reset(self, start_time):
        """
        Reset dei valori per una nuova simulazione o replica.
        - Riferimento: Sezione 5.1 → la simulazione comincia a START.
        """
        self.t.current = start_time
        self.t.arrival = start_time
        self.t.completion_edge = float("inf")
        self.t.completion_cloud = float("inf")
        self.t.completion_coord = float("inf")

        # Pulizia code
        self.queue_edge.clear()
        self.queue_coord_high.clear()
        self.queue_coord_low.clear()

        # Pulizia metriche transiente
        self.edge_wait_times.clear()
        self.cloud_wait_times.clear()
        self.coord_wait_times.clear()

        # Reset contatori dinamici
        self.job_arrived = 0
        self.index_edge = 0
        self.index_cloud = 0
        self.index_coord = 0
        self.count_E = 0
        self.count_C = 0
        self.number_edge = 0
        self.number_cloud = 0
        self.number_coord = 0
        self.number_E = 0
        self.number_C = 0

        # Reset aree integrate
        self.area_edge = Track()
        self.area_cloud = Track()
        self.area_coord = Track()
        self.area_E = Track()
        self.area_C = Track()

    def calculate_area_queue(self):
        """
        Calcola le aree delle code (nodo - servizio).
        Riferimento: metodologia classica delle aree (Sezione 5.2).
        """
        self.area_edge.queue = self.area_edge.node - self.area_edge.service
        self.area_cloud.queue = self.area_cloud.node - self.area_cloud.service
        self.area_coord.queue = self.area_coord.node - self.area_coord.service


class ReplicationStats:
    """
    Statistiche aggregate su più repliche della simulazione.

    - seeds: lista dei seed usati (riproducibilità).
    - edge_wait_times / cloud_wait_times / coord_wait_times:
      tempi medi raccolti per ogni replica.
    """
    def __init__(self):
        self.seeds = []
        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []
        # Aggiungo anche i dati per l'analisi transiente
        self.edge_wait_interval = []   # (tempo, attesa media) per ogni replica
        self.cloud_wait_interval = []
        self.coord_wait_interval = []