# utils/improved_simulation_stats.py
import utils.constants as cs

class Track:
    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0

class Time_improved:
    def __init__(self):
        self.arrival = -1
        self.completion_edge = float("inf")
        self.completion_cloud = float("inf")
        self.completion_coord = float("inf")

        self.completion_feedback = float("inf")
        self.current = 0.0
        self.next = 0.0
        self.last = 0.0



class SimulationStats_improved:
    def __init__(self):
        self.count_P1 = 0
        self.count_P2 = 0
        self.count_P3 = 0
        self.count_P4 = 0
        self.count_P5 = 0
        self.job_arrived = 0
        self.index_edge = 0
        self.index_cloud = 0
        self.index_coord = 0
        self.count_E = 0  # Aggiunto contatore per pacchetti E (P1-P4)
        self.count_C = 0  # Già presente per pacchetti C (P5)
        self.count_E_P1 = 0
        self.count_E_P2 = 0
        self.count_E_P3 = 0
        self.count_E_P4 = 0


        self.number_edge = 0
        self.number_cloud = 0
        self.number_coord = 0
        self.number_E = 0
        self.number_C = 0

        self.area_edge = Track()
        self.area_cloud = Track()
        self.area_coord = Track()
        self.area_E = Track()
        self.area_C = Track()

        self.t = Time_improved()
        self.queue_edge = []
        self.queue_coord_high = []
        self.queue_coord_low = []

        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []

        self.number_feedback = 0
        self.area_feedback = Track()
        self.queue_feedback = []

    def reset(self, start_time):
        self.t.current = start_time
        self.t.arrival = start_time
        self.t.completion_edge = float("inf")
        self.t.completion_cloud = float("inf")
        self.t.completion_coord = float("inf")
        self.queue_edge.clear()
        self.queue_coord_high.clear()
        self.queue_coord_low.clear()
        self.count_E = 0  # Reset del contatore E
        self.count_C = 0  # Reset del contatore C

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

    def calculate_area_queue(self):
        self.area_edge.queue = self.area_edge.node - self.area_edge.service
        self.area_cloud.queue = self.area_cloud.node - self.area_cloud.service
        self.area_coord.queue = self.area_coord.node - self.area_coord.service

        self.area_E.queue = self.area_E.node - self.area_E.service
        self.area_C.queue = self.area_C.node - self.area_C.service

class ReplicationStats_improved:
    def __init__(self):
        self.seeds = []
        self.lambdas = []
        self.slots = []

        # tempi medi attesa e risposta classe E
        self.edge_E_delay_times = []
        self.edge_E_response_times = []

        # tempi di risposta (già esistenti)
        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []

        # nuovi: tempi di attesa in coda
        self.edge_delay_times = []
        self.cloud_delay_times = []
        self.coord_delay_times = []

        # L e Lq
        self.edge_L = []; self.edge_Lq = []
        self.cloud_L = []; self.cloud_Lq = []
        self.coord_L = []; self.coord_Lq = []

        # utilizzazioni
        self.edge_utilization = []
        self.coord_utilization = []
        self.cloud_busy = []  # n° medio server occupati

        # throughput
        self.edge_X = []; self.cloud_X = []; self.coord_X = []

        # per grafici transiente (già esistenti)
        self.edge_wait_interval = []
        self.cloud_wait_interval = []
        self.coord_wait_interval = []
