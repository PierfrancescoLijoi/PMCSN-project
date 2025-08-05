# utils/simulation_stats.py

class Track:
    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0

class Time:
    def __init__(self):
        self.arrival = -1
        self.completion_edge = float("inf")
        self.completion_cloud = float("inf")
        self.completion_coord = float("inf")
        self.current = 0.0
        self.next = 0.0
        self.last = 0.0

class SimulationStats:
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

        self.t = Time()
        self.queue_edge = []
        self.queue_coord_high = []
        self.queue_coord_low = []

        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []

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

    def calculate_area_queue(self):
        self.area_edge.queue = self.area_edge.node - self.area_edge.service
        self.area_cloud.queue = self.area_cloud.node - self.area_cloud.service
        self.area_coord.queue = self.area_coord.node - self.area_coord.service

class ReplicationStats:
    def __init__(self):
        self.seeds = []
        self.edge_wait_times = []
        self.cloud_wait_times = []
        self.coord_wait_times = []
        self.lambdas = []  # nuovo campo
        self.slots = []  # nuovo campo
        self.edge_wait_interval = []
        self.cloud_wait_interval = []
        self.coord_wait_interval = []