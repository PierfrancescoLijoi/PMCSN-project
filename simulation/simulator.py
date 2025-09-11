# simulation/simulator.py

from utils.sim_utils import *
from utils.simulation_output import write_file, plot_analysis
from utils.simulation_stats import SimulationStats, ReplicationStats
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random
import heapq


# Inizializzazione semi randomici
plantSeeds(cs.SEED)
# stream 0 -> external arrival
# stream 1 -> type E service at Edge node
# stream 2 -> Cloud server service
# stream 3 -> routing probability
# stream 4 -> type C service at Edge node
# stream 5 -> # stream 6 -> coordinator service for P1-P2
# stream 6 -> coordinator service for P3-P4
def finite_simulation(stop, forced_lambda=None):
    """
    Esegue una simulazione fino al tempo 'stop'.
    Se forced_lambda è specificato, usa sempre quel λ invece di GetLambda(current_time).
    """
    seed = getSeed()
    reset_arrival_temp()
    stats = SimulationStats()
    stats.reset(cs.START)
    stats.cloud_heap = []  # min-heap dei tempi di completamento Cloud
    stats.t.completion_cloud = cs.INFINITY

    # Primo arrivo
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)

    # Loop principale di simulazione
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        execute(stats, stop, forced_lambda)

    stats.calculate_area_queue()
    return return_stats(stats, stats.t.current, seed), stats

def infinite_simulation(forced_lambda=None):
    """
    Simulazione a orizzonte infinito (batch-means) per stimare il regime stazionario.
    - forced_lambda: se specificato, usa quel λ fisso invece di GetLambda().
    - sim_type: etichetta usata per salvare i grafici.
    """
    seeds = []
    wait_times_edge, wait_times_cloud, wait_times_coord = [], [], []
    batch_stats = ReplicationStats()
    stats = SimulationStats()
    stats.reset(cs.START)
    stats.cloud_heap = []  # min-heap dei completamenti Cloud (∞-server)

    reset_arrival_temp()  # ← azzera l’orologio degli arrivi
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)  # ← primo arrivo > 0

    seed = getSeed()
    seeds.append(seed)
    start_time = 0
    results_list = []

    while len(batch_stats.edge_wait_times) < cs.K:
        # esegue il batch fino a B job
        while stats.job_arrived < cs.B:
            execute(stats, cs.STOP_INFINITE, forced_lambda)

        stop_time = stats.t.current - start_time
        start_time = stats.t.current

        # calcolo aree batch
        stats.calculate_area_queue()
        results = return_stats(stats, stop_time, seed)
        results["batch"] = len(results_list)  # 0,1,2,... al momento della raccolta

        # salvataggio risultati batch
        results_list.append(results)
        append_stats(batch_stats, results, stats)

        # registra andamento per grafici
        avg_edge = results['edge_avg_wait']
        avg_cloud = results['cloud_avg_wait']
        avg_coord = results['coord_avg_wait']

        if len(wait_times_edge) < len(seeds):
            wait_times_edge.append([])
            wait_times_cloud.append([])
            wait_times_coord.append([])

        wait_times_edge[-1].append((stats.t.current, avg_edge))
        wait_times_cloud[-1].append((stats.t.current, avg_cloud))
        wait_times_coord[-1].append((stats.t.current, avg_coord))

        # reset parziale per batch successivo
        stats.reset_infinite()

    # scarta batch iniziali per rimuovere il transitorio
    remove_batch(batch_stats, 25)

    # attacco lista di risultati all'oggetto batch_stats
    batch_stats.results = results_list
    return batch_stats



def execute(stats, stop, forced_lambda=None):
    """
    Esegue il prossimo evento e aggiorna lo stato della simulazione.
    Supporta forced_lambda per ignorare la logica delle fasce orarie.
    """
    # Prossimo evento
    stats.t.next = Min(stats.t.arrival,
                       stats.t.completion_edge,
                       stats.t.completion_cloud,
                       stats.t.completion_coord)

    # --- Aggiorna aree (globali + per classe all'Edge) ---
    dt = stats.t.next - stats.t.current
    if dt < 0:
        dt = 0.0

    if stats.number_edge > 0:
        stats.area_edge.node += dt * stats.number_edge
    if stats.number_cloud > 0:
        stats.area_cloud.node += dt * stats.number_cloud
    if stats.number_coord > 0:
        stats.area_coord.node += dt * stats.number_coord

    # Classi all'Edge
    if stats.number_E > 0:
        stats.area_E.node += dt * stats.number_E
    if stats.number_C > 0:
        stats.area_C.node += dt * stats.number_C

    stats.t.current = stats.t.next

    # --- Registrazione transiente ogni 1000s (facoltativa/unchanged) ---
    interval = 1000.0
    if not hasattr(stats, "_next_dump"):
        stats._next_dump = interval

    while stats.t.current >= stats._next_dump:
        avg_edge = stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0.0
        avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
        avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0

        # Salva il punto a t = _next_dump (x = tempo simulazione)
        stats.edge_wait_times.append((stats._next_dump, avg_edge))
        stats.cloud_wait_times.append((stats._next_dump, avg_cloud))
        stats.coord_wait_times.append((stats._next_dump, avg_coord))

        # Edge per classi
        avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
        avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
        stats.edge_E_wait_times_interval.append((stats._next_dump, avg_edge_E))
        stats.edge_C_wait_times_interval.append((stats._next_dump, avg_edge_C))

        stats._next_dump += interval

    # --- ARRIVAL (arrivo esterno) ---
    if stats.t.current == stats.t.arrival:
        stats.job_arrived += 1
        stats.number_edge += 1
        stats.number_E    += 1
        stats.queue_edge.append("E")

        # Prossimo arrivo
        stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
        if stats.t.arrival > stop:
            stats.t.arrival = cs.INFINITY

        # Se Edge era idle, parte subito un E
        if stats.number_edge == 1:
            service = GetServiceEdgeE()
            stats.t.completion_edge  = stats.t.current + service
            stats.area_edge.service += service        # tempo occupazione totale Edge
            stats.area_E.service    += service        # quota classe E

    # --- EDGE COMPLETION ---
    elif stats.t.current == stats.t.completion_edge:
        if stats.queue_edge:
            job_type = stats.queue_edge.pop(0)
            stats.index_edge += 1
            stats.number_edge -= 1

            if job_type == "E":
                selectStream(3)
                stats.number_E -= 1
                stats.index_edge_E += 1

                # routing: Cloud (P_C) oppure Coordinator (1-P_C)
                r = rng_random()
                if r < cs.P_C:
                    # verso Cloud (∞-server): ogni job schedula il suo completamento
                    stats.number_cloud += 1
                    service = GetServiceCloud()
                    completion = stats.t.current + service
                    heapq.heappush(stats.cloud_heap, completion)
                    stats.area_cloud.service += service
                    stats.t.completion_cloud = stats.cloud_heap[0] if stats.cloud_heap else cs.INFINITY
                else:
                    # verso Coordinator (P1..P4)
                    stats.number_coord += 1
                    # routing tra le classi del Coordinator: usa le PROBABILITÀ CONDIZIONATE (somma=1)
                    coord_r = (r - cs.P_C) / (1 - cs.P_C)  # U(0,1)
                    if coord_r < cs.P1:
                        stats.queue_coord_low.append("P1")
                    elif coord_r < cs.P1 + cs.P2:
                        stats.queue_coord_low.append("P2")
                    elif coord_r < cs.P1 + cs.P2 + cs.P3:
                        stats.queue_coord_high.append("P3")
                    else:
                        stats.queue_coord_high.append("P4")

                    if stats.number_coord == 1:
                        if stats.queue_coord_high:
                            service = GetServiceCoordP3P4()
                        else:
                            service = GetServiceCoordP1P2()
                        stats.t.completion_coord  = stats.t.current + service
                        stats.area_coord.service += service

            else:  # job_type == "C"
                stats.count_C      += 1
                stats.number_C     -= 1
                stats.index_edge_C += 1

            # Se restano job all'Edge, schedula il prossimo
            if stats.number_edge > 0:
                next_job = stats.queue_edge[0]
                if next_job == "E":
                    service = GetServiceEdgeE()
                    stats.t.completion_edge  = stats.t.current + service
                    stats.area_edge.service += service
                    stats.area_E.service    += service
                else:
                    service = GetServiceEdgeC()
                    stats.t.completion_edge  = stats.t.current + service
                    stats.area_edge.service += service
                    stats.area_C.service    += service
            else:
                stats.t.completion_edge = cs.INFINITY
        else:
            stats.t.completion_edge = cs.INFINITY

    elif stats.t.current == stats.t.completion_cloud:
        stats.index_cloud += 1
        stats.number_cloud -= 1

        if stats.cloud_heap and stats.cloud_heap[0] <= stats.t.current + 1e-12:
            heapq.heappop(stats.cloud_heap)

        stats.t.completion_cloud = stats.cloud_heap[0] if stats.cloud_heap else cs.INFINITY

        # ritorno all'Edge come classe C
        stats.number_edge += 1
        stats.queue_edge.append("C")
        stats.number_C += 1

        if stats.number_edge == 1:
            service = GetServiceEdgeC()
            stats.t.completion_edge = stats.t.current + service
            stats.area_edge.service += service
            stats.area_C.service += service


    # --- COORDINATOR COMPLETION ---
    elif stats.t.current == stats.t.completion_coord:
        stats.index_coord += 1
        stats.number_coord -= 1

        if stats.queue_coord_high:
            p = stats.queue_coord_high.pop(0)
            stats.count_E += 1
            if p == "P3":
                stats.count_E_P3 += 1
            else:
                stats.count_E_P4 += 1
        else:
            p = stats.queue_coord_low.pop(0)
            stats.count_E += 1
            if p == "P1":
                stats.count_E_P1 += 1
            else:
                stats.count_E_P2 += 1

        if stats.number_coord > 0:
            if stats.queue_coord_high:
                service = GetServiceCoordP3P4()
            else:
                service = GetServiceCoordP1P2()
            stats.t.completion_coord  = stats.t.current + service
            stats.area_coord.service += service
        else:
            stats.t.completion_coord = cs.INFINITY

def return_stats(stats, t, seed):
    # Assicurati che calculate_area_queue() sia stato chiamato prima

    # Tempi di risposta medi per centro
    edge_W   = (stats.area_edge.node  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    cloud_W = (stats.area_cloud.node / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    coord_W  = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0

    # Tempi di coda medi per centro
    edge_Wq  = (stats.area_edge.queue  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    cloud_Wq  = (stats.area_cloud.queue / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    coord_Wq = (stats.area_coord.queue / stats.index_coord) if stats.index_coord > 0 else 0.0

    # L e Lq
    edge_L   = (stats.area_edge.node  / t) if t > 0 else 0.0
    cloud_L = (stats.area_cloud.node / t) if t > 0 else 0.0
    coord_L  = (stats.area_coord.node / t) if t > 0 else 0.0

    edge_Lq  = (stats.area_edge.queue  / t) if t > 0 else 0.0
    cloud_Lq = (stats.area_cloud.queue / t) if t > 0 else 0.0
    coord_Lq = (stats.area_coord.queue / t) if t > 0 else 0.0

    # Utilizzazioni
    edge_util  = (stats.area_edge.service  / t) if t > 0 else 0.0   # ∈ [0,1]
    coord_util = (stats.area_coord.service / t) if t > 0 else 0.0   # ∈ [0,1]
    cloud_busy = cloud_L  # ∞-server: busy servers medi = L

    # Throughput
    X_edge   = (stats.index_edge  / t) if t > 0 else 0.0
    X_cloud  = (stats.index_cloud / t) if t > 0 else 0.0
    X_coord  = (stats.index_coord / t) if t > 0 else 0.0

    # Tempi medi di servizio "realizzati"
    s_edge   = (stats.area_edge.service  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    s_cloud  = (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    s_coord  = (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0

    # --- Medie per classe al nodo Edge (usano completamenti Edge della rispettiva classe) ---
    edge_E_W   = (stats.area_E.node   / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq  = (stats.area_E.queue  / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_C_W   = (stats.area_C.node   / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq  = (stats.area_C.queue  / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0

    # Utilizzazioni per classe all'Edge (devono sommare a edge_util)
    edge_E_util = (stats.area_E.service / t) if t > 0 else 0.0
    edge_C_util = (stats.area_C.service / t) if t > 0 else 0.0

    # --- Popolazioni medie per classe all'Edge ---
    edge_E_L = (stats.area_E.node / t) if t > 0 else 0.0
    edge_E_Lq = (stats.area_E.queue / t) if t > 0 else 0.0
    edge_C_L = (stats.area_C.node / t) if t > 0 else 0.0
    edge_C_Lq = (stats.area_C.queue / t) if t > 0 else 0.0

    return {
        'seed': seed,

        # tempi di risposta (centro)
        'edge_avg_wait': edge_W,
        'cloud_avg_wait': cloud_W,
        'coord_avg_wait': coord_W,

        # tempi di coda (centro)
        'edge_avg_delay': edge_Wq,
        'cloud_avg_delay': cloud_Wq,
        'coord_avg_delay': coord_Wq,

        # per classe al nodo Edge
        'edge_E_avg_delay': edge_E_Wq,
        'edge_E_avg_response': edge_E_W,
        'edge_C_avg_delay': edge_C_Wq,
        'edge_C_avg_response': edge_C_W,

        # L e Lq
        'edge_L': edge_L, 'edge_Lq': edge_Lq,
        'cloud_L': cloud_L, 'cloud_Lq': cloud_Lq,
        'coord_L': coord_L, 'coord_Lq': coord_Lq,

        # utilizzazioni
        'edge_utilization': edge_util,
        'coord_utilization': coord_util,
        'cloud_avg_busy_servers': cloud_busy,

        # utilizzazioni per classe (Edge)
        'edge_E_utilization': edge_E_util,
        'edge_C_utilization': edge_C_util,

        # --- NUOVO: L e Lq per classe all'Edge ---
        'edge_E_L': edge_E_L, 'edge_E_Lq': edge_E_Lq,
        'edge_C_L': edge_C_L, 'edge_C_Lq': edge_C_Lq,

        # throughput
        'edge_throughput': X_edge,
        'cloud_throughput': X_cloud,
        'coord_throughput': X_coord,

        # tempi di servizio realizzati
        'edge_service_time_mean': s_edge,
        'cloud_service_time_mean': s_cloud,
        'coord_service_time_mean': s_coord,

        # contatori (breakdown E al Coordinator + C all'Edge)
        'count_E': stats.count_E,
        'count_E_P1': stats.count_E_P1,
        'count_E_P2': stats.count_E_P2,
        'count_E_P3': stats.count_E_P3,
        'count_E_P4': stats.count_E_P4,
        'count_C': stats.count_C,

        # legacy (se li vuoi ancora nel CSV)
        'E_utilization': edge_E_util,
        'C_utilization': edge_C_util,
    }

