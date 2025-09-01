# simulation/simulator.py

from utils.sim_utils import *
from utils.simulation_output import write_file, plot_analysis
from utils.simulation_stats import SimulationStats, ReplicationStats
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

# Inizializzazione semi randomici
plantSeeds(cs.SEED)


def finite_simulation(stop, forced_lambda=None):
    """
    Esegue una simulazione fino al tempo 'stop'.
    Se forced_lambda è specificato, usa sempre quel λ invece di GetLambda(current_time).
    """
    seed = getSeed()
    reset_arrival_temp()
    stats = SimulationStats()
    stats.reset(cs.START)

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
    interval = 1000
    if int(stats.t.current) % interval == 0:
        avg_edge  = stats.area_edge.node  / stats.index_edge  if stats.index_edge  > 0 else 0.0
        avg_cloud = cs.CLOUD_SERVICE  # ∞-server: tempo medio = servizio
        avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0
        stats.edge_wait_times.append((stats.t.current, avg_edge))
        stats.cloud_wait_times.append((stats.t.current, avg_cloud))
        stats.coord_wait_times.append((stats.t.current, avg_coord))

        # Edge per classi (E e C)
        avg_edge_E = stats.area_E.node / stats.index_edge_E if stats.index_edge_E > 0 else 0.0
        avg_edge_C = stats.area_C.node / stats.index_edge_C if stats.index_edge_C > 0 else 0.0
        stats.edge_E_wait_times_interval.append((stats.t.current, avg_edge_E))
        stats.edge_C_wait_times_interval.append((stats.t.current, avg_edge_C))

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
                    # verso Cloud
                    stats.number_cloud += 1
                    if stats.number_cloud == 1:
                        service = GetServiceCloud()
                        stats.t.completion_cloud  = stats.t.current + service
                        stats.area_cloud.service += service    # SOLO Cloud qui (non toccare area_C!)
                else:
                    # verso Coordinator (P1..P4)
                    stats.number_coord += 1
                    coord_r = (r - cs.P_C) / (1 - cs.P_C)
                    if coord_r < cs.P1_PROB:
                        stats.queue_coord_low.append("P1")
                    elif coord_r < cs.P1_PROB + cs.P2_PROB:
                        stats.queue_coord_low.append("P2")
                    elif coord_r < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
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

    # --- CLOUD COMPLETION ---
    elif stats.t.current == stats.t.completion_cloud:
        stats.index_cloud += 1
        stats.number_cloud -= 1

        # se altri job al Cloud, parte il prossimo servizio Cloud
        if stats.number_cloud > 0:
            service = GetServiceCloud()
            stats.t.completion_cloud  = stats.t.current + service
            stats.area_cloud.service += service   # SOLO Cloud
        else:
            stats.t.completion_cloud = cs.INFINITY

        # ritorno all'Edge come classe C
        stats.number_edge += 1
        stats.queue_edge.append("C")
        stats.number_C += 1

        # se Edge era idle, parte subito un C
        if stats.number_edge == 1:
            service = GetServiceEdgeC()
            stats.t.completion_edge  = stats.t.current + service
            stats.area_edge.service += service
            stats.area_C.service    += service     # quota C all'Edge

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
    cloud_W  = cs.CLOUD_SERVICE  # ∞-server: tempo medio = servizio
    coord_W  = (stats.area_coord.node / stats.index_coord) if stats.index_coord > 0 else 0.0

    # Tempi di coda medi per centro
    edge_Wq  = (stats.area_edge.queue  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    cloud_Wq = 0.0
    coord_Wq = (stats.area_coord.queue / stats.index_coord) if stats.index_coord > 0 else 0.0

    # L e Lq
    edge_L   = (stats.area_edge.node  / t) if t > 0 else 0.0
    cloud_L  = (stats.index_cloud / t) * cs.CLOUD_SERVICE
    coord_L  = (stats.area_coord.node / t) if t > 0 else 0.0

    edge_Lq  = (stats.area_edge.queue  / t) if t > 0 else 0.0
    cloud_Lq = 0.0
    coord_Lq = (stats.area_coord.queue / t) if t > 0 else 0.0

    # Utilizzazioni
    edge_util  = (stats.area_edge.service  / t) if t > 0 else 0.0   # ∈ [0,1]
    coord_util = (stats.area_coord.service / t) if t > 0 else 0.0   # ∈ [0,1]
    cloud_busy = (stats.index_cloud / t) * cs.CLOUD_SERVICE if t > 0 else 0.0

    # Throughput
    X_edge   = (stats.index_edge  / t) if t > 0 else 0.0
    X_cloud  = (stats.index_cloud / t) if t > 0 else 0.0
    X_coord  = (stats.index_coord / t) if t > 0 else 0.0

    # Tempi medi di servizio "realizzati"
    s_edge   = (stats.area_edge.service  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    s_cloud  = (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    s_coord  = (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0

    # --- Ripartizione per classe robusta (mantiene L e Lq coerenti) ---
    # Condividiamo l'area di nodo dell'Edge fra E e C in proporzione all'area di servizio
    # (invarianti: area_E.node + area_C.node == area_edge.node, idem per le queue).
    total_serv = stats.area_edge.service
    share_E = (stats.area_E.service / total_serv) if total_serv > 0 else 0.0
    share_C = (stats.area_C.service / total_serv) if total_serv > 0 else 0.0

    # Aree "ricostruite" per classe (durante il batch corrente)
    E_node_area = share_E * stats.area_edge.node
    C_node_area = share_C * stats.area_edge.node
    E_serv_area = stats.area_E.service
    C_serv_area = stats.area_C.service

    # Queue areas (clamp a zero per tolleranze numeriche)
    E_queue_area = max(0.0, E_node_area - E_serv_area)
    C_queue_area = max(0.0, C_node_area - C_serv_area)

    # --- Medie per classe al nodo Edge ---
    # Tempi medi (risposta / coda) calcolati con i completamenti della rispettiva classe
    edge_E_W = (E_node_area / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_E_Wq = (E_queue_area / stats.index_edge_E) if stats.index_edge_E > 0 else 0.0
    edge_C_W = (C_node_area / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0
    edge_C_Wq = (C_queue_area / stats.index_edge_C) if stats.index_edge_C > 0 else 0.0

    # Utilizzazioni per classe (invarianti: sommano a edge_util)
    edge_E_util = (E_serv_area / t) if t > 0 else 0.0
    edge_C_util = (C_serv_area / t) if t > 0 else 0.0

    # Popolazioni medie per classe
    edge_E_L = (E_node_area / t) if t > 0 else 0.0
    edge_E_Lq = (E_queue_area / t) if t > 0 else 0.0
    edge_C_L = (C_node_area / t) if t > 0 else 0.0
    edge_C_Lq = (C_queue_area / t) if t > 0 else 0.0

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

