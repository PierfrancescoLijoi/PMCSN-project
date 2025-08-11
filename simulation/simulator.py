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

    # plot analisi transiente
    plot_analysis(wait_times_edge, seeds, "edge_node", "standard")
    plot_analysis(wait_times_cloud, seeds, "cloud_server", "standard")
    plot_analysis(wait_times_coord, seeds, "coord_server", "standard")

    # attacco lista di risultati all'oggetto batch_stats
    batch_stats.results = results_list
    return batch_stats



def execute(stats, stop, forced_lambda=None):
    """
    Esegue il prossimo evento e aggiorna lo stato della simulazione.
    Supporta forced_lambda per ignorare la logica delle fasce orarie.
    """
    stats.t.next = Min(stats.t.arrival,
                       stats.t.completion_edge,
                       stats.t.completion_cloud,
                       stats.t.completion_coord)

    # Aggiorna aree
    if stats.number_edge > 0:
        stats.area_edge.node += (stats.t.next - stats.t.current) * stats.number_edge
    if stats.number_cloud > 0:
        stats.area_cloud.node += (stats.t.next - stats.t.current) * stats.number_cloud
    if stats.number_coord > 0:
        stats.area_coord.node += (stats.t.next - stats.t.current) * stats.number_coord

    stats.t.current = stats.t.next

    # Registrazione per analisi transiente ogni 1000 secondi
    interval = 1000
    if int(stats.t.current) % interval == 0:
        avg_edge = stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0
        avg_cloud = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0
        avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0

        stats.edge_wait_times.append((stats.t.current, avg_edge))
        stats.cloud_wait_times.append((stats.t.current, avg_cloud))
        stats.coord_wait_times.append((stats.t.current, avg_coord))

    # Gestione arrivo dal sistema
    if stats.t.current == stats.t.arrival:
        stats.job_arrived += 1
        stats.number_edge += 1
        stats.number_E += 1
        stats.queue_edge.append("E")

        # Calcolo prossimo arrivo
        stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
        if stats.t.arrival > stop:
            stats.t.arrival = cs.INFINITY

        if stats.number_edge == 1:
            service = GetServiceEdgeE()
            stats.t.completion_edge = stats.t.current + service
            stats.area_edge.service += service
            stats.area_E.service += service

    # Completamento Edge
    elif stats.t.current == stats.t.completion_edge:
        if stats.queue_edge:  # evita pop da lista vuota
            job_type = stats.queue_edge.pop(0)
            stats.index_edge += 1
            stats.number_edge -= 1

            if job_type == "E":
                selectStream(3)
                rand_val = rng_random()  # numero casuale per la classificazione globale

                if rand_val < cs.P_C:  # 40% va al Cloud
                    stats.number_cloud += 1
                    if stats.number_cloud == 1:
                        service = GetServiceCloud()
                        stats.t.completion_cloud = stats.t.current + service
                        stats.area_cloud.service += service
                        stats.area_C.service += service
                else:  # 60% va al Coordinator Server Edge
                    stats.number_coord += 1

                    coord_rand = (rand_val - cs.P_C) / (1 - cs.P_C)
                    if coord_rand < cs.P1_PROB:
                        stats.queue_coord_low.append("P1")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB:
                        stats.queue_coord_low.append("P2")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:
                        stats.queue_coord_high.append("P3")
                    else:
                        stats.queue_coord_high.append("P4")

                    if stats.number_coord == 1:
                        if stats.queue_coord_high:
                            service = GetServiceCoordP3P4()
                        else:
                            service = GetServiceCoordP1P2()
                        stats.t.completion_coord = stats.t.current + service
                        stats.area_coord.service += service

            elif job_type == "C":
                stats.count_C += 1

            # Se rimangono job in Edge, programma il prossimo servizio
            if stats.number_edge > 0:
                next_job = stats.queue_edge[0]
                if next_job == "E":
                    service = GetServiceEdgeE()
                    stats.t.completion_edge = stats.t.current + service
                    stats.area_edge.service += service
                    stats.area_E.service += service
                else:  # job_type == "C"
                    service = GetServiceEdgeC()
                    stats.t.completion_edge = stats.t.current + service
                    stats.area_edge.service += service
            else:
                stats.t.completion_edge = cs.INFINITY
        else:
            stats.t.completion_edge = cs.INFINITY

    # Completamento Cloud
    elif stats.t.current == stats.t.completion_cloud:
        stats.index_cloud += 1
        stats.number_cloud -= 1
        if stats.number_cloud > 0:
            service = GetServiceCloud()
            stats.t.completion_cloud = stats.t.current + service
            stats.area_cloud.service += service
        else:
            stats.t.completion_cloud = cs.INFINITY

        stats.number_edge += 1
        stats.queue_edge.append("C")
        if stats.number_edge == 1:
            service = GetServiceEdgeC()
            stats.t.completion_edge = stats.t.current + service
            stats.area_edge.service += service

    # Completamento Coordinator Edge
    elif stats.t.current == stats.t.completion_coord:
        stats.index_coord += 1
        stats.number_coord -= 1

        if stats.queue_coord_high:  # P3/P4
            p_type = stats.queue_coord_high.pop(0)
            stats.count_E += 1
            if p_type == "P3":
                stats.count_E_P3 += 1
            else:
                stats.count_E_P4 += 1
        else:  # P1/P2
            p_type = stats.queue_coord_low.pop(0)
            stats.count_E += 1
            if p_type == "P1":
                stats.count_E_P1 += 1
            else:
                stats.count_E_P2 += 1

        if stats.number_coord > 0:
            if stats.queue_coord_high:
                service = GetServiceCoordP3P4()
            else:
                service = GetServiceCoordP1P2()
            stats.t.completion_coord = stats.t.current + service
            stats.area_coord.service += service
        else:
            stats.t.completion_coord = cs.INFINITY

def return_stats(stats, t, seed):
    # Assicurati che calculate_area_queue() sia stato chiamato prima
    edge_W  = stats.area_edge.node  / stats.index_edge if stats.index_edge  > 0 else 0.0
    cloud_W = stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0.0
    coord_W = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0.0

    edge_Wq  = stats.area_edge.queue  / stats.index_edge if stats.index_edge  > 0 else 0.0
    cloud_Wq = stats.area_cloud.queue / stats.index_cloud if stats.index_cloud > 0 else 0.0
    coord_Wq = stats.area_coord.queue / stats.index_coord if stats.index_coord > 0 else 0.0

    edge_L  = stats.area_edge.node  / t if t > 0 else 0.0
    cloud_L = stats.area_cloud.node / t if t > 0 else 0.0
    coord_L = stats.area_coord.node / t if t > 0 else 0.0

    edge_Lq  = stats.area_edge.queue  / t if t > 0 else 0.0
    cloud_Lq = stats.area_cloud.queue / t if t > 0 else 0.0
    coord_Lq = stats.area_coord.queue / t if t > 0 else 0.0

    edge_util  = stats.area_edge.service  / t if t > 0 else 0.0       # mono-server: frazione ∈ [0,1]
    coord_util = stats.area_coord.service / t if t > 0 else 0.0       # mono-server: frazione ∈ [0,1]
    cloud_busy = stats.area_cloud.service / t if t > 0 else 0.0       # ∞-server: n° medio server occupati

    X_edge  = stats.index_edge  / t if t > 0 else 0.0
    X_cloud = stats.index_cloud / t if t > 0 else 0.0
    X_coord = stats.index_coord / t if t > 0 else 0.0

    s_edge  = stats.area_edge.service  / stats.index_edge  if stats.index_edge  > 0 else 0.0
    s_cloud = stats.area_cloud.service / stats.index_cloud if stats.index_cloud > 0 else 0.0
    s_coord = stats.area_coord.service / stats.index_coord if stats.index_coord > 0 else 0.0

    return {
        'seed': seed,

        # tempi di risposta (già presenti ma manteniamo i nomi)
        'edge_avg_wait': edge_W,
        'cloud_avg_wait': cloud_W,
        'coord_avg_wait': coord_W,

        # nuove: tempi di attesa in coda
        'edge_avg_delay': edge_Wq,
        'cloud_avg_delay': cloud_Wq,
        'coord_avg_delay': coord_Wq,

        # L e Lq
        'edge_L': edge_L, 'edge_Lq': edge_Lq,
        'cloud_L': cloud_L, 'cloud_Lq': cloud_Lq,
        'coord_L': coord_L, 'coord_Lq': coord_Lq,

        # utilizzazioni
        'edge_utilization': edge_util,
        'coord_utilization': coord_util,
        'cloud_avg_busy_servers': cloud_busy,

        # throughput
        'edge_throughput': X_edge,
        'cloud_throughput': X_cloud,
        'coord_throughput': X_coord,

        # tempi di servizio realizzati
        'edge_service_time_mean': s_edge,
        'cloud_service_time_mean': s_cloud,
        'coord_service_time_mean': s_coord,

        # contatori già esistenti
        'count_E': stats.count_E,
        'count_E_P1': stats.count_E_P1,
        'count_E_P2': stats.count_E_P2,
        'count_E_P3': stats.count_E_P3,
        'count_E_P4': stats.count_E_P4,
        'count_C': stats.count_C,

        # NB: queste due nel codice attuale sono “per classe”, non per centro.
        # Le lasciamo per compatibilità, ma ora hai anche quelle per centro.
        'E_utilization': stats.area_E.service / t if t > 0 else 0.0,
        'C_utilization': stats.area_C.service / t if t > 0 else 0.0,
    }
