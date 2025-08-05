# simulation/simulator.py

from utils.sim_utils import *
from utils.simulation_stats import SimulationStats
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def finite_simulation(stop):
    seed = getSeed()
    reset_arrival_temp()
    stats = SimulationStats()
    stats.reset(cs.START)

    stats.t.arrival = GetArrival(stats.t.current)

    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord > 0):
        execute(stats, stop)

    stats.calculate_area_queue()
    return return_stats(stats, stats.t.current, seed), stats

def execute(stats, stop):
    stats.t.next = Min(stats.t.arrival,
                       stats.t.completion_edge,
                       stats.t.completion_cloud,
                       stats.t.completion_coord)

    # aggiorna aree
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
        if stats.index_edge > 0:
            avg_edge = stats.area_edge.node / stats.index_edge
        else:
            avg_edge = 0
        if stats.index_cloud > 0:
            avg_cloud = stats.area_cloud.node / stats.index_cloud
        else:
            avg_cloud = 0
        if stats.index_coord > 0:
            avg_coord = stats.area_coord.node / stats.index_coord
        else:
            avg_coord = 0

        stats.edge_wait_times.append((stats.t.current, avg_edge))
        stats.cloud_wait_times.append((stats.t.current, avg_cloud))
        stats.coord_wait_times.append((stats.t.current, avg_coord))


    # Gestione arrivo dal sistema
    if stats.t.current == stats.t.arrival:
        stats.job_arrived += 1
        stats.number_edge += 1
        stats.number_E += 1
        stats.queue_edge.append("E")
        stats.t.arrival = GetArrival(stats.t.current)
        if stats.t.arrival > stop:
            stats.t.arrival = cs.INFINITY

        if stats.number_edge == 1:
            service = GetServiceEdgeE()
            stats.t.completion_edge = stats.t.current + service
            stats.area_edge.service += service
            stats.area_E.service += service

    # Completamento Edge
    elif stats.t.current == stats.t.completion_edge:
        if stats.queue_edge:  # <-- evita pop da lista vuota
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

                    # normalizza tra 0 e 1 all'interno del ramo coordinator
                    coord_rand = (rand_val - cs.P_C) / (1 - cs.P_C)
                    if coord_rand < cs.P1_PROB:  # 20% assoluto
                        stats.queue_coord_low.append("P1")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB:  # 25% assoluto
                        stats.queue_coord_low.append("P2")
                    elif coord_rand < cs.P1_PROB + cs.P2_PROB + cs.P3_PROB:  # 10% assoluto
                        stats.queue_coord_high.append("P3")
                    else:  # 5% assoluto
                        stats.queue_coord_high.append("P4")

                    if stats.number_coord == 1:
                        if stats.queue_coord_high:
                            service = GetServiceCoordP3P4()
                        else:
                            service = GetServiceCoordP1P2()
                        stats.t.completion_coord = stats.t.current + service
                        stats.area_coord.service += service

            elif job_type == "C":
                # gestisci pacchetti C (update)
                stats.count_C += 1

            # se rimangono job in Edge, programma il prossimo servizio
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

        # Contiamo separatamente P1/P2/P3/P4
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
    return {
        'seed': seed,
        'edge_avg_wait': stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0,
        'cloud_avg_wait': stats.area_cloud.node / stats.index_cloud if stats.index_cloud > 0 else 0,
        'coord_avg_wait': stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0,
        'count_E': stats.count_E,
        'count_E_P1': stats.count_E_P1,
        'count_E_P2': stats.count_E_P2,
        'count_E_P3': stats.count_E_P3,
        'count_E_P4': stats.count_E_P4,
        'count_C': stats.count_C,
        'E_utilization': stats.area_E.service / t if t > 0 else 0,
        'C_utilization': stats.area_C.service / t if t > 0 else 0,
    }
