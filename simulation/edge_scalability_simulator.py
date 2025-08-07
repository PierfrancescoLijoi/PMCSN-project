# MODIFICHE COMPLETE A edge_scalability_simulation.py PER SUPPORTARE:
# - Server paralleli: traffico smistato tra più nodi Edge attivi
# - Scaling dinamico basato su utilizzo aggregato ogni 1000s
# - Misurazione per server attivi da 1 a EDGE_SERVERS_MAX

from utils.sim_utils import *
from utils.simulation_stats import SimulationStats
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def edge_scalability_simulation(stop, forced_lambda=None, slot_index=None):
    seed = getSeed()
    reset_arrival_temp()
    cs.EDGE_SERVERS = 1

    stats = SimulationStats()
    stats.reset(cs.START)
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)

    interval = 1000
    last_checkpoint = stats.t.current

    server_util_data = {i: {"active_time": 0.0, "area_service": 0.0} for i in range(1, cs.EDGE_SERVERS_MAX + 1)}
    scalability_trace = []

    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy = [False] * cs.EDGE_SERVERS_MAX

    while stats.t.arrival < stop or stats.number_edge + stats.number_cloud + stats.number_coord > 0:
        next_completion_edge = min(edge_completion_times[:cs.EDGE_SERVERS])
        stats.t.next = Min(stats.t.arrival, next_completion_edge, stats.t.completion_cloud, stats.t.completion_coord)
        delta = stats.t.next - stats.t.current

        if stats.number_edge > 0:
            stats.area_edge.node += delta * stats.number_edge
        if stats.number_cloud > 0:
            stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0:
            stats.area_coord.node += delta * stats.number_coord

        # Corretto: attivo tutti i server, e accumulo area_service solo se occupati
        for i in range(cs.EDGE_SERVERS):
            server_util_data[i + 1]["active_time"] += delta
            if edge_server_busy[i]:
                server_util_data[i + 1]["area_service"] += delta

        stats.t.current = stats.t.next

        if stats.t.current - last_checkpoint >= interval:
            total_service = sum(server_util_data[i + 1]["area_service"] for i in range(cs.EDGE_SERVERS))
            total_time = sum(server_util_data[i + 1]["active_time"] for i in range(cs.EDGE_SERVERS))
            utilization = total_service / total_time if total_time > 0 else 0
            scalability_trace.append((stats.t.current, cs.EDGE_SERVERS, utilization))

            if utilization > cs.UTILIZATION_UPPER and cs.EDGE_SERVERS < cs.EDGE_SERVERS_MAX:
                cs.EDGE_SERVERS += 1
            elif utilization < cs.UTILIZATION_LOWER and cs.EDGE_SERVERS > 1:
                cs.EDGE_SERVERS -= 1

            last_checkpoint = stats.t.current

        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E += 1
            stats.queue_edge.append("E")
            stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
            if stats.t.arrival > stop:
                stats.t.arrival = cs.INFINITY

            for i in range(cs.EDGE_SERVERS):
                if not edge_server_busy[i]:
                    job = stats.queue_edge.pop(0)
                    service = GetServiceEdgeE() if job == "E" else GetServiceEdgeC()
                    edge_completion_times[i] = stats.t.current + service
                    edge_server_busy[i] = True
                    stats.area_edge.service += service
                    if job == "E":
                        stats.area_E.service += service
                    break

        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                stats.index_edge += 1
                stats.number_edge -= 1

                job_type = "E"
                if stats.queue_edge:
                    job_type = stats.queue_edge.pop(0)

                    if job_type == "E":
                        selectStream(3)
                        rand_val = rng_random()
                        if rand_val < cs.P_C:
                            stats.number_cloud += 1
                            if stats.number_cloud == 1:
                                service = GetServiceCloud()
                                stats.t.completion_cloud = stats.t.current + service
                                stats.area_cloud.service += service
                                stats.area_C.service += service
                        else:
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
                                service = GetServiceCoordP3P4() if stats.queue_coord_high else GetServiceCoordP1P2()
                                stats.t.completion_coord = stats.t.current + service
                                stats.area_coord.service += service
                    else:
                        stats.count_C += 1

                    service = GetServiceEdgeE() if job_type == "E" else GetServiceEdgeC()
                    edge_completion_times[i] = stats.t.current + service
                    edge_server_busy[i] = True
                    stats.area_edge.service += service
                    if job_type == "E":
                        stats.area_E.service += service
                break

        if stats.t.current == stats.t.completion_cloud:
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

        if stats.t.current == stats.t.completion_coord:
            stats.index_coord += 1
            stats.number_coord -= 1
            if stats.queue_coord_high:
                stats.queue_coord_high.pop(0)
                stats.count_E += 1
            elif stats.queue_coord_low:
                stats.queue_coord_low.pop(0)
                stats.count_E += 1
            # else: nessun job da rimuovere → inconsistenza, ma non crashiamo

            if stats.number_coord > 0:
                service = GetServiceCoordP3P4() if stats.queue_coord_high else GetServiceCoordP1P2()
                stats.t.completion_coord = stats.t.current + service
                stats.area_coord.service += service
            else:
                stats.t.completion_coord = cs.INFINITY

    stats.calculate_area_queue()

    per_server_utilization = {}
    for s in range(1, cs.EDGE_SERVERS_MAX + 1):
        data = server_util_data[s]
        per_server_utilization[s] = data["area_service"] / data["active_time"] if data["active_time"] > 0 else None

    return {
        'seed': seed,
        'lambda': forced_lambda,
        'slot': slot_index,
        'edge_avg_wait': stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0,
        'edge_avg_delay': stats.area_edge.queue / stats.index_edge if stats.index_edge > 0 else 0,
        'edge_server_service': stats.area_edge.service / stats.index_edge if stats.index_edge > 0 else 0,
        'edge_server_utilization': stats.area_edge.service / stats.t.current if stats.t.current > 0 else 0,
        'edge_weight_utilization': stats.area_edge.node / (cs.EDGE_SERVERS * stats.t.current) if stats.t.current > 0 else 0,
        'edge_server_number': cs.EDGE_SERVERS,
        'server_utilization_by_count': per_server_utilization,
        'scalability_trace': scalability_trace
    }, stats
