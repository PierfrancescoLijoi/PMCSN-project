from utils.sim_utils import *
from utils.simulation_stats import SimulationStats
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

plantSeeds(cs.SEED)

def edge_scalability_simulation(stop, forced_lambda=None, slot_index=None):
    seed = getSeed()
    reset_arrival_temp()

    # Almeno 1 server SEMPRE attivo
    cs.EDGE_SERVERS = max(1, getattr(cs, "EDGE_SERVERS", 1))

    stats = SimulationStats()
    stats.reset(cs.START)
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)

    interval = 1000.0
    last_checkpoint = stats.t.current

    # Finestre di misura per utilizzo per ciascun conteggio server (1..MAX)
    server_util_data = {i: {"active_time": 0.0, "area_service": 0.0} for i in range(1, cs.EDGE_SERVERS_MAX + 1)}
    scalability_trace = []

    # Stato dei server Edge
    edge_completion_times = [cs.INFINITY] * cs.EDGE_SERVERS_MAX
    edge_server_busy       = [False]       * cs.EDGE_SERVERS_MAX
    edge_server_jobtype    = [None]        * cs.EDGE_SERVERS_MAX  # "E" o "C"

    def assign_job_to_server(sidx):
        """Prova ad assegnare un job dal fronte coda al server sidx (se libero)."""
        if not edge_server_busy[sidx] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # "E" o "C"
            service = GetServiceEdgeE() if job == "E" else GetServiceEdgeC()
            edge_completion_times[sidx] = stats.t.current + service
            edge_server_busy[sidx] = True
            edge_server_jobtype[sidx] = job
            stats.area_edge.service += service
            if job == "E":
                stats.area_E.service += service
            return True
        return False

    while stats.t.arrival < stop or stats.number_edge + stats.number_cloud + stats.number_coord > 0:
        # Prossimo completamento tra i server attivi
        next_completion_edge = min(edge_completion_times[:cs.EDGE_SERVERS])
        stats.t.next = Min(stats.t.arrival, next_completion_edge, stats.t.completion_cloud, stats.t.completion_coord)

        delta = stats.t.next - stats.t.current

        # Aree numero in nodo (per Little integrali)
        if stats.number_edge > 0:
            stats.area_edge.node += delta * stats.number_edge
        if stats.number_cloud > 0:
            stats.area_cloud.node += delta * stats.number_cloud
        if stats.number_coord > 0:
            stats.area_coord.node += delta * stats.number_coord

        # Accumulo finestra di utilizzo per i server attivi (1..EDGE_SERVERS)
        for i in range(cs.EDGE_SERVERS):
            server_util_data[i + 1]["active_time"] += delta
            if edge_server_busy[i]:
                server_util_data[i + 1]["area_service"] += delta

        stats.t.current = stats.t.next

        # Checkpoint finestra utilizzo
        if stats.t.current - last_checkpoint >= interval:
            # Utilizzo medio aggregato sui server correnti nell'ULTIMA finestra
            total_service = sum(server_util_data[i + 1]["area_service"] for i in range(cs.EDGE_SERVERS))
            total_time    = sum(server_util_data[i + 1]["active_time"]  for i in range(cs.EDGE_SERVERS))
            utilization   = (total_service / total_time) if total_time > 0 else 0.0
            scalability_trace.append((stats.t.current, cs.EDGE_SERVERS, utilization))

            # Decisione scaling
            if utilization > cs.UTILIZATION_UPPER and cs.EDGE_SERVERS < cs.EDGE_SERVERS_MAX:
                # Scale-up: attiva un nuovo server (idle)
                cs.EDGE_SERVERS += 1
                # Il nuovo server è idle per definizione
                edge_completion_times[cs.EDGE_SERVERS - 1] = cs.INFINITY
                edge_server_busy[cs.EDGE_SERVERS - 1] = False
                edge_server_jobtype[cs.EDGE_SERVERS - 1] = None

            elif utilization < cs.UTILIZATION_LOWER and cs.EDGE_SERVERS > 1:
                # Scale-down sicuro: spegni solo se gli ultimi server sono TUTTI idle
                target = cs.EDGE_SERVERS - 1
                # Possiamo ridurre di 1 solo se il server con indice target è idle
                # (si procede a piccoli passi per semplicità/robustezza)
                if not edge_server_busy[target]:
                    # "Spegni" il server con indice target
                    edge_completion_times[target] = cs.INFINITY
                    edge_server_jobtype[target] = None
                    cs.EDGE_SERVERS = max(1, target + 0)  # nuovo numero attivo

            # Reset finestra per prossimi 1000s
            for k in range(1, cs.EDGE_SERVERS_MAX + 1):
                server_util_data[k]["active_time"] = 0.0
                server_util_data[k]["area_service"] = 0.0

            last_checkpoint = stats.t.current

        # Eventi: Arrivo
        if stats.t.current == stats.t.arrival:
            stats.job_arrived += 1
            stats.number_edge += 1
            stats.number_E += 1
            stats.queue_edge.append("E")
            stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
            if stats.t.arrival > stop:
                stats.t.arrival = cs.INFINITY

            # Prova ad assegnare il job in arrivo ad un server libero
            for i in range(cs.EDGE_SERVERS):
                if assign_job_to_server(i):
                    break

        # Eventi: Completamento Edge
        # (gestiamo al massimo un completamento perché stats.t.next è un singolo istante)
        for i in range(cs.EDGE_SERVERS):
            if stats.t.current == edge_completion_times[i]:
                # Job completato su server i
                completed_type = edge_server_jobtype[i]  # "E" o "C"

                # Libera il server i
                edge_server_busy[i] = False
                edge_completion_times[i] = cs.INFINITY
                edge_server_jobtype[i] = None

                stats.index_edge += 1
                stats.number_edge -= 1

                # Routing del job COMPLETATO
                if completed_type == "E":
                    # Decide C o Coordinator
                    selectStream(3)
                    rand_val = rng_random()
                    if rand_val < cs.P_C:
                        # Va al Cloud
                        stats.number_cloud += 1
                        if stats.number_cloud == 1:
                            service = GetServiceCloud()
                            stats.t.completion_cloud = stats.t.current + service
                            stats.area_cloud.service += service
                            stats.area_C.service += service
                    else:
                        # Va al Coordinator (classi P1..P4)
                        stats.number_coord += 1
                        coord_rand = (rand_val - cs.P_C) / (1.0 - cs.P_C)
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
                    # completed_type == "C": job ritornato dal Cloud e servito all'Edge
                    stats.count_C += 1

                # Assegna un nuovo job (se presente) allo stesso server i
                assign_job_to_server(i)
                break  # abbiamo gestito un completamento Edge

        # Eventi: Completamento Cloud
        if stats.t.current == stats.t.completion_cloud:
            stats.index_cloud += 1
            stats.number_cloud -= 1
            if stats.number_cloud > 0:
                service = GetServiceCloud()
                stats.t.completion_cloud = stats.t.current + service
                stats.area_cloud.service += service
            else:
                stats.t.completion_cloud = cs.INFINITY

            # Ritorno al nodo Edge come job "C"
            stats.number_edge += 1
            stats.queue_edge.append("C")
            # Se c'è un server libero, assegna subito
            for i in range(cs.EDGE_SERVERS):
                if assign_job_to_server(i):
                    break

        # Eventi: Completamento Coordinator
        if stats.t.current == stats.t.completion_coord:
            stats.index_coord += 1
            stats.number_coord -= 1

            # Consuma il job in testa (se presente) dando priorità alta
            if stats.queue_coord_high:
                stats.queue_coord_high.pop(0)
                stats.count_E += 1
            elif stats.queue_coord_low:
                stats.queue_coord_low.pop(0)
                stats.count_E += 1
            # altrimenti nessun job da rimuovere; ignora per robustezza

            if stats.number_coord > 0:
                service = GetServiceCoordP3P4() if stats.queue_coord_high else GetServiceCoordP1P2()
                stats.t.completion_coord = stats.t.current + service
                stats.area_coord.service += service
            else:
                stats.t.completion_coord = cs.INFINITY

    # Calcolo aree coda a fine simulazione
    stats.calculate_area_queue()

    # Utilizzo per ciascun conteggio server (sull’intera simulazione)
    per_server_utilization = {}
    for s in range(1, cs.EDGE_SERVERS_MAX + 1):
        data = server_util_data[s]
        per_server_utilization[s] = (data["area_service"] / data["active_time"]) if data["active_time"] > 0 else None

    return {
        'seed': seed,
        'lambda': forced_lambda,
        'slot': slot_index,
        'edge_avg_wait': (stats.area_edge.node / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_avg_delay': (stats.area_edge.queue / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_server_service': (stats.area_edge.service / stats.index_edge) if stats.index_edge > 0 else 0.0,
        'edge_server_utilization': (stats.area_edge.service / stats.t.current) if stats.t.current > 0 else 0.0,
        # Nota: edge_weight_utilization sotto usa il numero di server corrente; se vuoi un peso time-varying
        # serve un accumulo nel tempo. Questo resta compatibile con la tua API.
        'edge_weight_utilization': (stats.area_edge.node / (max(1, cs.EDGE_SERVERS) * stats.t.current)) if stats.t.current > 0 else 0.0,
        'edge_server_number': max(1, cs.EDGE_SERVERS),
        'server_utilization_by_count': per_server_utilization,
        'scalability_trace': scalability_trace
    }, stats
