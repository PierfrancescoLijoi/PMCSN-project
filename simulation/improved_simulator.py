# simulation/improved_simulator.py
#
# Modello aggiornato secondo le specifiche richieste:
# - Nodo Edge = coda FIFO con **2 server (dual-core)**, servizio Exp(mean = 0.5*4 = 2.0s)
#   (gestisce solo job di classe "E" provenienti dagli arrivi)
# - Il resto del flusso resta invariato fino al Cloud/Coordinator.
# - Dopo il Cloud, i pacchetti **NON** tornano all'Edge come "C":
#   finiscono in una **nuova coda "feedback" FIFO** a **singolo server**
#   con servizio Exp(mean = 0.5*2 = 1.0s), poi escono dal sistema.
#
# Nota: NON modifichiamo intestazioni CSV, plotting e raccolta metriche;
#       la coda "feedback" è interna al flusso e non ha metriche dedicate.

from utils.sim_utils import *
from utils.improved_simulation_output import write_file_improved, plot_analysis_improved
from utils.improved_simulation_stats import SimulationStats_improved, ReplicationStats_improved
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

# Inizializzazione semi randomici
plantSeeds(cs.SEED)

# --- Parametri del modello locale a questo file ---
EDGE_M = 2  # dual-core fisso per il modello base

# Helpers ---------------------------------------------------------------

def _GetServiceFeedback_improved():
    """Servizio della coda feedback: Exp(mean = 0.5 * 2 = 1.0s)."""
    return Exponential(0.5 * 2)


def _ensure_runtime_structs_improved(stats):
    """
    Inizializza, se mancanti, le strutture runtime per:
    - Edge multi-server (dual-core)
    - Coda di feedback (post-Cloud, single-server)
    - Tempo di completamento feedback nell'orologio globale
    """
    # Edge: array di completamenti e flag busy
    if not hasattr(stats, 'edge_m'):
        stats.edge_m = EDGE_M
    if not hasattr(stats, 'edge_comp'):
        stats.edge_comp = [cs.INFINITY] * stats.edge_m
    if not hasattr(stats, 'edge_busy'):
        stats.edge_busy = [False] * stats.edge_m
    if not hasattr(stats, 'edge_jobtype'):
        stats.edge_jobtype = [None] * stats.edge_m  # "E" nell'attuale modello

    # Feedback queue (post-Cloud)
    if not hasattr(stats, 'queue_feedback'):
        stats.queue_feedback = []
    if not hasattr(stats, 'number_feedback'):
        stats.number_feedback = 0
    if not hasattr(stats.t, 'completion_feedback'):
        stats.t.completion_feedback = cs.INFINITY

    # Garantisce coerenza del completion Edge
    _update_edge_next_completion_improved(stats)


def _update_edge_next_completion_improved(stats):
    """Aggiorna stats.t.completion_edge con il minimo tra i server Edge attivi."""
    if hasattr(stats, 'edge_comp'):
        mn = min(stats.edge_comp) if stats.edge_comp else cs.INFINITY
        stats.t.completion_edge = mn
    else:
        # fallback single-server (non dovrebbe mai capitare qui)
        pass


def _edge_try_start_service_improved(stats):
    """
    Prova ad assegnare job dall'edge queue ai server liberi.
    Consuma dalla queue_edge (FIFO) e programma il completamento sui server liberi.
    Aggiorna area di servizio (sia centro Edge sia classe E).
    """
    for s in range(stats.edge_m):
        if not stats.edge_busy[s] and stats.queue_edge:
            job = stats.queue_edge.pop(0)  # nel modello attuale è sempre "E"
            # Avvio servizio su server s
            if job == "E":
                service = GetServiceEdgeE()
                stats.area_E.service += service  # per classe E
            else:
                # Con il nuovo modello non si servono più job "C" all'Edge
                service = GetServiceEdgeE()

            stats.area_edge.service += service  # tempo totale di servizio erogato all'Edge
            stats.edge_comp[s] = stats.t.current + service
            stats.edge_busy[s] = True
            stats.edge_jobtype[s] = job

    _update_edge_next_completion_improved(stats)


def _edge_complete_one_improved(stats):
    """
    Gestisce un completamento al nodo Edge: individua il server che ha terminato,
    libera il server, aggiorna contatori/indici e inoltra il job (Cloud/Coordinator).
    Ritorna True se un server è stato gestito, False altrimenti.
    """
    tnow = stats.t.current
    # Individua il/i server che hanno completato in questo istante
    cand = [i for i, tc in enumerate(stats.edge_comp) if tc == tnow]
    if not cand:
        return False

    s = cand[0]  # gestiamo un server per evento; altri verranno presi in eventi successivi
    job_type = stats.edge_jobtype[s] or "E"

    # Libera il server
    stats.edge_busy[s] = False
    stats.edge_comp[s] = cs.INFINITY
    stats.edge_jobtype[s] = None

    # Aggiorna metriche del centro
    stats.index_edge += 1
    stats.number_edge -= 1

    # Routing post-Edge (come in modello attuale):
    if job_type == "E":
        selectStream(3)
        stats.number_E -= 1
        rand_val = rng_random()  # numero casuale per la classificazione globale

        if rand_val < cs.P_C:
            # Va al Cloud
            stats.number_cloud += 1
            if stats.number_cloud == 1:
                service = GetServiceCloud()
                stats.t.completion_cloud = stats.t.current + service
                stats.area_cloud.service += service
                stats.area_C.service += service
        else:
            # Va al Coordinator Server Edge (P1..P4)
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

    # (Se ci fossero "C", qui potremmo aggiornare count_C, ma nel nuovo modello non vengono più serviti)
    _edge_try_start_service_improved(stats)  # prova ad assegnare un nuovo job al server liberato
    _update_edge_next_completion_improved(stats)
    return True


# API pubbliche ---------------------------------------------------------

def finite_simulation_improved(stop, forced_lambda=None):
    """
    Esegue una simulazione fino al tempo 'stop'.
    Se forced_lambda è specificato, usa sempre quel λ invece di GetLambda(current_time).
    """
    seed = getSeed()
    reset_arrival_temp()
    stats = SimulationStats_improved()
    stats.reset(cs.START)

    # Inizializza strutture runtime per edge dual-core + feedback
    _ensure_runtime_structs_improved(stats)

    # Primo arrivo
    stats.t.arrival = GetArrival(stats.t.current, forced_lambda)

    # Loop principale di simulazione
    while (stats.t.arrival < stop) or (stats.number_edge + stats.number_cloud + stats.number_coord + stats.number_feedback > 0):
        execute_improved(stats, stop, forced_lambda)

    stats.calculate_area_queue()
    return return_stats_improved(stats, stats.t.current, seed), stats


def infinite_simulation_improved(forced_lambda=None):
    """
    Simulazione a orizzonte infinito (batch-means) per stimare il regime stazionario.
    - forced_lambda: se specificato, usa quel λ fisso invece di GetLambda().
    """
    seeds = []
    wait_times_edge, wait_times_cloud, wait_times_coord = [], [], []
    batch_stats = ReplicationStats_improved()
    stats = SimulationStats_improved()
    stats.reset(cs.START)

    # Inizializza strutture runtime per edge dual-core + feedback
    _ensure_runtime_structs_improved(stats)

    seed = getSeed()
    seeds.append(seed)
    start_time = 0
    results_list = []

    while len(batch_stats.edge_wait_times) < cs.K:
        # esegue il batch fino a B job
        while stats.job_arrived < cs.B:
            execute_improved(stats, cs.STOP_INFINITE, forced_lambda)

        stop_time = stats.t.current - start_time
        start_time = stats.t.current

        # calcolo aree batch
        stats.calculate_area_queue()
        results = return_stats_improved(stats, stop_time, seed)

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


def execute_improved(stats, stop, forced_lambda=None):
    """
    Esegue il prossimo evento e aggiorna lo stato della simulazione.
    Supporta forced_lambda per ignorare la logica delle fasce orarie.
    """
    # Lazy init (necessaria se execute chiamato senza passare da finite_simulation/infinite_simulation)
    _ensure_runtime_structs_improved(stats)

    # Prossimo evento (includiamo anche il completamento della coda feedback)
    stats.t.next = Min(stats.t.arrival,
                       stats.t.completion_edge,
                       stats.t.completion_cloud,
                       stats.t.completion_coord,
                       stats.t.completion_feedback)

    # Aggiorna aree (tempo integrale * numerosità nel nodo)
    dt = stats.t.next - stats.t.current
    if dt < 0:
        dt = 0.0  # salvaguardia numerica

    if stats.number_edge > 0:
        stats.area_edge.node += dt * stats.number_edge
    if stats.number_cloud > 0:
        stats.area_cloud.node += dt * stats.number_cloud
    if stats.number_coord > 0:
        stats.area_coord.node += dt * stats.number_coord
    if stats.number_E > 0:
        stats.area_E.node += dt * stats.number_E

    # Avanza il clock
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

    # Gestione eventi ----------------------------------------------------

    # 1) Arrivo nel sistema → va all'Edge (classe "E")
    if stats.t.current == stats.t.arrival:
        stats.job_arrived += 1
        stats.number_edge += 1
        stats.number_E += 1
        stats.queue_edge.append("E")

        # Calcolo prossimo arrivo
        stats.t.arrival = GetArrival(stats.t.current, forced_lambda)
        if stats.t.arrival > stop:
            stats.t.arrival = cs.INFINITY

        # Se ci sono server liberi, avvia subito il servizio
        _edge_try_start_service_improved(stats)

    # 2) Completamento Edge (dual-core)
    elif stats.t.current == stats.t.completion_edge:
        # gestisce un solo server per evento
        handled = _edge_complete_one_improved(stats)
        if not handled:
            # nessun server coincideva con il tempo corrente: protezione numerica
            _update_edge_next_completion_improved(stats)

    # 3) Completamento Cloud → (NUOVO) va nella coda feedback (single server)
    elif stats.t.current == stats.t.completion_cloud:
        stats.index_cloud += 1
        stats.number_cloud -= 1

        if stats.number_cloud > 0:
            service = GetServiceCloud()
            stats.t.completion_cloud = stats.t.current + service
            stats.area_cloud.service += service
        else:
            stats.t.completion_cloud = cs.INFINITY

        # Routing modificato: niente ritorno all'Edge come "C"
        # Entra nella coda feedback (single server FIFO)
        stats.number_feedback += 1
        stats.queue_feedback.append("FB")
        if stats.number_feedback == 1:
            service = _GetServiceFeedback_improved()  # nuovo generatore (locale a questo file)
            stats.t.completion_feedback = stats.t.current + service
            # (Non tracciamo metriche dedicate della coda feedback)

    # 4) Completamento Coordinator Edge
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

    # 5) Completamento coda feedback → uscita dal sistema
    elif stats.t.current == stats.t.completion_feedback:
        # Consuma il job in testa
        if stats.queue_feedback:
            stats.queue_feedback.pop(0)
        stats.number_feedback -= 1

        if stats.number_feedback > 0:
            service = _GetServiceFeedback_improved()
            stats.t.completion_feedback = stats.t.current + service
            # (nessun tracciamento metriche)
        else:
            stats.t.completion_feedback = cs.INFINITY


def return_stats_improved(stats, t, seed):
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

    m_edge = getattr(stats, 'edge_m', 1)
    edge_util  = (stats.area_edge.service / (t * m_edge)) if t > 0 else 0.0  # frazione per server (0..1)
    coord_util = stats.area_coord.service / t if t > 0 else 0.0             # mono-server
    cloud_busy = stats.area_cloud.service / t if t > 0 else 0.0             # ∞-server: n° medio server occupati

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

        # tempi di coda e risposta medi per classe E (Edge)
        'edge_E_avg_delay': (stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0,
        'edge_E_avg_response': ((stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0) \
                               + cs.EDGE_SERVICE_E_im,

        # nuove: tempi di attesa in coda
        'edge_avg_delay': edge_Wq,
        'cloud_avg_delay': cloud_Wq,
        'coord_avg_delay': coord_Wq,

        # L e Lq
        'edge_L': edge_L, 'edge_Lq': edge_Lq,
        'cloud_L': cloud_L, 'cloud_Lq': cloud_Lq,
        'coord_L': coord_L, 'coord_Lq': coord_Lq,

        # utilizzazioni
        'edge_utilization': edge_util,        # ora normalizzata per numero server Edge
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
