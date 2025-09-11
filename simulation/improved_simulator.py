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
from utils.improved_simulation_stats import SimulationStats_improved, ReplicationStats_improved
import utils.constants as cs
from libraries.rngs import plantSeeds, getSeed, selectStream, random as rng_random

# Inizializzazione semi randomici
plantSeeds(cs.SEED)

# --- Parametri del modello locale a questo file ---
EDGE_M = cs.EDGE_SERVERS

def _update_cloud_next_completion_improved(stats):
    stats.t.completion_cloud = (min(stats.cloud_comp) if stats.cloud_comp else cs.INFINITY)


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

    # Cloud ∞-server: lista completamenti
    if not hasattr(stats, 'cloud_comp'):
        stats.cloud_comp = []  # lista di tempi di completamento
    if not hasattr(stats.t, 'completion_cloud'):
        stats.t.completion_cloud = cs.INFINITY

    # Feedback queue (post-Cloud)
    if not hasattr(stats, 'queue_feedback'):
        stats.queue_feedback = []
    if not hasattr(stats, 'number_feedback'):
        stats.number_feedback = 0
    if not hasattr(stats.t, 'completion_feedback'):
        stats.t.completion_feedback = cs.INFINITY
    if not hasattr(stats, 'index_feedback'):
        stats.index_feedback = 0

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
                service = GetServiceEdgeE_im()  # Usa Ts = EDGE_SERVICE_E_im (0.42s)

            else:
                print("Con il nuovo modello non si servono più job 'C' all'Edge")
                return # (non dovrebbe mai capitare)



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

    # Routing post-Edge
    if job_type == "E":
        selectStream(3)
        stats.number_E -= 1
        r = rng_random()  # numero casuale per la classificazione globale

        if r < cs.P_C:
            # --- CLOUD ∞-SERVER: ogni arrivo avvia il proprio servizio indipendente ---
            stats.number_cloud += 1
            service = GetServiceCloud()
            stats.area_cloud.service += service
            stats.area_C.service     += service

            # assicurati che esista la lista dei completamenti Cloud
            if not hasattr(stats, 'cloud_comp'):
                stats.cloud_comp = []
            stats.cloud_comp.append(stats.t.current + service)

            # aggiorna il prossimo completamento Cloud al minimo della lista
            _update_cloud_next_completion_improved(stats)

        else:
            # Va al Coordinator Server Edge (P1..P4)
            stats.number_coord += 1
            # Dopo aver deciso che va al Coordinator:
            # Dopo aver deciso che va al Coordinator:
            coord_rand = (r - cs.P_C) / (1.0 - cs.P_C)  # ∈ [0,1] condizionato

            # Soglie condizionate: normalizza dividendo per P_COORD
            t1 = cs.P1_PROB
            t2 = (cs.P1_PROB + cs.P2_PROB)
            t3 = (cs.P1_PROB + cs.P2_PROB + cs.P3_PROB)

            if coord_rand < t1:
                stats.queue_coord_low.append("P1")
            elif coord_rand < t2:
                stats.queue_coord_low.append("P2")
            elif coord_rand < t3:
                stats.queue_coord_high.append("P3")
            else:
                stats.queue_coord_high.append("P4")

            # se il Coordinator era idle, avvia subito un servizio
            if stats.number_coord == 1:
                if stats.queue_coord_high:
                    service = GetServiceCoordP3P4()
                else:
                    service = GetServiceCoordP1P2()
                stats.t.completion_coord = stats.t.current + service
                stats.area_coord.service += service

    # Prova ad assegnare un nuovo job al server Edge appena liberato
    _edge_try_start_service_improved(stats)
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

    T = max(1e-12, stats.t.current - cs.START)
    return return_stats_improved(stats, T, seed), stats


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
        append_stats_improved(batch_stats, results, stats)

        # registra andamento per grafici
        avg_edge = results['edge_NuoviArrivi_avg_wait']
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
    if stats.number_feedback > 0: stats.area_feedback.node += dt * stats.number_feedback

    # --- NEW: integrazione dell’area di servizio Edge (multi-server) ---
    if hasattr(stats, 'edge_busy'):
        busy_edge = sum(1 for b in stats.edge_busy if b)
    else:
        busy_edge = min(stats.number_edge, getattr(stats, 'edge_m', 1))  # fallback
    if busy_edge > 0:
        stats.area_edge.service += dt * busy_edge  # Ls integrale del centro Edge
        stats.area_E.service += dt * busy_edge  # (se tieni il tracking per classe E)

    # Avanza il clock
    stats.t.current = stats.t.next

    # Registrazione per analisi transiente ogni 1000 secondi
    interval = 1000
    if int(stats.t.current) % interval == 0:
        avg_edge = stats.area_edge.node / stats.index_edge if stats.index_edge > 0 else 0
        avg_cloud = cs.CLOUD_SERVICE  # (tempo di servizio fisso, non coda)
        avg_coord = stats.area_coord.node / stats.index_coord if stats.index_coord > 0 else 0
        avg_feedback = stats.area_feedback.node / stats.index_feedback if stats.index_feedback > 0 else 0
        stats.edge_wait_times.append((stats.t.current, avg_edge))
        stats.cloud_wait_times.append((stats.t.current, avg_cloud))
        stats.coord_wait_times.append((stats.t.current, avg_coord))
        stats.feedback_wait_times.append((stats.t.current, avg_feedback))

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
        # Completa ESATTAMENTE un job: rimuovi il min dalla lista
        stats.index_cloud += 1
        stats.number_cloud -= 1
        # rimuovi l'istanza che ha completato (il min corrente)
        try:
            stats.cloud_comp.remove(stats.t.current)
        except ValueError:
            # (se min==current per floating point, fai una rimozione tollerante)
            # rimuovi la più vicina
            if stats.cloud_comp:
                m = min(stats.cloud_comp, key=lambda x: abs(x - stats.t.current))
                stats.cloud_comp.remove(m)
        # nessun "nuovo avvio": gli altri sono già in servizio
        _update_cloud_next_completion_improved(stats)

        # Ora il job va al FEEDBACK (come già fai)
        stats.number_feedback += 1
        stats.queue_feedback.append("FB")
        if stats.number_feedback == 1:
            service = GetServiceFeedback_improved()
            stats.t.completion_feedback = stats.t.current + service
            stats.area_feedback.service += service
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
        stats.index_feedback += 1

        if stats.number_feedback > 0:
            service = GetServiceFeedback_improved()
            stats.t.completion_feedback = stats.t.current + service
            stats.area_feedback.service += service
            # (nessun tracciamento metriche)
        else:
            stats.t.completion_feedback = cs.INFINITY


def return_stats_improved(stats, t, seed):
    """
    Versione 'solo nuovi nomi':
      - Edge_NuoviArrivi_*  (ex-Edge)
      - Edge_Feedback_*     (stadio post-Cloud)
      - Cloud_*, Coord_*    invariati
    Assicurati di aver chiamato stats.calculate_area_queue() PRIMA.
    """
    # --- Medie per job completato ---
    ENA_W   = (stats.area_edge.node  / stats.index_edge)   if stats.index_edge   > 0 else 0.0
    ENA_Wq  = (stats.area_edge.queue / stats.index_edge)   if stats.index_edge   > 0 else 0.0
    C_W     = cs.CLOUD_SERVICE
    C_Wq    = 0.0
    CO_W    = (stats.area_coord.node / stats.index_coord)  if stats.index_coord  > 0 else 0.0
    CO_Wq   = (stats.area_coord.queue/ stats.index_coord)  if stats.index_coord  > 0 else 0.0

    # --- Medie nel tempo (Little) ---
    ENA_L   = (stats.area_edge.node   / t) if t > 0 else 0.0
    ENA_Lq  = (stats.area_edge.queue  / t) if t > 0 else 0.0
    C_L     = (stats.area_cloud.node  / t) if t > 0 else 0.0
    C_Lq    = (stats.area_cloud.queue / t) if t > 0 else 0.0
    CO_L    = (stats.area_coord.node  / t) if t > 0 else 0.0
    CO_Lq   = (stats.area_coord.queue / t) if t > 0 else 0.0

    # --- Numero medio "nel servente" (Ls) = tempo totale di servizio / orizzonte ---
    ENA_Ls = (stats.area_edge.service / t) if t > 0 else 0.0
    FB_Ls = (stats.area_feedback.service / t) if t > 0 else 0.0

    # --- Utilizzazioni / busy ---

    ENA_rho = (stats.area_edge.service  / (t)) if t > 0 else 0.0
    CO_rho  = (stats.area_coord.service / t)            if t > 0 else 0.0
    C_busy  = (stats.area_cloud.service / t)            if t > 0 else 0.0

    # --- Throughput ---
    ENA_X = (stats.index_edge  / t) if t > 0 else 0.0
    C_X   = (stats.index_cloud / t) if t > 0 else 0.0
    CO_X  = (stats.index_coord / t) if t > 0 else 0.0

    # --- Tempi di servizio empirici ---
    ENA_s = (stats.area_edge.service  / stats.index_edge)  if stats.index_edge  > 0 else 0.0
    C_s   = (stats.area_cloud.service / stats.index_cloud) if stats.index_cloud > 0 else 0.0
    CO_s  = (stats.area_coord.service / stats.index_coord) if stats.index_coord > 0 else 0.0

    # ===== Edge_Feedback (richiede che in loop tu abbia aggiornato area_feedback.* e index_feedback) =====
    fb_compl = getattr(stats, 'index_feedback', 0)
    FB_W   = (stats.area_feedback.node  / fb_compl) if fb_compl > 0 else 0.0
    FB_Wq  = (stats.area_feedback.queue / fb_compl) if fb_compl > 0 else 0.0
    FB_L   = (stats.area_feedback.node  / t) if t > 0 else 0.0
    FB_Lq  = (stats.area_feedback.queue / t) if t > 0 else 0.0
    FB_s   = (stats.area_feedback.service / fb_compl) if fb_compl > 0 else 0.0
    FB_rho = (stats.area_feedback.service / t) if t > 0 else 0.0   # single-server
    FB_X   = (fb_compl / t) if (t > 0) else 0.0

    return {
        'seed': seed,

        # --- Edge_NuoviArrivi (ex-Edge) ---
        'edge_NuoviArrivi_avg_wait':            ENA_W,
        'edge_NuoviArrivi_avg_delay':           ENA_Wq,
        'edge_NuoviArrivi_L':                   ENA_L,
        'edge_NuoviArrivi_Lq':                  ENA_Lq,
        "edge_NuoviArrivi_Ls":                  ENA_Ls,  # << NEW
        'edge_NuoviArrivi_utilization':         ENA_rho,
        'edge_NuoviArrivi_throughput':          ENA_X,
        'edge_NuoviArrivi_service_time_mean':   ENA_s,
        'Edge_NuoviArrivi_E_Ts':                ENA_s,

        # --- Edge_Feedback ---
        'edge_Feedback_avg_wait':               FB_W,
        'edge_Feedback_avg_delay':              FB_Wq,
        'edge_Feedback_L':                      FB_L,
        'edge_Feedback_Lq':                     FB_Lq,
        "edge_Feedback_Ls":                     FB_Ls,  # << NEW
        'edge_Feedback_utilization':            FB_rho,
        'edge_Feedback_throughput':             FB_X,
        'edge_Feedback_service_time_mean':      FB_s,
        'Edge_Feedback_E_Ts':                   FB_s,

        # --- Cloud ---
        'cloud_avg_wait':                       C_W,
        'cloud_avg_delay':                      C_Wq,
        'cloud_L':                              C_L,
        'cloud_Lq':                             C_Lq,
        'cloud_avg_busy_servers':               C_busy,
        'cloud_throughput':                     C_X,
        'cloud_service_time_mean':              C_s,

        # --- Coordinator ---
        'coord_avg_wait':                       CO_W,
        'coord_avg_delay':                      CO_Wq,
        'coord_L':                              CO_L,
        'coord_Lq':                             CO_Lq,
        'coord_utilization':                    CO_rho,
        'coord_throughput':                     CO_X,
        'coord_service_time_mean':              CO_s,

        # --- (opzionale: classe E legacy se ti servono ancora per report interni) ---
        'edge_E_avg_delay': (stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0,
        'edge_E_avg_response': ((stats.area_E.queue / stats.count_E) if stats.count_E > 0 else 0.0) + cs.EDGE_SERVICE_E_im,

        # --- contatori classe (se usati altrove) ---
        'count_E': stats.count_E,
        'count_E_P1': stats.count_E_P1, 'count_E_P2': stats.count_E_P2,
        'count_E_P3': stats.count_E_P3, 'count_E_P4': stats.count_E_P4,
        'count_C': stats.count_C,
    }


