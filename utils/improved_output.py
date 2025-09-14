from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import utils.constants as cs

def plot_edge_servers_mode_over_time(traces_all, out_png: str, step: float = None):
    """
    traces_all: list of traces (one per replica); each trace is list of (t, m, metrics).
    Calcola la MODA del #server Edge su una griglia temporale uniforme nelle 24h e disegna uno step plot.
    """
    if not traces_all:
        return
    if step is None:
        step = float(getattr(cs, "SCALING_WINDOW", 900.0))
    stop = 86400.0
    times = [t for t in frange(0.0, stop, step)]
    mode_servers = []
    for t in times:
        values = []
        for trace in traces_all:
            # trova l'ultimo punto con tempo <= t
            m_at_t = trace[0][1] if trace else 1
            for (tt, mm, _) in trace:
                if tt <= t:
                    m_at_t = mm
                else:
                    break
            values.append(m_at_t)
        c = Counter(values)
        most = c.most_common()
        if not most:
            mode = 1
        else:
            top_freq = most[0][1]
            candidates = [v for v,f in most if f == top_freq]
            mode = min(candidates)
        mode_servers.append(mode)
    import matplotlib
    matplotlib.use("Agg")
    plt.figure()
    plt.step(times, mode_servers, where="post", label="Edge (moda repliche)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Numero server")
    plt.title("Andamento server nel tempo — moda su repliche")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_edge_response(traces_all, out_png: str):
    """
    Plot della *media cumulata* del tempo di risposta all'Edge:
    W̄(t) = area_edge.node(t) / index_edge(t)  (se index_edge>0)
    Ogni replica produce una curva.
    """
    if not traces_all:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    for trace in traces_all:
        if not trace:
            continue
        times = []
        Wbar  = []
        last  = 0.0
        for (t, _, m) in trace:
            comp = m.get("comp_edge", 0)
            if comp > 0:
                w = m.get("W_bar", 0.0)
                last = w
            else:
                w = last
            times.append(t)
            Wbar.append(w)

        if times and times[0] > 0:
            times = [0.0] + times
            Wbar  = [Wbar[0]] + Wbar
        plt.plot(times, Wbar)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Average response time (s)")
    plt.title("edge_node — finite variable lambda")
    plt.grid(True, linestyle="--", alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def frange(start, stop, step):
    t = start
    while t < stop:
        yield t
        t += step
