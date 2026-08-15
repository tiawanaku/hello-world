# -*- coding: utf-8 -*-
"""
================================================================================
 XAUUSD - PRONOSTICO DE RUTA: MONTE CARLO MARKOV-SWITCHING (sin EMA)
================================================================================
 Que hace:
   1. Carga datos H1 de XAUUSD (CSV de MT5) o genera datos demo.
   2. Calibra un HMM de 3 regimenes (alcista / bajista / rango).
   3. Detecta el REGIMEN ACTUAL del mercado (ultima vela).
   4. Lanza 10,000 simulaciones Monte Carlo desde el precio actual,
      arrancando en el regimen actual detectado.
   5. Reporta la RUTA pronosticada en puntos de control (4h, 8h, 12h...):
        - mediana (camino central)
        - banda de confianza (por defecto 80%: percentiles 10-90)
        - probabilidad de estar por encima del precio inicial
   6. Reporta niveles alcanzables con cierta probabilidad (primer paso):
        "con 80% de prob. el precio TOCA X hacia arriba / Y hacia abajo"
   7. Genera fan chart (abanico de percentiles) e histograma final.

 USO:
   python xauusd_ruta_montecarlo.py                    -> modo demo
   python xauusd_ruta_montecarlo.py XAUUSD_H1.csv      -> con tus datos MT5
   python xauusd_ruta_montecarlo.py XAUUSD_H1.csv 4280 -> + prob. de tocar 4280

 DEPENDENCIAS:  pip install numpy pandas matplotlib hmmlearn
================================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACION
# ==============================================================================
CONFIG = {
    "n_states":         3,       # regimenes del HMM
    "n_paths":          10000,   # simulaciones Monte Carlo
    "horizon_hours":    48,      # horizonte del pronostico (velas H1)
    "checkpoint_every": 4,       # puntos de control de la ruta (cada 4 horas)
    "confidence":       0.80,    # banda de confianza para la ruta
    "ou_kappa":         0.03,    # reversion a la media en regimen rango
    "seed":             42,
    "out_prefix":       "xauusd_ruta",
}

# ==============================================================================
# 1. DATOS
# ==============================================================================
def load_mt5_csv(path):
    for sep in ["\t", ",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 4:
                break
        except Exception:
            continue
    else:
        raise ValueError("No pude leer el CSV. Verifica el separador.")
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    if "CLOSE" not in df.columns:
        raise ValueError(f"No encuentro la columna CLOSE. Columnas: {list(df.columns)}")
    closes = pd.to_numeric(df["CLOSE"], errors="coerce").dropna().values
    print(f"[DATOS] {len(closes)} velas H1 cargadas de {path}")
    print(f"[DATOS] Ultimo cierre: {closes[-1]:.2f}")
    return closes


def generate_demo_data(n=8000, seed=7, p0=4200.0):
    """Datos sinteticos SOLO para el demo (vol diaria ~1-1.8% como el oro)."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.005, 0.010],
                  [0.005, 0.980, 0.015],
                  [0.012, 0.012, 0.976]])
    mus    = [0.00035, -0.00040, 0.0]
    sigmas = [0.0028,   0.0033,  0.0018]
    state, logp = 2, np.log(p0)
    anchor, prices = logp, []
    for _ in range(n):
        state = rng.choice(3, p=P[state])
        if state == 2:
            logp += 0.05 * (anchor - logp) + rng.normal(0, sigmas[2])
        else:
            logp += mus[state] + rng.normal(0, sigmas[state])
            anchor = logp
        prices.append(np.exp(logp))
    print(f"[DEMO] {n} velas sinteticas generadas (usa tu CSV real para calibrar)")
    return np.array(prices)


# ==============================================================================
# 2. CALIBRACION HMM (retorno + canal de drift para separar tendencias)
# ==============================================================================
def fit_hmm(returns, n_states, seed):
    """Devuelve (transicion, mus, sigmas, estados ocultos) ya ordenados:
       0=alcista, 1=bajista, 2=rango."""
    try:
        from hmmlearn.hmm import GaussianHMM
        SCALE = 100.0
        drift = pd.Series(returns).rolling(24, min_periods=1).mean().values
        X = np.column_stack([returns * SCALE, drift * SCALE * 20.0])
        best_model, best_ll = None, -np.inf
        for s in range(5):
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=500, random_state=seed + s, tol=1e-6,
                            min_covar=1e-6)
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_model, best_ll = m, ll
        model  = best_model
        trans  = model.transmat_
        mus    = model.means_[:, 0] / SCALE
        covs   = np.array([np.diag(c)[0] for c in model.covars_])
        sigmas = np.sqrt(covs) / SCALE
        order_key = model.means_[:, 1]
        hidden = model.predict(X)
        print(f"[HMM] hmmlearn OK | log-verosimilitud: {best_ll:.1f}")
    except ImportError:
        print("[HMM] hmmlearn no instalado -> calibrador de respaldo")
        roll = pd.Series(returns).rolling(24, min_periods=1).mean().values
        q_lo, q_hi = np.quantile(roll, [0.30, 0.70])
        hidden = np.where(roll > q_hi, 0, np.where(roll < q_lo, 1, 2))
        trans = np.zeros((n_states, n_states))
        for a, b in zip(hidden[:-1], hidden[1:]):
            trans[a, b] += 1
        trans  = trans / trans.sum(axis=1, keepdims=True)
        mus    = np.array([returns[hidden == s].mean() for s in range(n_states)])
        sigmas = np.array([returns[hidden == s].std()  for s in range(n_states)])
        order_key = mus
    # reordenar: 0=alcista (drift max), 1=bajista (drift min), 2=rango
    order = [int(np.argmax(order_key)), int(np.argmin(order_key))]
    order.append([s for s in range(n_states) if s not in order][0])
    idx   = np.array(order)
    remap = {old: new for new, old in enumerate(idx)}
    return (trans[np.ix_(idx, idx)], mus[idx], sigmas[idx],
            np.array([remap[s] for s in hidden]))


def print_calibration(trans, mus, sigmas, hidden, current_state):
    names = ["ALCISTA", "BAJISTA", "RANGO  "]
    print("\n===== REGIMENES CALIBRADOS =====")
    for s in range(3):
        occup = 100.0 * np.mean(hidden == s)
        dur   = 1.0 / max(1e-9, 1.0 - trans[s, s])
        print(f"  {names[s]} | mu={mus[s]*100:+.4f}%/H1  sigma={sigmas[s]*100:.4f}%  "
              f"ocupacion={occup:5.1f}%  duracion media={dur:6.1f}h (~{dur/24:.1f} dias)")
    print(f"\n  >>> REGIMEN ACTUAL DETECTADO: {names[current_state].strip()} <<<")
    print(f"  (el pronostico arranca desde este regimen, no desde cero)\n")


# ==============================================================================
# 3. MONTE CARLO MARKOV-SWITCHING
# ==============================================================================
def simulate(trans, mus, sigmas, p0, s0, n_paths, n_bars, ou_kappa, seed):
    """Simula n_paths caminos de n_bars velas H1 desde precio p0 y regimen s0.
       Devuelve matriz (n_bars+1, n_paths); la fila 0 es el precio inicial."""
    rng    = np.random.default_rng(seed)
    states = np.full(n_paths, s0)
    logp   = np.full(n_paths, np.log(p0))
    anchor = logp.copy()
    cum_trans = np.cumsum(trans, axis=1)
    prices = np.empty((n_bars + 1, n_paths))
    prices[0] = p0
    for t in range(1, n_bars + 1):
        u = rng.random(n_paths)
        new_states = np.empty(n_paths, dtype=int)
        for s in range(3):
            mask = states == s
            if mask.any():
                new_states[mask] = np.searchsorted(cum_trans[s], u[mask])
        entering_range = (new_states == 2) & (states != 2)
        anchor[entering_range] = logp[entering_range]
        states = new_states
        z = rng.standard_normal(n_paths)
        trend = states != 2
        logp = np.where(
            trend,
            logp + mus[np.clip(states, 0, 1)] + sigmas[np.clip(states, 0, 1)] * z,
            logp + ou_kappa * (anchor - logp) + sigmas[2] * z,
        )
        prices[t] = np.exp(logp)
    return prices


# ==============================================================================
# 4. REPORTE DE LA RUTA
# ==============================================================================
def report_route(paths, p0, cfg, target=None):
    conf   = cfg["confidence"]
    q_lo   = (1.0 - conf) / 2.0 * 100.0        # 80% -> percentil 10
    q_hi   = 100.0 - q_lo                      # 80% -> percentil 90
    step   = cfg["checkpoint_every"]
    n_bars = paths.shape[0] - 1

    print(f"===== RUTA PRONOSTICADA ({paths.shape[1]} simulaciones, "
          f"banda de confianza {conf*100:.0f}%) =====")
    print(f"  Inicio: precio XAUUSD {p0:.2f}\n")
    print(f"  {'hora':>6} | {'mediana':>9} | {'banda ' + format(conf*100,'.0f') + '%':^23} | {'P(subida)':>9}")
    print(f"  {'-'*6} | {'-'*9} | {'-'*23} | {'-'*9}")
    rows = []
    for h in range(step, n_bars + 1, step):
        med = np.median(paths[h])
        lo  = np.percentile(paths[h], q_lo)
        hi  = np.percentile(paths[h], q_hi)
        pup = 100.0 * np.mean(paths[h] > p0)
        rows.append((h, med, lo, hi, pup))
        print(f"  En {h:2d}h | {med:9.2f} | [{lo:9.2f} - {hi:9.2f}] | {pup:7.1f}%")

    # ---- niveles alcanzables (primer paso: usa el MAXIMO/MINIMO del camino)
    run_max = paths.max(axis=0)
    run_min = paths.min(axis=0)
    lvl_up   = np.percentile(run_max, (1.0 - conf) * 100.0)  # 80% de caminos lo tocan
    lvl_down = np.percentile(run_min, conf * 100.0)
    lvl_up50   = np.percentile(run_max, 50.0)
    lvl_down50 = np.percentile(run_min, 50.0)
    print(f"\n===== NIVELES ALCANZABLES EN {n_bars}h (primer paso) =====")
    print(f"  Con {conf*100:.0f}% de probabilidad el precio TOCA >= {lvl_up:.2f} "
          f"({(lvl_up-p0)/0.10:+.0f} pips)")
    print(f"  Con {conf*100:.0f}% de probabilidad el precio TOCA <= {lvl_down:.2f} "
          f"({(lvl_down-p0)/0.10:+.0f} pips)")
    print(f"  Con 50% de probabilidad el precio TOCA >= {lvl_up50:.2f} "
          f"({(lvl_up50-p0)/0.10:+.0f} pips)")
    print(f"  Con 50% de probabilidad el precio TOCA <= {lvl_down50:.2f} "
          f"({(lvl_down50-p0)/0.10:+.0f} pips)")

    # ---- probabilidad de tocar un objetivo dado por el usuario
    if target is not None:
        if target > p0:
            p_touch = 100.0 * np.mean(run_max >= target)
            # primera hora en que la mediana de los caminos que tocan lo alcanza
            touch_hours = [np.argmax(paths[:, j] >= target)
                           for j in range(paths.shape[1]) if run_max[j] >= target]
        else:
            p_touch = 100.0 * np.mean(run_min <= target)
            touch_hours = [np.argmax(paths[:, j] <= target)
                           for j in range(paths.shape[1]) if run_min[j] <= target]
        print(f"\n===== OBJETIVO {target:.2f} =====")
        print(f"  Probabilidad de TOCARLO en {n_bars}h: {p_touch:.1f}%")
        if touch_hours:
            print(f"  Si lo toca, hora tipica (mediana): {int(np.median(touch_hours))}h "
                  f"| mas temprano (p10): {int(np.percentile(touch_hours,10))}h "
                  f"| mas tarde (p90): {int(np.percentile(touch_hours,90))}h")
    return rows


# ==============================================================================
# 5. GRAFICOS
# ==============================================================================
def make_plots(paths, p0, cfg, target=None):
    pre    = cfg["out_prefix"]
    n_bars = paths.shape[0] - 1
    hours  = np.arange(n_bars + 1)

    # (a) fan chart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for lo, hi, a in [(2.5, 97.5, 0.15), (10, 90, 0.25), (25, 75, 0.35)]:
        ax.fill_between(hours, np.percentile(paths, lo, axis=1),
                        np.percentile(paths, hi, axis=1),
                        color="#1565c0", alpha=a,
                        label=f"banda {hi-lo:.0f}%")
    ax.plot(hours, np.median(paths, axis=1), "k-", lw=2, label="mediana (ruta central)")
    rng = np.random.default_rng(1)
    for j in rng.choice(paths.shape[1], size=min(25, paths.shape[1]), replace=False):
        ax.plot(hours, paths[:, j], lw=0.4, alpha=0.4, color="#616161")
    ax.axhline(p0, color="k", ls=":", lw=1)
    if target is not None:
        ax.axhline(target, color="#c62828", ls="--", lw=1.5, label=f"objetivo {target:.2f}")
    ax.set_title(f"XAUUSD: ruta pronosticada a {n_bars}h "
                 f"({paths.shape[1]} simulaciones Markov-switching)")
    ax.set_xlabel("horas"); ax.set_ylabel("USD/oz"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{pre}_fanchart.png", dpi=120); plt.close(fig)

    # (b) histograma del precio final
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(paths[-1], bins=120, color="#2e7d32", alpha=0.85)
    ax.axvline(p0, color="k", ls=":", lw=1.5, label=f"inicio {p0:.2f}")
    ax.axvline(np.median(paths[-1]), color="#c62828", lw=1.5,
               label=f"mediana {np.median(paths[-1]):.2f}")
    if target is not None:
        ax.axvline(target, color="#ef6c00", ls="--", lw=1.5, label=f"objetivo {target:.2f}")
    ax.set_title(f"Distribucion del precio a {n_bars}h")
    ax.set_xlabel("USD/oz"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{pre}_final.png", dpi=120); plt.close(fig)
    print(f"\n[GRAFICOS] Guardados: {pre}_fanchart.png, {pre}_final.png")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    cfg = CONFIG

    # ---- argumentos: [csv] [precio_objetivo]
    csv_path, target = None, None
    for arg in sys.argv[1:]:
        try:
            target = float(arg)
        except ValueError:
            csv_path = arg

    closes = load_mt5_csv(csv_path) if csv_path else generate_demo_data()
    returns = np.diff(np.log(closes))

    # ---- calibrar y detectar regimen actual
    trans, mus, sigmas, hidden = fit_hmm(returns, cfg["n_states"], cfg["seed"])
    current_state = int(hidden[-1])
    print_calibration(trans, mus, sigmas, hidden, current_state)

    # ---- simular desde el ultimo precio y el regimen actual
    p0 = float(closes[-1])
    print(f"[MC] Simulando {cfg['n_paths']} caminos x {cfg['horizon_hours']} velas H1 "
          f"desde {p0:.2f}...\n")
    paths = simulate(trans, mus, sigmas, p0, current_state,
                     cfg["n_paths"], cfg["horizon_hours"], cfg["ou_kappa"], cfg["seed"])

    # ---- reporte de ruta + graficos
    report_route(paths, p0, cfg, target)
    make_plots(paths, p0, cfg, target)
    print("\n[FIN] Ajusta CONFIG (horizonte, checkpoints, confianza) a tu gusto.")


if __name__ == "__main__":
    main()
