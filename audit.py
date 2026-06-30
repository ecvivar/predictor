"""
AUDITORIA CIENTIFICA EJECUTABLE (OPTIMIZADA)
Modelo Predictivo de 16 Etapas - Forecast MVP

Usa snapshots de Elo precomputados para evitar
reconstruir el predictor en cada partido.
"""

import sys, os, math, json, time, random
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.predictor import Predictor
from backend.data_loader import load_matches, compute_elo_history, get_team_matches, get_elo_at_date
from backend.utils import elo_expected, elo_update, tournament_k_factor, poisson_distribution
from backend.utils import monte_carlo_simulation, dixon_coles_adjustment, expected_goals_from_matrix

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TEST_MATCHES = 500
TEST_START  = "2020-01-01"
REPORT_FILE = "audit_report.json"

random.seed(42)
np.random.seed(42)

def resolve_winner(m):
    if m["home_score"] > m["away_score"]:
        return 0
    elif m["home_score"] == m["away_score"]:
        return 1
    return 2

def log_loss(y_true_probs, y_pred_probs, eps=1e-15):
    yp = np.clip(y_pred_probs, eps, 1 - eps)
    return -np.mean(y_true_probs * np.log(yp))

def brier_score(y_true_onehot, y_pred_probs):
    return np.mean(np.sum((y_true_onehot - y_pred_probs) ** 2, axis=1))

def top1_accuracy(y_true, y_pred_probs):
    preds = np.argmax(y_pred_probs, axis=1)
    return np.mean(preds == y_true)

def exact_score_accuracy(y_true_scores, y_pred_scores):
    if not y_true_scores:
        return 0.0
    return sum(1 for p, a in zip(y_pred_scores, y_true_scores) if p == a) / len(y_true_scores)

def calibration_error(y_true, y_pred_probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred_probs >= bins[i]) & (y_pred_probs < bins[i + 1])
        if np.sum(in_bin) == 0:
            continue
        bin_acc = np.mean(y_true[in_bin])
        bin_conf = np.mean(y_pred_probs[in_bin])
        ece += (np.sum(in_bin) / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

def calibration_slope(y_true, y_pred_probs, n_bins=10):
    bin_confidences, bin_accuracies = [], []
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (y_pred_probs >= bins[i]) & (y_pred_probs < bins[i + 1])
        if np.sum(in_bin) >= 5:
            bin_confidences.append(np.mean(y_pred_probs[in_bin]))
            bin_accuracies.append(np.mean(y_true[in_bin]))
    if len(bin_confidences) >= 2:
        return float(np.polyfit(bin_confidences, bin_accuracies, 1)[0])
    return 0.0

# ─── ELO BASELINE ─────────────────────────────────────────────────────────────

def elo_predict_probs(elo_a, elo_b):
    """Elo only: win prob + draw as residual."""
    p_a = elo_expected(elo_a, elo_b)
    p_d = 0.25
    total = p_a + p_d + (1 - p_a)
    return [p_a / total, p_d / total, (1 - p_a) / total]

# ─── RUN BACKTEST ─────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 70)
    print("  AUDITORIA CIENTIFICA - BACKTESTING HISTORICO")
    print("=" * 70)

    print("\n[1/5] Cargando partidos...")
    all_matches = load_matches()
    print(f"      Total: {len(all_matches)}")

    test_candidates = [m for m in all_matches if m["date"] >= TEST_START]
    random.shuffle(test_candidates)
    test_matches = test_candidates[:TEST_MATCHES]
    print(f"      Test ({TEST_START}+): {len(test_matches)} partidos")

    print("\n[2/5] Precomputando Elo historico...")
    # Compute full Elo history once
    elo_final, elo_history = compute_elo_history(all_matches)
    # Build date-sorted Elo snapshots
    print(f"      Equipos con historial: {len(elo_history)}")

    print("\n[3/5] Ejecutando modelo de 16 etapas (time-aware)...")
    full_probs, full_outcomes = [], []
    baseline_probs, baseline_outcomes = [], []
    exact_scores_pred, exact_scores_actual = [], []
    errors = 0

    # Pre-compute Elo at each test match date for all involved teams
    # Build a date index
    def get_elo_snapshot(team, date_str, history):
        if team not in history:
            return 1500
        best = 1500
        for entry in history[team]:
            if entry["date"] <= date_str:
                best = entry["elo"]
            else:
                break
        return best

    # Create predictor once and override elo per match
    base_predictor = Predictor()

    for idx, m in enumerate(test_matches):
        if (idx + 1) % 50 == 0:
            print(f"      Progreso: {idx+1}/{len(test_matches)}")

        date = m["date"]
        team_a = m["home_team"]
        team_b = m["away_team"]

        # Get Elo at match date (snapshot)
        elo_a = get_elo_snapshot(team_a, date, elo_history)
        elo_b = get_elo_snapshot(team_b, date, elo_history)

        # Baseline: Elo-only
        bp = elo_predict_probs(elo_a, elo_b)
        baseline_probs.append(bp)
        baseline_outcomes.append(resolve_winner(m))

        # 16-stage model: use time-appropriate data
        try:
            matches_up_to = [x for x in all_matches if x["date"] < date]
            base_predictor.matches = matches_up_to
            base_predictor.elo_ratings = {}
            base_predictor.elo_history = defaultdict(list)
            # Only compute Elo for teams that existed
            for x in matches_up_to:
                for t in [x["home_team"], x["away_team"]]:
                    if t not in base_predictor.elo_ratings:
                        base_predictor.elo_ratings[t] = 1500
            for x in matches_up_to:
                k = tournament_k_factor(x["tournament"])
                nh, na = elo_update(
                    base_predictor.elo_ratings[x["home_team"]],
                    base_predictor.elo_ratings[x["away_team"]],
                    x["home_score"], x["away_score"], k=k)
                base_predictor.elo_ratings[x["home_team"]] = nh
                base_predictor.elo_ratings[x["away_team"]] = na
                base_predictor.elo_history[x["home_team"]].append({"date": x["date"], "elo": nh})
                base_predictor.elo_history[x["away_team"]].append({"date": x["date"], "elo": na})
            base_predictor.teams = sorted(base_predictor.elo_ratings.keys())
            base_predictor.global_avg_goals = 1.35
            base_predictor._compute_global_stats()
            base_predictor._compute_recent_elo_trends()

            params = {"neutral": m.get("neutral", False),
                "home_team": m["home_team"],
                "altitude": 0, "temperature": 20, "humidity": 50,
                "rest_days_a": 7, "rest_days_b": 7,
                "injuries_a": 0, "injuries_b": 0,
                "suspensions_a": 0, "suspensions_b": 0,
                "travel_a": 0, "travel_b": 0, "importance": "normal"}

            result = base_predictor.predict(team_a, team_b, params)
            probs = result["stage_11"]["probabilities"]
            p = [probs["win_a"], probs["draw"], probs["win_b"]]
            s = sum(p)
            if s > 0:
                p = [x / s for x in p]
            full_probs.append(p)
            full_outcomes.append(resolve_winner(m))

            # Exact score
            try:
                top_scores = result.get("stage_16", {}).get("top_10_marcadores", [])
                if top_scores:
                    exact_scores_pred.append(top_scores[0]["score"])
                    exact_scores_actual.append(f"{m['home_score']}-{m['away_score']}")
            except Exception:
                pass

        except Exception as e:
            errors += 1

    full_probs = np.array(full_probs)
    full_outcomes = np.array(full_outcomes)
    baseline_probs = np.array(baseline_probs)
    baseline_outcomes = np.array(baseline_outcomes)

    print(f"\n      Exitosos: {len(full_probs)}, Errores: {errors}")
    if len(full_probs) < 50:
        print("  ERROR: muy pocas predicciones exitosas")
        return None

    # ─── METRICAS ─────────────────────────────────────────────────────────
    print("\n[4/5] Calculando metricas...")

    n = len(full_outcomes)
    y_true_onehot = np.zeros((n, 3))
    y_true_onehot[np.arange(n), full_outcomes] = 1

    nb = len(baseline_outcomes)
    y_base_onehot = np.zeros((nb, 3))
    y_base_onehot[np.arange(nb), baseline_outcomes] = 1

    # Cut to same length
    min_n = min(n, nb)
    fp = full_probs[:min_n]
    fo = full_outcomes[:min_n]
    bp = baseline_probs[:min_n]
    bo = baseline_outcomes[:min_n]
    yt = y_true_onehot[:min_n]
    yb = y_base_onehot[:min_n]

    metrics = {}

    metrics["log_loss"] = float(log_loss(yt, fp))
    metrics["log_loss_elo"] = float(log_loss(yb, bp))

    metrics["brier"] = float(brier_score(yt, fp))
    metrics["brier_elo"] = float(brier_score(yb, bp))

    metrics["top1_acc"] = float(top1_accuracy(fo, fp))
    metrics["top1_acc_elo"] = float(top1_accuracy(bo, bp))

    metrics["exact_score_acc"] = float(exact_score_accuracy(exact_scores_actual, exact_scores_pred))
    metrics["exact_score_n"] = len(exact_scores_pred)

    # Calibration
    for label, outcomes, probs in [("full", fo, fp), ("elo", bo, bp)]:
        for i, outcome in enumerate(["win", "draw", "loss"]):
            key = f"ece_{label}_{outcome}"
            metrics[key] = float(calibration_error(
                (outcomes == i).astype(int), probs[:, i]))

    metrics["ece_avg"] = float(np.mean([metrics.get(f"ece_full_{o}", 0) for o in ["win","draw","loss"]]))
    metrics["ece_avg_elo"] = float(np.mean([metrics.get(f"ece_elo_{o}", 0) for o in ["win","draw","loss"]]))

    metrics["cal_slope"] = float(calibration_slope(
        (fo == 0).astype(int), fp[:, 0]))
    metrics["cal_slope_elo"] = float(calibration_slope(
        (bo == 0).astype(int), bp[:, 0]))

    # Normalization
    ps = fp.sum(axis=1)
    metrics["norm_min"] = float(ps.min())
    metrics["norm_max"] = float(ps.max())
    metrics["norm_mean"] = float(ps.mean())
    metrics["norm_std"] = float(ps.std())

    # Naive baseline
    naive_p = np.tile([0.45, 0.25, 0.30], (min_n, 1))
    metrics["log_loss_naive"] = float(log_loss(yt, naive_p))
    metrics["brier_naive"] = float(brier_score(yt, naive_p))
    metrics["top1_acc_naive"] = float(top1_accuracy(fo, naive_p))

    metrics["improvement_logloss_vs_elo"] = float(metrics["log_loss_elo"] - metrics["log_loss"])
    metrics["improvement_brier_vs_elo"] = float(metrics["brier_elo"] - metrics["brier"])
    metrics["improvement_top1_vs_elo"] = float(metrics["top1_acc"] - metrics["top1_acc_elo"])
    metrics["improvement_logloss_vs_naive"] = float(metrics["log_loss_naive"] - metrics["log_loss"])

    # Brier per sample for statistical test
    full_brier_ps = np.sum((yt - fp) ** 2, axis=1)
    base_brier_ps = np.sum((yb - bp) ** 2, axis=1)
    diff = full_brier_ps - base_brier_ps
    t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
    from scipy.stats import norm
    metrics["p_value"] = float(2 * (1 - norm.cdf(abs(t_stat))))

    # ─── ROBUSTEZ ──────────────────────────────────────────────────────────
    print("[5/5] Pruebas de robustez...")

    if len(test_matches) > 0:
        m0 = test_matches[0]
        d0 = m0["date"]
        try:
            mup = [x for x in all_matches if x["date"] < d0]
            p = Predictor()
            p.matches = mup
            p.elo_ratings = {}
            for x in mup:
                for t in [x["home_team"], x["away_team"]]:
                    p.elo_ratings.setdefault(t, 1500)
            for x in mup:
                k = tournament_k_factor(x["tournament"])
                nh, na = elo_update(p.elo_ratings[x["home_team"]], p.elo_ratings[x["away_team"]],
                    x["home_score"], x["away_score"], k=k)
                p.elo_ratings[x["home_team"]] = nh
                p.elo_ratings[x["away_team"]] = na
            p.teams = sorted(p.elo_ratings.keys())
            p._compute_global_stats = lambda: None
            p.global_avg_goals = 1.35
            p.elo_history = defaultdict(list)
            p._compute_recent_elo_trends = lambda: None
            p.elo_recent_trend = {}

            params = {"neutral": False, "home_team": m0["home_team"],
                "altitude": 0, "temperature": 20, "humidity": 50,
                "rest_days_a": 7, "rest_days_b": 7,
                "injuries_a": 0, "injuries_b": 0,
                "suspensions_a": 0, "suspensions_b": 0,
                "travel_a": 0, "travel_b": 0, "importance": "normal"}

            r_base = p.predict(m0["home_team"], m0["away_team"], dict(params))
            p_base = r_base["stage_11"]["probabilities"]

            # Perturb Elo
            old_elos = p.elo_ratings.copy()
            for t in p.elo_ratings:
                p.elo_ratings[t] += 200
            r_elo = p.predict(m0["home_team"], m0["away_team"], dict(params))
            p_elo = r_elo["stage_11"]["probabilities"]
            delta = sum(abs(p_base[k] - p_elo[k]) for k in ["win_a","draw","win_b"])
            metrics["robustness_elo_200"] = float(delta)

            # Restore
            p.elo_ratings = old_elos

            # Injuries
            pi = dict(params)
            pi["injuries_a"] = 3
            r_inj = p.predict(m0["home_team"], m0["away_team"], pi)
            p_inj = r_inj["stage_11"]["probabilities"]
            delta = sum(abs(p_base[k] - p_inj[k]) for k in ["win_a","draw","win_b"])
            metrics["robustness_injuries_3"] = float(delta)

            # Home vs neutral
            pn = dict(params)
            pn["neutral"] = True
            pn.pop("home_team", None)
            r_neut = p.predict(m0["home_team"], m0["away_team"], pn)
            p_neut = r_neut["stage_11"]["probabilities"]
            delta = sum(abs(p_base[k] - p_neut[k]) for k in ["win_a","draw","win_b"])
            metrics["robustness_home_advantage"] = float(delta)

        except Exception as e:
            metrics["robustness_error"] = str(e)

    metrics["config"] = {
        "dataset_size": len(all_matches),
        "test_matches": TEST_MATCHES,
        "succeeded": len(full_probs),
        "errors": errors,
        "test_start": TEST_START,
        "timestamp": datetime.now().isoformat()
    }

    return metrics, test_matches[:min_n], fp, fo, bp, bo

# ─── STATIC FINDINGS ──────────────────────────────────────────────────────────

STATIC_FINDINGS = [
    {"id":"CRIT-001","severity":"CRITICO","title":"Data Leakage: Elo calculado con datos futuros",
     "evidence":"predictor.py:17-18 — self.elo_ratings se computa con TODOS los partidos",
     "explanation":"Al predecir partidos historicos, el modelo usa ratings Elo que reflejan resultados posteriores. Esto infla artificialmente la precision.",
     "impact":"Las metricas de precision NO son representativas del rendimiento real del modelo. El backtesting de este informe corrige esto parcialmente.",
     "solution":"Reconstruir Elo incrementalmente: solo usar partidos anteriores a la fecha del match.",
     "priority":1},
    {"id":"CRIT-002","severity":"CRITICO","title":"Squad quality no deterministica (hash de Python)",
     "evidence":"predictor.py:621-624 — return 90 + hash(team) % 10",
     "explanation":"Python usa PYTHONHASHSEED aleatorio. El hash del mismo equipo cambia entre ejecuciones, produciendo predicciones diferentes.",
     "impact":"El modelo NO es reproducible. Cada reinicio del servidor produce resultados distintos.",
     "solution":"Usar hash determinista: hashlib.md5 o diccionario fijo de puntuaciones.",
     "priority":2},
    {"id":"CRIT-003","severity":"CRITICO","title":"'xG' no es expected goals",
     "evidence":"predictor.py:387-388 — xg = 0.7 * actual_gf + 0.3 * base_exp",
     "explanation":"El 'xG' es 70% goles reales + 30% estimacion de fuerza. Esto requiere datos de tiros (shot-level data) para ser xG real.",
     "impact":"La metrica 'xG' es enganosa. No mide calidad de oportunidades.",
     "solution":"Renombrar a 'rendimiento_ofensivo_ajustado'. No llamarlo xG sin datos de tiros.",
     "priority":3},
    {"id":"HIGH-001","severity":"ALTO","title":"Ataque/Defensa circular con suma fija a 2",
     "evidence":"predictor.py:632-635 — attack_a = (iff_a/total)*2; defense_a = 2 - attack_a",
     "explanation":"Defensa es 2 - Ataque. No hay medicion independiente de capacidad defensiva.",
     "impact":"Equipos con IFF similar siempre tienen attack=defense=1. Se pierde informacion.",
     "solution":"Modelar ataque y defensa por separado desde datos historicos de goles.",
     "priority":4},
    {"id":"HIGH-002","severity":"ALTO","title":"Pesos del MMP elegidos manualmente sin validacion",
     "evidence":"predictor.py:555-558 — weights = { elo:0.25, form:0.20, ... }",
     "explanation":"Los pesos del Meta Modelo Predictivo fueron elegidos a criterio, no optimizados contra datos historicos.",
     "impact":"Los pesos pueden estar lejos del optimo.",
     "solution":"Optimizar pesos mediante grid search o regresion logistica sobre datos historicos.",
     "priority":5},
    {"id":"HIGH-003","severity":"ALTO","title":"Truncamiento de distribucion Poisson a 6 goles",
     "evidence":"utils.py:10 — poisson_distribution(lam, max_goals=6)",
     "explanation":"Distribucion truncada a 6x6=36 resultados. Suma de probabilidades < 1.0 por ignorar goles > 5.",
     "impact":"Subestimacion de resultados de alto marcador. Error en mercados Over/Under.",
     "solution":"Extender a 8x8 o 10x10 y renormalizar.",
     "priority":6},
    {"id":"HIGH-004","severity":"ALTO","title":"Confianza del modelo con constantes magicas",
     "evidence":"predictor.py:908-913 — 50 + edge*120 + 0.75*20 - var_penalty*100",
     "explanation":"La metrica de confianza usa constantes arbitrarias sin justificacion estadistica.",
     "impact":"El 'nivel de confianza' no es interpretable ni calibrado.",
     "solution":"Derivar confianza de intervalos de credibilidad Bayesianos o calibracion historica.",
     "priority":7},
    {"id":"HIGH-005","severity":"ALTO","title":"Falta de modelo de empate en Elo",
     "evidence":"predictor.py:210-211 — solo win_a y win_b en probabilidad Elo",
     "explanation":"El empate se infiere como residual. No hay modelo independiente.",
     "impact":"Probabilidades de empate no calibradas.",
     "solution":"Usar modelo ordenado (Elo con threshold para empate) o incorporar frecuencia historica.",
     "priority":8},
    {"id":"MED-001","severity":"MEDIO","title":"Ajustes contextuales: aditivo vs multiplicativo",
     "evidence":"predictor.py:636-641 — adj sumado, aplicado como 1+adj",
     "explanation":"Ajustes contextuales se calculan como sumas en Stage 6 pero se aplican como multiplicadores en Stage 8.",
     "impact":"Interpretacion inconsistente del ajuste entre etapas.",
     "solution":"Unificar: o todo aditivo sobre log(lambda) o todo multiplicativo sobre lambda.",
     "priority":9},
    {"id":"MED-002","severity":"MEDIO","title":"Dixon-Coles tau puede dar probabilidades negativas",
     "evidence":"utils.py:21 — tau[0][0] = 1 - lam_a * lam_b * rho",
     "explanation":"Si lam_a*lam_b*|rho| > 1, tau se vuelve negativo. Raro pero posible.",
     "impact":"Probabilidades negativas => matriz no valida.",
     "solution":"Agregar clamp: tau[i][j] = max(tau[i][j], 0).",
     "priority":10},
    {"id":"MED-003","severity":"MEDIO","title":"Global avg goals constante (1.35)",
     "evidence":"predictor.py:20 — self.global_avg_goals = 1.35",
     "explanation":"El promedio global de goles es fijo, no se recalcula por epoca futbolistica.",
     "impact":"Descalibracion para epocas con diferente ritmo goleador.",
     "solution":"Calcular global_avg_goals dinamicamente con datos hasta la fecha del match.",
     "priority":11},
    {"id":"MED-004","severity":"MEDIO","title":"Peso de localia fijo (0.15)",
     "evidence":"predictor.py:435-436 — impact_a: 0.15",
     "explanation":"Ventaja de localia constante, pero varia por torneo, epoca y distancia.",
     "impact":"Sobreestimacion en torneos con poca ventaja real.",
     "solution":"Estimar localia por torneo/epoca desde datos historicos.",
     "priority":12},
    {"id":"MED-005","severity":"MEDIO","title":"Probabilidades formateadas con strings en vez de floats",
     "evidence":"predictor.py:722-726 — '42.70%' en vez de 0.427",
     "explanation":"Probabilidades como strings con '%'. Dificulta calculos posteriores.",
     "impact":"Bajo. Solo cosmetico.",
     "solution":"Mantener floats y strings solo para display.",
     "priority":13},
    {"id":"OBS-001","severity":"OBSERVACION","title":"get_elo_at_date existe pero nunca se usa",
     "evidence":"data_loader.py:121 — funcion implementada, nunca llamada",
     "explanation":"Funcion para obtener Elo historico en fecha especifica esta implementada pero no integrada.",
     "impact":"Codigo muerto. Podria resolver CRIT-001 si se integrara.",
     "solution":"Integrarla en el predictor para resolver data leakage.",
     "priority":14},
    {"id":"OBS-002","severity":"OBSERVACION","title":"Sobre-rendimiento siempre da 0",
     "evidence":"predictor.py:407 — sobre_rendimiento = xg - avg(all_xg)",
     "explanation":"Resta el promedio del promedio = 0 siempre.",
     "impact":"Bug cosmetico. No afecta predicciones.",
     "solution":"Calcular como (goles reales - xG) en el periodo.",
     "priority":15}
]

# ─── GENERATE REPORT ──────────────────────────────────────────────────────────

def generate_report(metrics):
    if metrics is None:
        return

    cfg = metrics["config"]
    n = cfg["succeeded"]
    n = cfg["succeeded"]

    print("\n" + "=" * 70)
    print("  INFORME DE AUDITORIA CIENTIFICA")
    print("  Forecast - Football Prediction MVP (16 Etapas)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print(f"\n  1. METRICAS DE RENDIMIENTO PREDICTIVO")
    print("  " + "-" * 66)
    print(f"  Dataset: {cfg['dataset_size']} partidos | Test: {n} partidos (desde {cfg['test_start']})")
    print(f"  Errores: {cfg['errors']}")

    print(f"\n  Log Loss (menor=mejor):")
    print(f"    16 etapas:     {metrics['log_loss']:.4f}")
    print(f"    Baseline Elo:  {metrics['log_loss_elo']:.4f}")
    print(f"    Baseline Naive:{metrics['log_loss_naive']:.4f}")
    ll_imp = metrics['improvement_logloss_vs_elo'] * 100
    print(f"    Mejora vs Elo: {ll_imp:+.2f}%")

    if metrics['log_loss'] <= 0.5:
        ll_g = "BUENO"
    elif metrics['log_loss'] <= 0.65:
        ll_g = "ACEPTABLE"
    else:
        ll_g = "REGULAR"
    print(f"    Calificacion:  {ll_g}")

    print(f"\n  Brier Score (menor=mejor):")
    print(f"    16 etapas:     {metrics['brier']:.4f}")
    print(f"    Baseline Elo:  {metrics['brier_elo']:.4f}")
    print(f"    Baseline Naive:{metrics['brier_naive']:.4f}")
    if metrics['brier'] <= 0.55:
        bs_g = "BUENO"
    elif metrics['brier'] <= 0.65:
        bs_g = "ACEPTABLE"
    else:
        bs_g = "REGULAR"
    print(f"    Calificacion:  {bs_g}")

    print(f"\n  Top-1 Accuracy:")
    print(f"    16 etapas:     {metrics['top1_acc']*100:.2f}%")
    print(f"    Baseline Elo:  {metrics['top1_acc_elo']*100:.2f}%")
    print(f"    Baseline Naive:{metrics['top1_acc_naive']*100:.2f}%")

    print(f"\n  Exact Score Accuracy:")
    print(f"    Acierto:       {metrics['exact_score_acc']*100:.2f}% ({metrics['exact_score_n']} muestras)")

    print(f"\n  Calibracion (ECE):")
    print(f"    16 etapas avg: {metrics['ece_avg']:.4f}")
    for o in ["win","draw","loss"]:
        print(f"      {o}: {metrics.get(f'ece_full_{o}',0):.4f}")
    print(f"    Pendiente:     {metrics['cal_slope']:.3f} (ideal=1.0)")
    if 0.8 <= metrics['cal_slope'] <= 1.2:
        print("    -> Bien calibrado")
    elif metrics['cal_slope'] < 0.8:
        print("    -> Subestima prob. altas (overconfidence)")
    else:
        print("    -> Sobreestima diferencias")

    print(f"\n  Significancia:")
    print(f"    P-valor:       {metrics.get('p_value',0):.4f}")
    if metrics.get('p_value',1) < 0.05:
        print("    -> Diferencia ESTADISTICAMENTE SIGNIFICATIVA vs Elo")
    else:
        print("    -> Diferencia NO significativa vs Elo")

    print(f"\n  Normalizacion:")
    print(f"    Media: {metrics['norm_mean']:.6f} (debe=1.0)")
    print(f"    Std:   {metrics['norm_std']:.6f}")
    print(f"    Min:   {metrics['norm_min']:.6f}, Max: {metrics['norm_max']:.6f}")
    if abs(metrics['norm_mean'] - 1.0) < 0.001:
        print("    -> NORMALIZADAS")
    else:
        d = (metrics['norm_mean'] - 1.0) * 100
        print(f"    -> DESVIACION: {d:+.4f}%")

    print(f"\n  Robustez (delta acumulado en 3 probabilidades):")
    for k, v in sorted(metrics.items()):
        if k.startswith("robustness"):
            label = k.replace("robustness_","").replace("_"," ").title()
            print(f"    {label}: {v:.4f}")

    # ─── SAVE JSON BEFORE ANY PRINTS ───────────────────────────────────────
    report_json = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "findings": STATIC_FINDINGS
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    # ─── SCORES ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  2. TARJETA DE PUNTAJE")
    print("=" * 70)

    arq = 6  # Modular, acoplamiento fuerte entre stages
    math_m = 5  # Fundamentos correctos, errores de implementacion
    if abs(metrics['norm_mean'] - 1.0) > 0.01:
        math_m -= 1
    var = 5  # Variables relevantes pero redundancia y pesos manuales
    cal = 4 if metrics['ece_avg'] > 0.10 else (6 if metrics['ece_avg'] > 0.05 else 7)
    rob = 5
    sens = metrics.get('robustness_elo_200', 0)
    if isinstance(sens, float) and sens > 0.3:
        rob -= 1
    gen = 6  # Backtesting en periodo 2020+
    interp = 7  # 16 etapas bien documentadas
    code = 4  # Hash bug, data leakage, codigo muerto
    sci = 3  # Varios errores conceptuales
    prec = 6 if metrics['log_loss'] <= 0.5 else (5 if metrics['log_loss'] <= 0.65 else 4)
    conf = 4  # Data leakage presente en codigo original

    scores = {
        "Arquitectura": arq,
        "Modelo Matematico": math_m,
        "Variables": var,
        "Calibracion": min(cal, 10),
        "Robustez": max(rob, 1),
        "Generalizacion": gen,
        "Interpretabilidad": interp,
        "Codigo": code,
        "Calidad Cientifica": sci,
        "Precision Esperada": prec,
        "Confiabilidad": conf
    }

    for cat, sc in scores.items():
        bar = "#" * sc + "-" * (10 - sc)
        print(f"  {cat:25s} [{bar}] {sc}/10")

    overall = sum(scores.values()) / len(scores)
    print(f"\n  {'GLOBAL':25s} [{'#' * round(overall) + '-' * (10 - round(overall))}] {overall:.1f}/10")

    if overall >= 7:
        verdict = "APROBADO CON RESERVAS - Corregir hallazgos criticos"
    elif overall >= 5:
        verdict = "NO APROBADO - Deficiencias cientificas significativas"
    else:
        verdict = "RECHAZADO - No cumple estandares minimos"
    print(f"\n  VEREDICTO: {verdict}")

    # ─── FINDINGS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  3. HALLAZGOS DE AUDITORIA")
    print("=" * 70)
    for f in STATIC_FINDINGS:
        print(f"\n  [{f['severity']:8s}] {f['id']}: {f['title']}")
        print(f"  {f['solution']}")
        print(f"  Prioridad: #{f['priority']}")

    # ─── MEJORAS ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  4. PLAN PRIORIZADO DE MEJORAS")
    print("=" * 70)
    mejoras = [
        ("Data leakage (Elo futuro)", "CRIT-001", "Alta: +5-15% precision real", 1),
        ("Hash determinista squad quality", "CRIT-002", "Alta: reproducibilidad", 2),
        ("Renombrar pseudo-xG", "CRIT-003", "Baja: claridad", 3),
        ("Ataque/defensa independientes", "HIGH-001", "Alta: +3-8%", 4),
        ("Optimizar pesos MMP", "HIGH-002", "Alta: +5-10%", 5),
        ("Poisson 10x10 con renormalizacion", "HIGH-003", "Media: +1-3%", 6),
        ("Confianza calibrada", "HIGH-004", "Media: interpretabilidad", 7),
        ("Modelo empate en Elo", "HIGH-005", "Media: +2-5%", 8),
        ("Unificar ajustes aditivo/multiplicativo", "MED-001", "Baja: consistencia", 9),
        ("Clamp Dixon-Coles tau", "MED-002", "Baja: estabilidad", 10),
        ("Global avg goals dinamico", "MED-003", "Media: +1-3%", 11),
        ("Localia variable por torneo", "MED-004", "Media: +1-3%", 12),
    ]
    for t, ref, imp, p in mejoras:
        print(f"\n  #{p}. {t}")
        print(f"     {ref} | Impacto: {imp}")

    # ─── RESUMEN ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  5. RESUMEN EJECUTIVO")
    print("=" * 70)
    print(f"""
  COMPONENTE:   Motor predictivo de 16 etapas para selecciones nacionales
  DATASET:      martj42/international_results ({cfg['dataset_size']} partidos)
  PERIODO TEST: {cfg['test_start']} - actual ({n} partidos)

  METRICAS AJUSTADAS (sin data leakage):
  - Log Loss:  {metrics['log_loss']:.4f} ({ll_g})
  - Brier:     {metrics['brier']:.4f} ({bs_g})
  - Top-1 Acc: {metrics['top1_acc']*100:.1f}%
  - ECE:       {metrics['ece_avg']:.4f}
  - Exact Scr: {metrics['exact_score_acc']*100:.2f}%

  PUNTAJE GLOBAL: {overall:.1f}/10
  VEREDICTO: {verdict}

  FORTALEZAS PRINCIPALES:
  - Arquitectura modular de 16 etapas totalmente trazable
  - Cobertura amplia de variables contextuales
  - Dixon-Coles + Monte Carlo como estado del arte
  - Alta interpretabilidad

  DEBILIDADES PRINCIPALES:
  - DATA LEAKAGE: Elo calculado con datos futuros (CRIT-001)
  - Modelo NO reproducible entre ejecuciones (CRIT-002)
  - 'xG' no es expected goals real (CRIT-003)
  - Pesos no derivados de datos (HIGH-002)
  - Sin calibracion contra mercado real

  RECOMENDACION: {verdict}
  Corregir prioritariamente CRIT-001 y CRIT-002 antes de cualquier uso en produccion.
  Luego optimizar pesos (HIGH-002) y modelar ataque/defensa independientes (HIGH-001).
  """.strip())

    print(f"\n  Reporte JSON guardado en: {REPORT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    t0 = time.time()
    result = run_backtest()
    if result:
        metrics, _, _, _, _, _ = result
        generate_report(metrics)
    print(f"\n  Tiempo total: {time.time() - t0:.1f}s")
