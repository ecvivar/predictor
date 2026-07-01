"""
Optimizacion de pesos del IFF mediante correlacion de Pearson.
Uso: python optimize_weights.py
"""
import sys, os, math, random, time
from datetime import datetime
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
random.seed(42)
np.random.seed(42)

from backend.predictor import Predictor
from backend.data_loader import load_matches, get_team_matches
from backend.utils import elo_update, tournament_k_factor

SAMPLE_SIZE = 300
MIN_DATE = "2018-01-01"
COMPONENTS = ["elo", "form", "perf", "opponent", "context", "h2h", "squad"]

def resolve_goal_diff(m):
    return m["home_score"] - m["away_score"]

def run():
    print("=" * 70)
    print("  OPTIMIZACION DE PESOS IFF")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Muestra: {SAMPLE_SIZE} partidos desde {MIN_DATE}")
    print("=" * 70)

    print("\n[1/4] Cargando partidos...")
    all_matches = load_matches()
    candidates = [m for m in all_matches if m["date"] >= MIN_DATE]
    random.shuffle(candidates)
    sample = candidates[:SAMPLE_SIZE]
    print(f"  Total: {len(all_matches)}, Muestra: {len(sample)}")

    print("\n[2/4] Precomputando Elo historico...")
    elo_final, elo_history = {}, defaultdict(list)
    for m in all_matches:
        k = tournament_k_factor(m["tournament"])
        nh, na = elo_update(
            elo_final.get(m["home_team"], 1500),
            elo_final.get(m["away_team"], 1500),
            m["home_score"], m["away_score"], k=k)
        elo_final[m["home_team"]] = nh
        elo_final[m["away_team"]] = na
        for t, e in [(m["home_team"], nh), (m["away_team"], na)]:
            elo_history[t].append({"date": m["date"], "elo": e})
    print(f"  Equipos: {len(elo_final)}")

    def get_elo_at(team, date):
        if team not in elo_history:
            return 1500
        best = 1500
        for entry in elo_history[team]:
            if entry["date"] <= date:
                best = entry["elo"]
            else:
                break
        return best

    print("\n[3/4] Recolectando scores de componentes...")
    component_diffs = {c: [] for c in COMPONENTS}
    goal_diffs = []
    errors = 0

    base_predictor = Predictor()

    for idx, m in enumerate(sample):
        if (idx + 1) % 50 == 0:
            print(f"  Progreso: {idx+1}/{len(sample)}")
        date = m["date"]
        team_a, team_b = m["home_team"], m["away_team"]

        try:
            matches_up_to = [x for x in all_matches if x["date"] < date]
            base_predictor.matches = matches_up_to
            base_predictor.elo_ratings = {}
            base_predictor.elo_history = defaultdict(list)
            for x in matches_up_to:
                base_predictor.elo_ratings.setdefault(x["home_team"], 1500)
                base_predictor.elo_ratings.setdefault(x["away_team"], 1500)
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
            comps = result["stage_7"]["componentes"]

            # Collect score differences for each component
            for c in COMPONENTS:
                if c == "elo":
                    sa = comps["elo_rating"]["a"]
                    sb = comps["elo_rating"]["b"]
                elif c == "form":
                    sa = comps["forma_reciente"]["a"]
                    sb = comps["forma_reciente"]["b"]
                elif c == "perf":
                    sa = comps["perf_diferencia"]["a"]
                    sb = comps["perf_diferencia"]["b"]
                elif c == "opponent":
                    sa = comps["fortaleza_rivales"]["a"]
                    sb = comps["fortaleza_rivales"]["b"]
                elif c == "context":
                    sa = comps["localia_contexto"]["a"]
                    sb = comps["localia_contexto"]["b"]
                elif c == "h2h":
                    sa = comps["historial_directo"]["a"]
                    sb = comps["historial_directo"]["b"]
                elif c == "squad":
                    sa = comps["calidad_plantel"]["a"]
                    sb = comps["calidad_plantel"]["b"]
                component_diffs[c].append(sa - sb)

            goal_diffs.append(resolve_goal_diff(m))

        except Exception as e:
            errors += 1

    print(f"  Exitosos: {len(goal_diffs)}, Errores: {errors}")
    if len(goal_diffs) < 30:
        print("ERROR: muestra insuficiente")
        return

    print("\n[4/4] Calculando correlaciones y pesos optimos...")
    goal_arr = np.array(goal_diffs)
    raw_corrs = {}
    for c in COMPONENTS:
        comp_arr = np.array(component_diffs[c][:len(goal_diffs)])
        r = np.corrcoef(comp_arr, goal_arr)[0, 1]
        raw_corrs[c] = r if not np.isnan(r) else 0.0
        print(f"  {c:10s}  r = {raw_corrs[c]:+.4f}")

    # Convert to weights: keep only positive correlations, normalize to sum=1
    positive = {c: max(0, r) for c, r in raw_corrs.items()}
    total = sum(positive.values())
    if total > 0:
        weights = {c: round(v / total, 4) for c, v in positive.items()}
    else:
        weights = {c: round(1.0 / len(COMPONENTS), 4) for c in COMPONENTS}

    print(f"\n  Pesos optimizados:")
    print(f"  weights = {weights}")
    w_sum = sum(weights.values())
    print(f"  Suma: {w_sum:.4f}")

    # Check against current weights
    current = {"elo": 0.25, "form": 0.20, "perf": 0.20, "opponent": 0.10,
               "context": 0.15, "h2h": 0.05, "squad": 0.05}
    print(f"\n  Comparacion vs actuales:")
    print(f"  {'Componente':<12s} {'Actual':<8s} {'Optimo':<8s} {'Delta':<8s}")
    print(f"  {'-'*36}")
    for c in COMPONENTS:
        d = weights[c] - current[c]
        sign = "+" if d > 0 else ""
        print(f"  {c:<12s} {current[c]:<8.4f} {weights[c]:<8.4f} {sign}{d:<+.4f}")

    print(f"\n  Codigo para _stage_7:")
    print(f"  weights = {weights}")

if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"  Tiempo total: {time.time() - t0:.1f}s")
