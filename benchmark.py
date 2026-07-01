import sys, os, time, json, random
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

random.seed(42)
np.random.seed(42)

HISTORY_FILE = "benchmark_history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def check_reproducibility():
    from backend.predictor import Predictor
    np.random.seed(42)
    p = Predictor()
    np.random.seed(42)
    r1 = p.predict("Argentina", "Brazil", {"neutral": False})
    np.random.seed(42)
    r2 = p.predict("Argentina", "Brazil", {"neutral": False})
    p1 = r1["stage_11"]["probabilities"]
    p2 = r2["stage_11"]["probabilities"]
    keys = ["win_a", "draw", "win_b"]
    for k in keys:
        if abs(p1[k] - p2[k]) > 1e-10:
            return False
    return True

def run():
    t0 = time.time()

    print("=" * 70)
    print("  BENCHMARK - Metricas Automaticas")
    print("  Forecast - Football Prediction MVP (16 Etapas)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\n[1/3] Verificando reproducibilidad...")
    reproducible = check_reproducibility()
    print(f"  {'OK' if reproducible else 'FALLO'}")
    if not reproducible:
        print("  ADVERTENCIA: el modelo NO es deterministico")

    print("\n[2/3] Ejecutando backtest (500 partidos)...")
    from audit import run_backtest
    result = run_backtest()
    if result is None:
        print("ERROR: backtest fallo")
        sys.exit(1)
    metrics, _, _, _, _, _ = result

    elapsed = time.time() - t0
    metrics["execution_time_s"] = round(elapsed, 2)
    metrics["reproducible"] = reproducible

    print("\n[3/3] Guardando historial...")
    history = load_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "execution_time_s": round(elapsed, 2),
        "reproducible": reproducible,
        "metrics": {k: v for k, v in metrics.items() if k != "config"}
    }
    history.append(entry)
    save_history(history)

    print(f"\n{'=' * 70}")
    print("  COMPARACION vs EJECUCION ANTERIOR")
    print(f"{'=' * 70}")

    if len(history) >= 2:
        prev = history[-2]["metrics"]
        curr = history[-1]["metrics"]
        rows = [
            ("Log Loss",          prev.get("log_loss", "N/A"),       curr.get("log_loss", "N/A"),       True),
            ("Brier",             prev.get("brier", "N/A"),          curr.get("brier", "N/A"),          True),
            ("Top-1 Accuracy",    prev.get("top1_acc", "N/A"),       curr.get("top1_acc", "N/A"),       False),
            ("Exact Score Acc",   prev.get("exact_score_acc", "N/A"),curr.get("exact_score_acc", "N/A"),False),
            ("ECE (avg)",         prev.get("ece_avg", "N/A"),        curr.get("ece_avg", "N/A"),        True),
            ("Tiempo (s)",        prev.get("execution_time_s", "N/A"), curr.get("execution_time_s", "N/A"), True),
        ]
        print(f"  {'Metrica':<20s} {'Anterior':<12s} {'Actual':<12s} {'Delta':<12s}")
        print(f"  {'-'*56}")
        for name, prev_v, curr_v, lower_better in rows:
            if prev_v == "N/A" or curr_v == "N/A":
                print(f"  {name:<20s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s}")
            else:
                delta = curr_v - prev_v
                arrow = "+" if delta > 0 else ""
                label = ""
                if lower_better:
                    label = " ▼" if delta < 0 else (" ▲" if delta > 0 else "  ")
                else:
                    label = " ▲" if delta > 0 else (" ▼" if delta < 0 else "  ")
                pct = (delta / prev_v) * 100 if prev_v != 0 else 0
                print(f"  {name:<20s} {prev_v:<12.4f} {curr_v:<12.4f} {arrow}{delta:<+9.4f} ({pct:+.1f}%){label}")
    else:
        print("  (Primera ejecucion — sin datos anteriores)")

    print(f"\n{'=' * 70}")
    print("  RESUMEN EJECUTIVO")
    print(f"{'=' * 70}")
    print(f"  Log Loss:          {metrics['log_loss']:.4f}  (Elo: {metrics['log_loss_elo']:.4f})")
    print(f"  Brier:             {metrics['brier']:.4f}  (Elo: {metrics['brier_elo']:.4f})")
    print(f"  Top-1 Accuracy:    {metrics['top1_acc']*100:.2f}%  (Elo: {metrics['top1_acc_elo']*100:.2f}%)")
    print(f"  Exact Score Acc:   {metrics['exact_score_acc']*100:.2f}%")
    print(f"  ECE (avg):         {metrics['ece_avg']:.4f}")
    print(f"  Tiempo:            {elapsed:.1f}s")
    print(f"  Reproducible:      {'SI' if reproducible else 'NO'}")
    print(f"  Dataset:           {metrics['config']['dataset_size']} partidos")
    print(f"  Test:              {metrics['config']['succeeded']}/{metrics['config']['test_matches']} exitosos")
    print(f"  Historial:         {len(history)} ejecuciones en {HISTORY_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    run()
