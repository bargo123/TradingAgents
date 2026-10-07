import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

DB = r"data_cache\live-market-20260921.db"

def _database_path():
    configured = os.environ.get("TRADINGAGENTS_FOREX_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parent / "data_cache"
    candidates = sorted(
        root.glob("live-market-clean-*.db"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no live shadow database found under {root}")
    return candidates[0]


def _readonly_connect(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = "file:" + quote(path.as_posix(), safe="/:\\") + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3)


DB = _database_path()


def table_exists(c, name):
    return c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    ).fetchone() is not None

def age(ts):
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        s = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
        return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02}"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "-"

while True:
    os.system("cls")
    print("=" * 72)
    print("               TRADINGAGENTS — LIVE MARKET DASHBOARD")
    print("=" * 72)

    try:
        con = _readonly_connect(DB)
        con.row_factory = sqlite3.Row
        c = con.cursor()

        if table_exists(c, "forex_watcher_state"):
            s = c.execute("""
                SELECT lifecycle_status,current_run_id,last_analysis_completed_at,
                       evaluation_due_pending,last_error_code
                FROM forex_watcher_state
                WHERE singleton_id=1
            """).fetchone()

            if s:
                print(f"\nSTATE       : {s['lifecycle_status']}")
                print(f"RUN ID      : {s['current_run_id'] or '-'}")
                print(f"EVAL PENDING: {bool(s['evaluation_due_pending'])}")
                print(f"LAST ERROR  : {s['last_error_code'] or 'NONE'}")

        if table_exists(c, "forex_watch_runs"):
            r = c.execute("""
                SELECT *
                FROM forex_watch_runs
                ORDER BY started_at DESC
                LIMIT 1
            """).fetchone()

            if r:
                print("\n--- CURRENT / LATEST ANALYSIS ---")
                print(f"SYMBOL      : {r['resolved_symbol'] or r['requested_symbol']}")
                print(f"STATUS      : {r['run_status']}")
                print(f"RUNNING FOR : {age(r['started_at'])}")
                print(f"ACTION      : {r['normalized_action'] or 'WAITING...'}")
                print(f"CONTEXT     : {r['decision_context_status'] or 'WAITING...'}")
                print(f"NORMALIZED  : {r['normalization_status'] or 'WAITING...'}")
                print(f"RUNTIME SEC : {r['runtime_seconds'] if r['runtime_seconds'] is not None else 'RUNNING'}")
                print(f"LLM CALLS   : {r['llm_calls'] if r['llm_calls'] is not None else 'RUNNING'}")
                print(f"FAILURE     : {r['failure_code'] or 'NONE'}")

            counts = c.execute("""
                SELECT run_status, COUNT(*) n
                FROM forex_watch_runs
                GROUP BY run_status
            """).fetchall()

            print("\n--- RUN COUNTS ---")
            if counts:
                for x in counts:
                    print(f"{x['run_status']:15}: {x['n']}")
            else:
                print("No completed runs yet.")

        latest_decision = None

        if table_exists(c, "shadow_decisions"):
            latest_decision = c.execute("""
                SELECT *
                FROM shadow_decisions
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()

            print("\n--- LATEST DECISION ---")

            if latest_decision:
                d = latest_decision
                print(f"SYMBOL      : {d['resolved_symbol']}")
                print(f"ACTION      : {d['action']}")
                print(f"BID / ASK   : {d['analysis_snapshot_bid']} / {d['analysis_snapshot_ask']}")
                print(f"SPREAD PTS  : {d['analysis_snapshot_spread_points']}")
                print(f"LATENCY SEC : {d['analysis_latency_seconds']}")
                print(f"CONTEXT     : {d['decision_context_status']}")
                print(f"NORMALIZED  : {d['normalization_status']}")
                print(f"EVALUATION  : {d['future_evaluation_status']}")
                print(f"EXECUTED    : {bool(d['executed'])}")
            else:
                print("No decision completed yet.")

        if (
            latest_decision
            and table_exists(c, "shadow_decision_evaluations")
        ):
            rows = c.execute("""
                SELECT evaluation_basis,
                       horizon_seconds,
                       evaluation_status,
                       selected_action,
                       selected_action_net_points,
                       best_counterfactual_action,
                       best_counterfactual_net_points,
                       buy_mfe_points,
                       buy_mae_points,
                       sell_mfe_points,
                       sell_mae_points
                FROM shadow_decision_evaluations
                WHERE decision_id=?
                ORDER BY evaluation_basis, horizon_seconds
            """, (latest_decision["decision_id"],)).fetchall()

            print("\n--- REAL MARKET OUTCOMES ---")

            if not rows:
                print("Waiting for 5m / 15m / 30m / 60m outcomes...")
            else:
                for e in rows:
                    mins = e["horizon_seconds"] // 60
                    net = e["selected_action_net_points"]
                    best = e["best_counterfactual_net_points"]

                    print(
                        f"{mins:>2}m | "
                        f"{e['evaluation_basis']:<18} | "
                        f"{e['evaluation_status']:<16} | "
                        f"Action={e['selected_action'] or '-':<4} | "
                        f"Net={net if net is not None else '-'} pts | "
                        f"Best={e['best_counterfactual_action'] or '-'} "
                        f"{best if best is not None else '-'} pts"
                    )

        con.close()

    except Exception as e:
        print("\nDATABASE READ ERROR:", e)

    print("\n" + "=" * 72)
    print("Refreshing every 10 seconds... Ctrl+C stops dashboard only.")
    time.sleep(10)

