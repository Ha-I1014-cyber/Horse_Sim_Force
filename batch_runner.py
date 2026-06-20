"""
batch_runner.py

Horse_Sim_Force を、アニメーションを伴わずに多数回実行するための土台です。
物理モデル本体(RaceSimulation)には一切手を加えず、race_core.py から
そのまま読み込みます。1レースを1回の関数呼び出しとして扱い、その結果を
馬ごとの構造化データ(行)として取り出し、CSVとして書き出します。

最初のリサーチ案(枠順バイアスの定量評価)は、この出力をそのまま使えます。
equalize_ability を有効にすると、全馬の能力パラメータを基準値にそろえ、
枠順(発走時の横位置)とコース形状に由来する差だけを残した実験になります。

実行例:
    python batch_runner.py --races 500 --equalize --jobs 4
"""

import io
import time
import contextlib
import argparse

import numpy as np
import pandas as pd

# 画面描画を完全に無効化してから物理本体を読み込む
import matplotlib
matplotlib.use("Agg")

import race_core as rc
from race_core import (
    RaceSimulation,
    N,
    SPEED_START,
    SPEED_CRUISE,
    SPEED_SPURT,
)


@contextlib.contextmanager
def _silence():
    """RaceSimulation内部のprint(設定表示・ハロンタイム)を捨てる"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def run_single_race(seed=None, equalize_ability=False, max_steps=None):
    """
    1レースをアニメーションなしで最後まで走らせ、馬ごとの結果を辞書のリストで返す。

    seed             : 乱数シード。指定すると再現可能になる。
    equalize_ability : Trueにすると全馬の能力パラメータ(スタートダッシュ・各目標速度)を
                       基準値にそろえ、枠順とコース形状に由来する差だけを残す。
    max_steps        : 打ち切りステップ数。Noneならrace_core.STEPSを使う。
    """
    if seed is not None:
        np.random.seed(seed)
    steps = rc.STEPS if max_steps is None else max_steps

    with _silence():
        sim = RaceSimulation(N)

    # 枠順(馬番)と発走時の横位置を、レース開始前に記録しておく。
    # index 0 が最内(ラチ側)、index N-1 が最外で、馬番は index+1 とする。
    init_lane = sim.pos[:, 1].copy()

    if equalize_ability:
        sim.start_dash[:] = 1.0
        sim.speed_start[:] = SPEED_START
        sim.speed_cruise[:] = SPEED_CRUISE
        sim.speed_spurt[:] = SPEED_SPURT

    # --- メインループ(描画なし) ---
    with _silence():
        for _ in range(steps):
            sim.update()
            if all(t is not None for t in sim.finish_times):
                break

    # --- 着順の確定 ---
    # 完走馬はゴールタイム順、未完走馬は到達距離が大きい順で後方に並べる。
    final_dist = sim.pos[:, 0] - sim.start_s_log
    keyed = []
    for i in range(N):
        t = sim.finish_times[i]
        # 完走:(0, time)、未完走:(1, -到達距離)。完走を優先しつつ距離で序列化する。
        key = (0, t) if t is not None else (1, -float(final_dist[i]))
        keyed.append((key, i))
    keyed.sort(key=lambda x: x[0])
    rank_of = {idx: rank + 1 for rank, (_, idx) in enumerate(keyed)}

    def corner(c, i):
        if sim.corner_passed[c] and sim.corner_orders[c] is not None:
            return int(sim.corner_orders[c][i])
        return -1  # 未通過は-1

    records = []
    for i in range(N):
        t = sim.finish_times[i]
        records.append({
            "seed": seed if seed is not None else -1,
            "umaban": i + 1,                              # 馬番(1=最内)
            "init_lane": round(float(init_lane[i]), 3),    # 発走時の横位置(m)
            "finished": int(t is not None),
            "finish_rank": rank_of[i],
            "finish_time": round(float(t), 3) if t is not None else None,
            "last3f": round(float(sim.last3f_times[i]), 3) if sim.last3f_times[i] is not None else None,
            "c1": corner(1, i),
            "c2": corner(2, i),
            "c3": corner(3, i),
            "c4": corner(4, i),
            "final_dist": round(float(final_dist[i]), 2),
        })
    return records


def run_batch(n_races, equalize_ability=False, seed0=0, max_steps=None, progress_every=10):
    """直列で n_races レースを回す"""
    rows = []
    for k in range(n_races):
        rows.extend(run_single_race(seed=seed0 + k,
                                    equalize_ability=equalize_ability,
                                    max_steps=max_steps))
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  ... {k + 1}/{n_races} races done", flush=True)
    return rows


def _worker(args):
    seed, equalize_ability, max_steps = args
    return run_single_race(seed=seed, equalize_ability=equalize_ability, max_steps=max_steps)


def run_batch_parallel(n_races, equalize_ability=False, seed0=0, max_steps=None, n_jobs=4):
    """複数プロセスに分散して n_races レースを回す。各レースは独立なので素直に並列化できる。"""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(seed0 + k, equalize_ability, max_steps) for k in range(n_races)]
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for rec in ex.map(_worker, tasks):
            rows.extend(rec)
            done += 1
            if done % 10 == 0:
                print(f"  ... {done}/{n_races} races done", flush=True)
    return rows


def summarize_gate_bias(df):
    """馬番(枠順)ごとに、平均着順・勝率・複勝率(3着以内)・未完走率・平均上り3Fを集計する"""
    df = df.copy()
    df["is_win"] = (df["finish_rank"] == 1).astype(int)
    df["is_top3"] = (df["finish_rank"] <= 3).astype(int)
    df["is_dnf"] = (df["finished"] == 0).astype(int)
    g = df.groupby("umaban")
    summary = g.agg(
        races=("umaban", "size"),
        mean_rank=("finish_rank", "mean"),
        win_rate=("is_win", "mean"),
        top3_rate=("is_top3", "mean"),
        dnf_rate=("is_dnf", "mean"),
        mean_last3f=("last3f", "mean"),
    ).reset_index()
    return summary.round({"mean_rank": 3, "win_rate": 3, "top3_rate": 3,
                          "dnf_rate": 3, "mean_last3f": 2})


def main():
    p = argparse.ArgumentParser(description="Horse_Sim_Force headless batch runner")
    p.add_argument("--races", type=int, default=200, help="走らせるレース数")
    p.add_argument("--equalize", action="store_true",
                   help="全馬の能力を基準値にそろえ、枠順効果だけを残す")
    p.add_argument("--seed0", type=int, default=0, help="先頭レースの乱数シード")
    p.add_argument("--max-steps", type=int, default=None, help="1レースの打ち切りステップ数")
    p.add_argument("--jobs", type=int, default=1, help="並列プロセス数(2以上で並列実行)")
    p.add_argument("--out", type=str, default="race_results.csv", help="馬ごとの結果CSV")
    p.add_argument("--summary", type=str, default="gate_bias_summary.csv", help="枠順集計CSV")
    args = p.parse_args()

    print(f"=== Batch start: races={args.races}, equalize={args.equalize}, jobs={args.jobs} ===",
          flush=True)
    t0 = time.time()
    if args.jobs and args.jobs > 1:
        rows = run_batch_parallel(args.races, equalize_ability=args.equalize,
                                  seed0=args.seed0, max_steps=args.max_steps, n_jobs=args.jobs)
    else:
        rows = run_batch(args.races, equalize_ability=args.equalize,
                         seed0=args.seed0, max_steps=args.max_steps)
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = summarize_gate_bias(df)
    summary.to_csv(args.summary, index=False)

    print(f"\n=== Done in {elapsed:.1f}s ({elapsed / args.races:.2f}s/race) ===")
    print(f"per-horse rows : {len(df)}  ->  {args.out}")
    print(f"gate summary   : {args.summary}\n")
    with pd.option_context("display.width", 120):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
