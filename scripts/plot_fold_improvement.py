#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_KCAL = 0.0019872041  # kcal/(mol·K)

# ---------------- Utils ----------------
def autodetect_cols(df, ddg_col=None, clash_col=None):
    cols = list(df.columns)
    if ddg_col is None:
        cand = [c for c in cols if c.lower().startswith("pred_") and "clash" not in c.lower()]
        if not cand:
            cand = [c for c in cols if ("ddg" in c.lower()) or (c.lower().endswith("gi"))]
        if not cand:
            raise ValueError("无法自动识别 ΔG 列，请用 --ddg-col 指定")
        ddg_col = cand[0]
    if clash_col is None:
        cand = [c for c in cols if "clash" in c.lower()]
        clash_col = cand[0] if cand else None
    return ddg_col, clash_col

def find_wt_mask(df):
    mask = np.zeros(len(df), dtype=bool)
    if "is_wt" in df.columns:
        mask |= (df["is_wt"] == 1)
    if "mutation" in df.columns:
        mask |= (df["mutation"].astype(str).str.upper() == "WT")
    if "pos" in df.columns:
        mask |= df["pos"].isna()
    return mask

def compute_folds(df, ddg_col, clash_col, tempK):
    out = df.copy()
    wt_mask = find_wt_mask(out)
    if not wt_mask.any():
        raise ValueError("没有找到 WT 行")
    RT = R_KCAL * float(tempK)
    dG_all = pd.to_numeric(out[ddg_col], errors="coerce")
    dG_wt  = dG_all[wt_mask].median(skipna=True)
    dddG = (dG_all - dG_wt).clip(-20, 20)
    affinity_improve = np.exp(-dddG / RT)
    affinity_improve.loc[wt_mask] = 1.0
    if clash_col is not None and clash_col in out.columns:
        clash_all = pd.to_numeric(out[clash_col], errors="coerce")
        clash_wt  = clash_all[wt_mask].median(skipna=True)
        eps = 1e-9
        structure_fold = clash_wt / clash_all.clip(lower=eps)
        structure_fold.loc[wt_mask] = 1.0
    else:
        structure_fold = pd.Series(np.nan, index=out.index)
    out["affinity_improve"] = affinity_improve
    out["structure_fold"]   = structure_fold
    return out

def pareto_mask(x, y):
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]: continue
        for j in range(n):
            if i==j: continue
            if x[j] >= x[i] and y[j] >= y[i] and (x[j]>x[i] or y[j]>y[i]):
                mask[i] = False
                break
    return mask

def plot_scatter(df, out_png, title=None, shrink_wt=True):
    x = pd.to_numeric(df["affinity_improve"], errors="coerce").values
    y = pd.to_numeric(df["structure_fold"],   errors="coerce").values
    wt_mask = find_wt_mask(df)

    plt.figure(figsize=(7,6))
    s_main = 28
    plt.scatter(x[~wt_mask], y[~wt_mask], s=s_main, alpha=0.8)
    if wt_mask.any():
        plt.scatter(x[wt_mask], y[wt_mask], s=max(12, int(s_main*0.6) if shrink_wt else s_main),
                    alpha=0.95, zorder=5, c="orange")
    plt.xscale("log"); plt.axvline(1.0, ls="--", lw=0.8, alpha=.6)
    plt.xlabel("Affinity improvement (×, >1 = better)")
    plt.ylabel("Structural rationality improvement (×)")
    if title: plt.title(title)
    plt.grid(True, linewidth=0.3, alpha=0.35)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_pareto(df, out_png, title=None):
    x = pd.to_numeric(df["affinity_improve"], errors="coerce").values
    y = pd.to_numeric(df["structure_fold"],   errors="coerce").values
    wt_mask = find_wt_mask(df)

    # Pareto mask
    mask = pareto_mask(x, y)
    pareto_df = df.loc[mask]

    plt.figure(figsize=(7,6))
    # Pareto 点：半透明粉红色
    plt.scatter(pareto_df["affinity_improve"], pareto_df["structure_fold"],
                s=40, alpha=0.6, c="pink", label="Pareto")
    # Pareto 点标注 mutation
    if "mutation" in pareto_df.columns:
        for _, row in pareto_df.iterrows():
            plt.text(row["affinity_improve"], row["structure_fold"],
                     str(row["mutation"]), fontsize=7, ha="left", va="bottom", alpha=0.8)

    # WT 点：半透明玫红色
    if wt_mask.any():
        plt.scatter(df.loc[wt_mask,"affinity_improve"], df.loc[wt_mask,"structure_fold"],
                    s=50, c="deeppink", alpha=0.6, zorder=5, label="WT")
        for _, row in df.loc[wt_mask].iterrows():
            plt.text(row["affinity_improve"], row["structure_fold"], "WT",
                     fontsize=8, ha="left", va="bottom", color="deeppink", alpha=0.9)

    plt.xscale("log"); plt.axvline(1.0, ls="--", lw=0.8, alpha=.6)
    plt.xlabel("Affinity improvement (×, >1 = better)")
    plt.ylabel("Structural rationality improvement (×)")
    if title: plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.3, alpha=0.35)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot fold improvements (scatter + Pareto+WT).")
    ap.add_argument("--csv", required=True, help="输入的 __pred.csv")
    ap.add_argument("--outdir", default=None, help="输出目录，默认与 csv 同目录")
    ap.add_argument("--ddg-col", default=None, help="ΔG 列名")
    ap.add_argument("--clash-col", default=None, help="clash 列名")
    ap.add_argument("--tempK", type=float, default=298.15)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ddg_col, clash_col = autodetect_cols(df, args.ddg_col, args.clash_col)
    out = compute_folds(df, ddg_col, clash_col, args.tempK)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]

    out_csv = os.path.join(outdir, f"{base}_with_folds.csv")
    out.to_csv(out_csv, index=False)

    out_png1 = os.path.join(outdir, f"{base}_fold_scatter.png")
    plot_scatter(out, out_png1, title=f"{base} (fold improvements)")

    out_png2 = os.path.join(outdir, f"{base}_pareto_fold.png")
    plot_pareto(out, out_png2, title=f"{base} (Pareto front + WT)")

    print(f"✅ 保存表格: {out_csv}")
    print(f"🖼️ 散点图: {out_png1}")
    print(f"🌈 Pareto图: {out_png2}")

if __name__ == "__main__":
    main()
