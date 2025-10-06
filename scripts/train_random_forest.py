#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV   = os.path.join(PROJ, "data", "merged_features.csv")
COLS_JSON  = os.path.join(PROJ, "results", "columns.json")
MODEL_OUT  = os.path.join(PROJ, "models", "rf_model.pkl")
PRED_CSV   = os.path.join(PROJ, "results", "rf_predictions.csv")
IMP_CSV    = os.path.join(PROJ, "results", "rf_feature_importances.csv")
IMP_PNG    = os.path.join(PROJ, "results", "rf_feature_importances.png")
CV_REPORT  = os.path.join(PROJ, "results", "rf_cv_report.txt")

os.makedirs(os.path.join(PROJ, "models"), exist_ok=True)
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)

def load_columns():
    """优先从 columns.json 读取；否则自动探测。"""
    if os.path.isfile(COLS_JSON):
        cfg = json.load(open(COLS_JSON, "r", encoding="utf-8"))
        feat_cols = cfg.get("feature_cols", [])
        target_cols = cfg.get("target_cols", [])
    else:
        df = pd.read_csv(DATA_CSV, nrows=5)
        feat_cols = [c for c in df.columns if c.startswith("emb_")]
        # 自动找 ddG & Clashscore
        target_cols = []
        # ddG
        for c in df.columns:
            if c.lower().replace(" ", "").startswith("ddg"):
                target_cols.append(c)
                break
        # Clashscore
        if "PDB MolProbity clashscore" in df.columns:
            target_cols.append("PDB MolProbity clashscore")
        else:
            # 兜底找 clash 字样
            for c in df.columns:
                if "clash" in c.lower():
                    target_cols.append(c)
                    break
    if not feat_cols:
        raise RuntimeError("找不到特征列（emb_*）。")
    if len(target_cols) == 0:
        raise RuntimeError("找不到目标列（ddG / Clashscore）。")
    # 若只找到一个目标，仍可单输出训练；找到两个就双输出
    return feat_cols, target_cols

def make_rf(n_estimators=700, max_depth=None, random_state=42, n_jobs=-1):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )

def compute_metrics(y_true, y_pred, names):
    if y_true.ndim == 1: 
        y_true = y_true.reshape(-1,1)
    if y_pred.ndim == 1: 
        y_pred = y_pred.reshape(-1,1)
    out = []
    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        mse = mean_squared_error(yt, yp)        # 不用 squared=False
        rmse = np.sqrt(mse)                     # 自己开根号
        out.append({
            "name": name,
            "R2": r2_score(yt, yp),
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": rmse,
        })
    return out


def plot_importance(importances, feat_cols, topk=30, out_png=IMP_PNG):
    idx = np.argsort(importances)[::-1][:topk]
    names = [feat_cols[i] for i in idx]
    vals  = importances[idx]
    plt.figure(figsize=(8, 10))
    y = np.arange(len(names))
    plt.barh(y, vals)
    plt.gca().invert_yaxis()
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Importance")
    plt.title(f"Random Forest Feature Importances (top {topk})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def aggregate_duplicates(df, feat_cols, target_cols):
    """将同一 (_pdb, _mutation, _chain) 的重复记录做均值聚合（标签），embedding取first。"""
    keys = ["_pdb", "_mutation", "_chain"]
    agg = {c: "first" for c in feat_cols + ["embedding_file", "length"] if c in df.columns}
    for c in target_cols:
        agg[c] = "mean"
    keep_meta = [c for c in ["Assay Name", "PDB DOI", "Partners(A)", "Protein-1", "Protein-2"] if c in df.columns]
    for c in keep_meta:
        agg[c] = "first"
    return df.groupby(keys, as_index=False).agg(agg)

def main(args):
    feat_cols, target_cols = load_columns()
    df = pd.read_csv(DATA_CSV)

    if args.aggregate:
        df = aggregate_duplicates(df, feat_cols, target_cols)

    X = df[feat_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    task_names = target_cols[:]  # 用于指标展示

    # 拆分训练/测试
    meta_cols = [c for c in ["_pdb","_mutation","_chain"] if c in df.columns]
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
    X, y, df[meta_cols], test_size=0.2, random_state=args.seed, shuffle=True
    )

    # Pipeline: 可选 PCA -> RF
    steps = []
    if args.pca_dim is not None and args.pca_dim > 0:
        steps.append(("pca", PCA(n_components=args.pca_dim, random_state=args.seed)))
    steps.append(("rf", make_rf(args.trees, args.max_depth, args.seed, args.jobs)))
    base = Pipeline(steps)

    multi_target = (y.shape[1] > 1)
    model = MultiOutputRegressor(base, n_jobs=args.jobs) if multi_target else base

    # 交叉验证
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_lines = []
    for fold, (tr, va) in enumerate(kf.split(X_tr), 1):
        mdl = MultiOutputRegressor(Pipeline(steps), n_jobs=args.jobs) if multi_target else Pipeline(steps)
        mdl.fit(X_tr[tr], y_tr[tr])
        pred = mdl.predict(X_tr[va])
        res = compute_metrics(y_tr[va], pred, task_names)
        line = "Fold {}: ".format(fold) + " | ".join(
            [f"{r['name']} R2={r['R2']:.3f} MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f}" for r in res]
        )
        print(line)
        cv_lines.append(line)

    # 全训练集拟合 & 测试集评估
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    test_res = compute_metrics(y_te, y_hat, task_names)
    print("\n=== Test ===")
    for r in test_res:
        print(f"{r['name']:>25s}  R2={r['R2']:.3f}  MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}")

    # 保存模型
    joblib.dump(model, MODEL_OUT)

    # 保存预测
    pred_df = meta_te.copy()
    for i, name in enumerate(task_names):
        pred_df[f"y_true_{name}"] = y_te[:, i]
        pred_df[f"y_pred_{name}"] = y_hat[:, i]
    pred_df.to_csv(PRED_CSV, index=False)

    # 提取/保存特征重要性（注意：Pipeline + MultiOutput）
    # 拿到最终 RF 的 feature_importances_；如果做了 PCA，要把重要性映射回原始特征较复杂，这里仅在未使用 PCA 时输出。
    if args.pca_dim in (None, 0):
        if multi_target:
            importances = np.mean(
                [est.named_steps["rf"].feature_importances_ for est in model.estimators_],
                axis=0
            )
        else:
            importances = model.named_steps["rf"].feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
        imp_df.sort_values("importance", ascending=False).to_csv(IMP_CSV, index=False)
        plot_importance(importances, feat_cols, topk=30, out_png=IMP_PNG)
    else:
        # 使用了 PCA：输出 PCA 维度的重要性（来自 RF 对 PCA 特征的权重）
        if multi_target:
            importances = np.mean(
                [est.named_steps["rf"].feature_importances_ for est in model.estimators_],
                axis=0
            )
        else:
            importances = model.named_steps["rf"].feature_importances_
        pca_imp_csv = os.path.join(PROJ, "results", "rf_feature_importances_pca_space.csv")
        pd.DataFrame({
            "pca_feature": [f"pc_{i}" for i in range(len(importances))],
            "importance": importances
        }).sort_values("importance", ascending=False).to_csv(pca_imp_csv, index=False)

    # 写交叉验证报告
    with open(CV_REPORT, "w", encoding="utf-8") as f:
        f.write("==== Cross-Validation ====\n")
        for line in cv_lines:
            f.write(line + "\n")
        f.write("\n==== Test ====\n")
        for r in test_res:
            f.write(f"{r['name']}  R2={r['R2']:.4f}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}\n")
        f.write(f"\nModel: {MODEL_OUT}\nPredictions: {PRED_CSV}\n")

    print("\n===== ✅ 训练完成 =====")
    print(f"模型保存：{MODEL_OUT}")
    print(f"预测保存：{PRED_CSV}")
    if args.pca_dim in (None, 0):
        print(f"重要性CSV：{IMP_CSV}")
        print(f"重要性图：{IMP_PNG}")
    else:
        print("使用了 PCA，已输出 PCA 空间的重要性 CSV（results/rf_feature_importances_pca_space.csv）")
    print(f"CV报告：{CV_REPORT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PCA + Random Forest for multi-output regression (ddG & Clashscore)")
    ap.add_argument("--trees", type=int, default=700, help="n_estimators")
    ap.add_argument("--max-depth", type=int, default=None, help="max_depth")
    ap.add_argument("--folds", type=int, default=5, help="KFold splits")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel jobs")
    ap.add_argument("--aggregate", action="store_true", help="对重复突变按均值聚合标签")
    ap.add_argument("--pca-dim", type=int, default=100, help="PCA 维度；设为 0/None 关闭 PCA")
    args = ap.parse_args()
    if args.pca_dim == 0:
        args.pca_dim = None
    main(args)
