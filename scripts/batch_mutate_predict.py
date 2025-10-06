#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量：读取 input/*.fasta
 -> 自动枚举单点突变（可限制region/氨基酸集合/抽样）
 -> ESM-1b 嵌入（mean-pool）
 -> 调你训练好的 RF 模型预测 (ddG & Clashscore)
 -> 输出到 output/<name>_pred.csv / _scatter.png / _pareto.csv / _pareto.png
"""

import os, re, glob, json, argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import torch

# --------- 项目路径 ----------
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR  = os.path.join(PROJ, "input")
OUTPUT_DIR = os.path.join(PROJ, "output")
MODEL_PATH = os.path.join(PROJ, "models", "rf_model.pkl")
COLS_JSON  = os.path.join(PROJ, "results", "columns.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- 常量 ----------
AA20 = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 1022  # ESM-1b 上限（不含CLS/EOS）

# --------- ESM-1b ----------
try:
    import esm
except ImportError:
    raise SystemExit("未找到 fair-esm，请先安装：pip install fair-esm")

def load_esm1b(device):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def embed_mean_batch(model, batch_converter, seq_list, device):
    data = [("seq", s) for s in seq_list]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]
    embs = []
    for i, s in enumerate(seq_list):
        rep = reps[i, 1:1+len(s), :].mean(0).cpu().numpy()
        embs.append(rep)
    return np.stack(embs, axis=0)  # (B,1280)

# --------- 突变器 ----------
def enumerate_single_mutations(seq, aa_set=AA20, region=None, max_mutants=None, seed=42):
    """返回 [{'pos':i,'wt':a,'mt':b,'mutation':'A12V'}...]（1-based）"""
    L = len(seq)
    s, e = 1, L
    if region:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", region)
        if not m: raise ValueError("--region 需形如 30-120（1-based）")
        s, e = int(m.group(1)), int(m.group(2))
        if s < 1 or e > L or s > e:
            raise ValueError(f"--region 越界：{s}-{e} for len={L}")
    outs = []
    for i in range(s, e+1):
        wt = seq[i-1]
        for b in aa_set:
            if b != wt:
                outs.append({"pos": i, "wt": wt, "mt": b, "mutation": f"{wt}{i}{b}"})
    if max_mutants and len(outs) > max_mutants:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(outs), size=max_mutants, replace=False)
        outs = [outs[i] for i in idx]
    return outs

def apply_single_mut(seq, wt, pos, mt):
    if pos < 1 or pos > len(seq):
        raise IndexError(f"pos越界: {pos} (len={len(seq)})")
    if seq[pos-1] != wt:
        raise ValueError(f"WT不匹配: 期望{wt}@{pos} 实际{seq[pos-1]}")
    if wt == mt:
        raise ValueError("wt==mt，无需突变")
    return seq[:pos-1] + mt + seq[pos:]

def pareto_mask_min2(x, y):
    """二维最小化的 Pareto 前沿（True=非支配）"""
    idx = np.argsort(x)
    best_y = np.inf
    mask = np.zeros_like(x, dtype=bool)
    for i in idx:
        if y[i] <= best_y:
            mask[i] = True
            best_y = y[i]
    return mask

# --------- 主流程 ----------
def process_one_fasta(path, model, targets, device, batch_size, aa_set, region, max_mutants):
    recs = list(SeqIO.parse(path, "fasta"))
    outputs = []
    base = os.path.splitext(os.path.basename(path))[0]

    for rec in recs:
        seq_id = rec.id
        wt_seq = str(rec.seq).upper().replace("*","")
        if len(wt_seq) > MAX_LEN:
            print(f"⚠️ {seq_id}: 长度 {len(wt_seq)} 超过 ESM-1b 上限 {MAX_LEN}，跳过")
            continue

        muts = enumerate_single_mutations(wt_seq, aa_set=aa_set, region=region, max_mutants=max_mutants)
        if not muts:
            print(f"⚠️ {seq_id}: 无可评估突变，跳过")
            continue

        # 生成突变序列
        mut_seqs, meta_rows = [], []
        for m in muts:
            try:
                ms = apply_single_mut(wt_seq, m["wt"], m["pos"], m["mt"])
            except Exception:
                continue
            mut_seqs.append(ms)
            meta_rows.append({"seq_id": seq_id, "pos": m["pos"], "wt": m["wt"], "mt": m["mt"], "mutation": m["mutation"]})

        if not mut_seqs:
            print(f"⚠️ {seq_id}: 突变序列为空，跳过")
            continue

        # ESM-1b 嵌入 + 预测
        esm_model, batch_converter = load_esm1b(device)
        preds = []
        for i in tqdm(range(0, len(mut_seqs), batch_size), desc=f"Embedding+Predict [{seq_id}]"):
            batch = mut_seqs[i:i+batch_size]
            embs = embed_mean_batch(esm_model, batch_converter, batch, device)
            y = model.predict(embs)
            preds.append(np.atleast_2d(y))
        Y = np.vstack(preds)

        out = pd.DataFrame(meta_rows)
        # 目标列名（尽量沿用训练时）
        if targets and len(targets) >= 2:
            ddg_name, clash_name = targets[0], targets[1]
        elif targets and len(targets) == 1:
            ddg_name, clash_name = targets[0], "clashscore"
        else:
            ddg_name, clash_name = "ddG(kcal/mol)", "PDB MolProbity clashscore"

        if Y.ndim == 1:
            out[f"pred_{ddg_name}"] = Y
            out[f"pred_{clash_name}"] = np.nan
        else:
            out[f"pred_{ddg_name}"] = Y[:, 0]
            out[f"pred_{clash_name}"] = Y[:, 1]

        # === 新增：预测 WT 并追加到 out，标记 is_wt ===
        try:
            wt_emb = embed_mean_batch(esm_model, batch_converter, [wt_seq], device)  # shape (1, F)
            wt_pred = model.predict(wt_emb)  # shape (1,) 或 (1,2)
            wt_pred = np.atleast_2d(wt_pred)[0]
            if wt_pred.size == 1:
                wt_ddg = float(wt_pred[0]); wt_clash = np.nan
            else:
                wt_ddg = float(wt_pred[0]); wt_clash = float(wt_pred[1])
        except Exception as e:
            print(f"⚠️ {seq_id}: 预测 WT 失败：{e}")
            wt_ddg, wt_clash = np.nan, np.nan

        out["is_wt"] = 0
        wt_row = {
            "seq_id": seq_id, "pos": np.nan, "wt": "", "mt": "",
            "mutation": "WT",
            f"pred_{ddg_name}": wt_ddg,
            f"pred_{clash_name}": wt_clash,
            "is_wt": 1
        }
        out = pd.concat([out, pd.DataFrame([wt_row])], ignore_index=True)

        # 保存 CSV（含 WT 行）
        csv_path = os.path.join(OUTPUT_DIR, f"{base}__{seq_id}_pred.csv")
        out.to_csv(csv_path, index=False)
        print(f"✅ 保存：{csv_path}  ({len(out)} 行, 含 WT)")

        # 散点：先画全部，再叠加 WT 红点
        x = out[f"pred_{ddg_name}"].values
        y = out[f"pred_{clash_name}"].values
        is_wt = (out["is_wt"] == 1).values

        plt.figure(figsize=(7,6))
        plt.scatter(x[~is_wt], y[~is_wt], s=15, alpha=0.75)
        plt.scatter(x[is_wt], y[is_wt], s=80, c="red", edgecolors="black", linewidths=0.7, zorder=5)
        plt.xlabel(ddg_name); plt.ylabel(clash_name)
        plt.title(f"{seq_id} predictions (WT in red)")
        plt.tight_layout()
        png = os.path.join(OUTPUT_DIR, f"{base}__{seq_id}_scatter_ddG_vs_clash.png")
        plt.savefig(png, dpi=180); plt.close()
        print(f"🖼️ 散点：{png}")

        # Pareto：正常算前沿，再叠加 WT 红点
        if np.all(~np.isnan(y)):
            mask = pareto_mask_min2(x, y)
            pareto_df = out.loc[mask].copy().sort_values([f"pred_{ddg_name}", f"pred_{clash_name}"])
            pcsv = os.path.join(OUTPUT_DIR, f"{base}__{seq_id}_pareto.csv")
            pareto_df.to_csv(pcsv, index=False)

            plt.figure(figsize=(7,6))
            plt.scatter(x, y, s=12, alpha=0.3, label="all")
            plt.scatter(pareto_df[f"pred_{ddg_name}"], pareto_df[f"pred_{clash_name}"],
                        s=24, alpha=0.95, label="Pareto", marker="o")
            # 叠加 WT
            plt.scatter(x[is_wt], y[is_wt], s=90, c="red", edgecolors="black", linewidths=0.7, zorder=6, label="WT")
            plt.xlabel(ddg_name); plt.ylabel(clash_name); plt.legend()
            plt.title(f"{seq_id} Pareto front")
            plt.tight_layout()
            ppng = os.path.join(OUTPUT_DIR, f"{base}__{seq_id}_pareto.png")
            plt.savefig(ppng, dpi=180); plt.close()
            print(f"🌈 Pareto：{ppng}")

        outputs.append(csv_path)
    return outputs


def main():
    ap = argparse.ArgumentParser(description="Batch mutate & predict from input/*.fasta")
    ap.add_argument("--aa-set", default=AA20, help="可替换的氨基酸集合（默认20种）")
    ap.add_argument("--region", default=None, help="限制突变区域，如 30-120（1-based，闭区间）")
    ap.add_argument("--max-mutants", type=int, default=None, help="最多评估的突变数（超出随机采样）")
    ap.add_argument("--batch-size", type=int, default=16, help="ESM 推理 batch size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 device:", device)

    # 模型 & 目标名
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型：{MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    targets = None
    if os.path.isfile(COLS_JSON):
        try:
            cfg = json.load(open(COLS_JSON, "r", encoding="utf-8"))
            targets = cfg.get("target_cols", None)
        except Exception:
            targets = None

    fasta_list = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fasta")))
    print(f"🔎 共发现 {len(fasta_list)} 个 FASTA")
    if not fasta_list:
        print(f"提示：请把 WT FASTA 放到 {INPUT_DIR}/ 下，再重试～")
        return

    for p in fasta_list:
        print(f"\n==> 处理 {os.path.basename(p)}")
        try:
            process_one_fasta(
                p, model, targets, device,
                batch_size=args.batch_size,
                aa_set=list(args.aa_set),
                region=args.region,
                max_mutants=args.max_mutants
            )
        except Exception as e:
            print(f"⚠️ 处理 {p} 失败：{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

