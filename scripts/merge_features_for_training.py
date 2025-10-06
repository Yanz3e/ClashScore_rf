#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并特征：
- 读取 data/embedding_index.csv（由嵌入脚本生成）
- 逐个加载 mutant_embeddings/*.npy（1280 维）
- 与 data/filtered_abbind_single_mutations.csv 合并标签
- 输出 data/merged_features.csv / .parquet（可用于训练随机森林）

合并键：
  csv:  '#PDB'（转大写、去空格）, 'Mutation'（去空格）
  idx:  'pdb'（大写）,           'mutation'
"""

import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ====== 路径 ======
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_MUT = os.path.join(PROJ, "data", "filtered_abbind_single_mutations.csv")
IDX_CSV = os.path.join(PROJ, "data", "embedding_index.csv")
EMB_DIR = os.path.join(PROJ, "mutant_embeddings")
OUT_CSV = os.path.join(PROJ, "data", "merged_features.csv")
OUT_PARQ = os.path.join(PROJ, "data", "merged_features.parquet")
REPORT = os.path.join(PROJ, "results", "merge_report.txt")
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)

# ====== 读取索引（来自嵌入脚本）======
if not os.path.isfile(IDX_CSV):
    raise FileNotFoundError(f"没有找到索引：{IDX_CSV}")
idx = pd.read_csv(IDX_CSV)

# 规范化
idx["pdb"] = idx["pdb"].astype(str).str.upper().str.strip()
idx["mutation"] = idx["mutation"].astype(str).str.strip()

# ====== 读取原始突变表（取标签与元数据）======
if not os.path.isfile(CSV_MUT):
    raise FileNotFoundError(f"没有找到突变表：{CSV_MUT}")
df = pd.read_csv(CSV_MUT)

# 兼容列名（不同数据来源可能稍有差异）
col_ddg_candidates = [c for c in df.columns if c.lower().replace(" ", "").startswith("ddg")]
ddg_col = col_ddg_candidates[0] if col_ddg_candidates else None

# 一些可能有用的“第二指标”（任选其一或多选）
side_cols = []
for cand in [
    "PDB MolProbity clashscore",  # 结构质量/冲突分数
    "Assay pH",
    "Assay Temp (Celcius)",
    "PDB Res. (Angstroms)",
    "PDB R-value",
    "PDB R-free",
]:
    if cand in df.columns:
        side_cols.append(cand)

# 提取链（从 Mutation 里取冒号左侧的链 ID）
def parse_chain(mut):
    if isinstance(mut, str) and ":" in mut:
        return mut.split(":")[0].strip()
    return None

df["_pdb"] = df["#PDB"].astype(str).str.upper().str.strip()
df["_mutation"] = df["Mutation"].astype(str).str.strip()
df["_chain"] = df["Mutation"].apply(parse_chain)

keep_cols = ["_pdb", "_mutation", "_chain"]
label_cols = []
if ddg_col is not None:
    label_cols.append(ddg_col)
label_cols += side_cols

meta_cols = []
for cand in ["Assay Name", "PDB DOI", "Partners(A)", "Protein-1", "Protein-2"]:
    if cand in df.columns:
        meta_cols.append(cand)

df_labels = df[keep_cols + label_cols + meta_cols].drop_duplicates()

# ====== 载入向量并拼 DataFrame ======
def load_vec(path):
    v = np.load(path)
    if v.ndim != 1 or v.shape[0] != 1280:
        raise ValueError(f"向量形状异常：{path} -> {v.shape}")
    return v

emb_rows = []
missing_files = 0

for i, r in tqdm(idx.iterrows(), total=len(idx), desc="Loading embeddings"):
    f = os.path.join(EMB_DIR, r["embedding_file"])
    if not os.path.isfile(f):
        missing_files += 1
        continue
    vec = load_vec(f)
    rec = {
        "_pdb": r["pdb"],
        "_mutation": r["mutation"],
        "_chain": r["chain"],
        "length": r.get("length", np.nan),
        "embedding_file": r["embedding_file"],
    }
    # 展开 1280 维
    for k in range(1280):
        rec[f"emb_{k}"] = float(vec[k])
    emb_rows.append(rec)

emb_df = pd.DataFrame(emb_rows)
if emb_df.empty:
    raise RuntimeError("没有加载到任何 embedding（索引为空或文件缺失）。")

# ====== 合并（inner join，仅保留双方都有的数据）=====
merged = emb_df.merge(df_labels, on=["_pdb", "_mutation", "_chain"], how="inner")

# ====== 基本检查 ======
total_idx = len(idx)
total_emb = len(emb_df)
total_merged = len(merged)

# ====== 输出 ======
merged.to_csv(OUT_CSV, index=False)
# parquet（可选，如未装 pyarrow 会失败，忽略即可）
try:
    merged.to_parquet(OUT_PARQ, index=False)
    wrote_parquet = True
except Exception:
    wrote_parquet = False

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("==== Merge Report ====\n")
    f.write(f"Index entries           : {total_idx}\n")
    f.write(f"Embeddings loaded       : {total_emb}\n")
    f.write(f"Merged rows (trainable) : {total_merged}\n")
    f.write(f"Missing embedding files : {missing_files}\n")
    f.write(f"Label columns           : {', '.join(label_cols) if label_cols else '(none)'}\n")
    f.write(f"Side/meta columns       : {', '.join(side_cols + meta_cols) if (side_cols or meta_cols) else '(none)'}\n")
    f.write(f"Output CSV              : {OUT_CSV}\n")
    f.write(f"Output Parquet          : {OUT_PARQ if wrote_parquet else '(skipped)'}\n")

print("===== ✅ 合并完成 =====")
print(f"索引条目        : {total_idx}")
print(f"向量加载        : {total_emb}")
print(f"可训练样本数    : {total_merged}")
print(f"缺失向量文件数  : {missing_files}")
print(f"输出 CSV        : {OUT_CSV}")
print(f"输出 Parquet    : {OUT_PARQ if wrote_parquet else '(未写入)'}")
print(f"报告            : {REPORT}")

# 小提示：给训练脚本准备好特征名 & 标签名
# 这两行写到 results 下，方便下游读取
feat_cols = [f"emb_{k}" for k in range(1280)]
targets = []
if ddg_col is not None:
    targets.append(ddg_col)
# 如需用 clashscore 作第二指标，自动加入
if "PDB MolProbity clashscore" in merged.columns:
    targets.append("PDB MolProbity clashscore")

json.dump({"feature_cols": feat_cols, "target_cols": targets},
          open(os.path.join(PROJ, "results", "columns.json"), "w"),
          ensure_ascii=False, indent=2)
print("列信息已写入：", os.path.join(PROJ, "results", "columns.json"))
