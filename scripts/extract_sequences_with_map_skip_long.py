#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 PDB 提取指定链的氨基酸序列（标准残基、有 CA 原子），
同时生成 PDB 残基编号(含插入码) → 序列索引(0-based) 的映射表。
若链长度 > ESM-1b 上限 (1022)，直接跳过并记录原因。

输入:
- data/filtered_abbind_single_mutations.csv   (需含列: #PDB, Mutation)
- structures/*.pdb                            (每个 PDB 一个文件)

输出:
- data/wt_antibodies.fasta
- data/residue_index_maps/<PDB>_<CHAIN>.json  (键: "resseq|icode"，例 "100|A" or "487|")
- results/extract_report.csv                  (保留/跳过明细)
"""

import os
import json
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# ================== 配置 ==================
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJ, "data", "filtered_abbind_single_mutations.csv")
PDB_DIR  = os.path.join(PROJ, "structures")
OUT_FASTA = os.path.join(PROJ, "data", "wt_antibodies.fasta")
MAP_DIR   = os.path.join(PROJ, "data", "residue_index_maps")
REPORT_CSV = os.path.join(PROJ, "results", "extract_report.csv")
MAX_LEN = 1022   # ESM-1b 单次可安全处理的最大长度（去掉CLS/EOS后）

os.makedirs(os.path.join(PROJ, "data"), exist_ok=True)
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

# ================== 读取需要的 PDB+链 ==================
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"找不到突变CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
if "#PDB" not in df.columns or "Mutation" not in df.columns:
    raise KeyError("CSV 中需要包含列：#PDB 和 Mutation")

pairs = set()
for pdb, mut in zip(df["#PDB"], df["Mutation"]):
    if not isinstance(mut, str):
        continue
    pdbid = str(pdb).upper().strip()
    chain = mut.split(":")[0].strip()
    if pdbid and chain:
        pairs.add((pdbid, chain))

if not pairs:
    raise RuntimeError("没有从 CSV 中解析到任何 PDB+链；请检查 CSV 内容。")

# ================== 提取序列 & 生成映射 ==================
parser = PDBParser(QUIET=True)
records = []
report_rows = []

kept, skipped = 0, 0

for pdbid, chain_id in sorted(pairs):
    pdb_path = os.path.join(PDB_DIR, f"{pdbid}.pdb")
    status = "kept"
    reason = ""
    length = 0

    if not os.path.isfile(pdb_path):
        status, reason = "skipped", f"missing_pdb:{pdb_path}"
        length = 0
        report_rows.append({
            "pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": length
        })
        skipped += 1
        continue

    try:
        structure = parser.get_structure(pdbid, pdb_path)
        if 0 not in structure:
            status, reason = "skipped", "no_model_0"
            report_rows.append({"pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": 0})
            skipped += 1
            continue

        model = structure[0]
        if chain_id not in model:
            status, reason = "skipped", "chain_not_found"
            report_rows.append({"pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": 0})
            skipped += 1
            continue

        chain = model[chain_id]

        seq_chars = []
        num2idx = {}  # 键: "resseq|icode" (icode 为空时用 "")
        idx = 0

        for res in chain:
            hetfield, resseq, icode = res.id  # (hetero flag, residue sequence number, insertion code)
            # 只保留标准氨基酸并确保有 CA 原子（过滤水分子、配体、缺失/不完整残基）
            if hetfield == " " and "CA" in res:
                aa1 = seq1(res.get_resname())
                # 遇到未知/非标准残基名时，seq1 可能返回 "X"；这里直接保留X或跳过都可以
                # 为了不破坏索引，我们保留它
                seq_chars.append(aa1)
                key = f"{int(resseq)}|{(icode or '').strip()}"
                num2idx[key] = idx
                idx += 1

        seq_str = "".join(seq_chars)
        length = len(seq_str)

        # 跳过过长的链（> MAX_LEN）
        if length > MAX_LEN:
            status, reason = "skipped", f"too_long:{length}>{MAX_LEN}"
            report_rows.append({"pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": length})
            skipped += 1
            continue

        # 写入 FASTA & 映射
        rec_id = f"{pdbid}_{chain_id}"
        records.append(SeqRecord(Seq(seq_str), id=rec_id, description=""))
        with open(os.path.join(MAP_DIR, rec_id + ".json"), "w") as f:
            json.dump(num2idx, f)

        kept += 1
        report_rows.append({"pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": length})

    except Exception as e:
        status, reason = "skipped", f"parse_error:{type(e).__name__}:{e}"
        report_rows.append({"pdb": pdbid, "chain": chain_id, "status": status, "reason": reason, "length": 0})
        skipped += 1

# 写 FASTA
if records:
    SeqIO.write(records, OUT_FASTA, "fasta")
else:
    # 即便没有保留链，也给一个空文件，避免后续脚本找不到
    open(OUT_FASTA, "w").close()

# 写报表
pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)

print("==============================================")
print(f"✅ 提取完成：保留 {kept} 条，跳过 {skipped} 条")
print(f"📄 FASTA： {OUT_FASTA}")
print(f"🗺️ 映射： {MAP_DIR}/*.json")
print(f"🧾 报表： {REPORT_CSV}")
print("==============================================")
