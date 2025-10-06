#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, json, numpy as np, pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJ, "data", "filtered_abbind_single_mutations.csv")
FASTA_PATH = os.path.join(PROJ, "data", "wt_antibodies.fasta")
MAP_DIR = os.path.join(PROJ, "data", "residue_index_maps")
OUT_DIR = os.path.join(PROJ, "mutant_embeddings")
INDEX_OUT = os.path.join(PROJ, "data", "embedding_index.csv")
LOG_PATH = os.path.join(PROJ, "results", "embedding_log.jsonl")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 device:", device)

try:
    import esm
except ImportError:
    raise SystemExit("请先安装 fair-esm: pip install fair-esm")

# 载入 ESM-1b
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# 允许插入码：H:Y100aF / L:D27bA / A:Q221A 等（icode 可有可无，大小写皆可）
MUT_RE = re.compile(r"^([A-Za-z])\s*:\s*([A-Za-z])(\d+)([A-Za-z]?)([A-Za-z])$")
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN = 1022  # 你的 FASTA 已经过滤 >1022，这里只是双保险

def parse_mutation(mut):
    m = MUT_RE.match(mut.strip())
    if not m:
        raise ValueError(f"无法解析突变: {mut}")
    chain, wt, num, icode, mt = m.groups()
    wt, mt = wt.upper(), mt.upper()
    icode = icode.upper()
    if wt not in AA1 or mt not in AA1: raise ValueError(f"非标准AA: {mut}")
    return chain.strip(), wt, int(num), icode, mt

def load_map(pdb, chain):
    p = os.path.join(MAP_DIR, f"{pdb}_{chain}.json")
    if not os.path.isfile(p): raise FileNotFoundError(f"缺映射: {p}")
    return json.load(open(p))  # 键 "num|icode", icode为空用 ""

def embed_mean(seq):
    if len(seq) > MAX_LEN:
        raise RuntimeError(f"sequence too long after filter: {len(seq)}")
    data = [("s", seq)]
    _, _, toks = batch_converter(data); toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33][0, 1:1+len(seq), :]
    return rep.mean(0).cpu().numpy()

def log_err(**k):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(k, ensure_ascii=False) + "\n")

# 载入数据
df = pd.read_csv(CSV_PATH)
seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(FASTA_PATH, "fasta")}

ok = fail = 0
rows = []

for _, row in tqdm(df.dropna(subset=["#PDB","Mutation"]).iterrows(),
                   total=len(df.dropna(subset=["#PDB","Mutation"]))):
    pdb = str(row["#PDB"]).upper().strip()
    mut = str(row["Mutation"]).strip()
    try:
        chain, wt, num, icode, mt = parse_mutation(mut)
        fasta_id = f"{pdb}_{chain}"
        if fasta_id not in seqs:
            raise FileNotFoundError(f"FASTA缺少序列: {fasta_id}")

        num2idx = load_map(pdb, chain)  # 例如 "100|A"、"487|"
        key = f"{num}|{icode}" if icode else f"{num}|"
        if key not in num2idx:
            # 容错：如果带icode没命中，尝试不带icode键
            alt = f"{num}|"
            if alt in num2idx: key = alt
            else: raise KeyError(f"PDB编号不存在于映射: {key}")

        idx = int(num2idx[key])
        wt_seq = seqs[fasta_id]
        if idx < 0 or idx >= len(wt_seq):
            raise IndexError(f"索引越界 idx={idx} len={len(wt_seq)}")
        if wt_seq[idx] != wt:
            raise ValueError(f"野生型不匹配: 期望 {wt}@{num}{icode or ''}, 实际 {wt_seq[idx]}")

        mut_seq = wt_seq[:idx] + mt + wt_seq[idx+1:]
        if len(mut_seq) > MAX_LEN:
            raise RuntimeError(f"mut seq too long: {len(mut_seq)}")

        emb = embed_mean(mut_seq)
        out_name = f"{pdb}_{chain}_{wt}{num}{icode or ''}{mt}.npy"
        np.save(os.path.join(OUT_DIR, out_name), emb)
        rows.append({"pdb": pdb, "chain": chain, "mutation": mut,
                     "embedding_file": out_name, "length": len(mut_seq)})
        ok += 1
    except Exception as e:
        fail += 1
        log_err(pdb=pdb, mutation=mut, error=f"{type(e).__name__}: {e}")
        print(f"⚠️ {pdb}::{mut} 失败: {type(e).__name__}: {e}")

pd.DataFrame(rows).to_csv(INDEX_OUT, index=False)
print(f"✅ 完成：成功 {ok} 条，失败 {fail} 条")
print(f"📄 索引：{INDEX_OUT}")
print(f"📂 向量：{OUT_DIR}")
print(f"📝 日志：{LOG_PATH}")
