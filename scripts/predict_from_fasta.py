#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量预测脚本：读取 input/*.fasta，提取 embedding，
用随机森林模型预测 ddG 和 clashscore，输出到 output/
"""

import os
import glob
import joblib
import torch
import esm
import numpy as np
import pandas as pd
from Bio import SeqIO

# ==== 路径配置 ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 加载模型 ====
print("📦 加载随机森林模型 ...")
model = joblib.load(MODEL_PATH)

# ==== 加载 ESM 模型 ====
print("📦 加载 ESM-1b 模型 ...")
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
if torch.cuda.is_available():
    esm_model = esm_model.cuda()


def get_embedding(sequence: str):
    """提取序列的平均池化 embedding"""
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    emb = token_representations.mean(1).cpu().numpy()
    return emb


# ==== 扫描 fasta ====
fasta_files = glob.glob(os.path.join(INPUT_DIR, "*.fasta"))
print(f"🔍 找到 {len(fasta_files)} 个 fasta 文件")

all_results = []

for fasta in fasta_files:
    for record in SeqIO.parse(fasta, "fasta"):
        seq_id = record.id
        seq = str(record.seq)
        print(f"⚡ 正在处理 {seq_id} ({len(seq)} aa)")

        # 生成 embedding
        emb = get_embedding(seq)

        # 预测
        pred = model.predict(emb)
        ddg, clash = pred[0]

        all_results.append({
            "seq_id": seq_id,
            "ddG": ddg,
            "clashscore": clash
        })

# ==== 保存结果 ====
df = pd.DataFrame(all_results)
out_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
df.to_csv(out_csv, index=False)
print(f"✅ 预测完成，结果已保存到 {out_csv}")

print("\n下次可以直接用 pandas + matplotlib 画散点图：")
print("  X=ddG, Y=clashscore")
