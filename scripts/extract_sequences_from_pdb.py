import os
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd

# ====== 配置路径 ======
PDB_DIR = "../structures"  # 放 .pdb 文件的路径（你可以改为你存放结构的路径）
CSV_PATH = "../data/filtered_abbind_single_mutations.csv"
OUTPUT_FASTA = "../data/wt_antibodies.fasta"

# ====== 读取突变表，提取需要的链信息 ======
df = pd.read_csv(CSV_PATH)
chains_needed = set()

for mut in df["Mutation"].dropna():
    try:
        pdbid = str(df.loc[df["Mutation"] == mut, "#PDB"].values[0])
        chain = mut.split(":")[0]
        chains_needed.add((pdbid.upper(), chain))
    except Exception as e:
        print(f"⚠️ 跳过异常突变: {mut} - {e}")

# ====== 提取并写入FASTA ======
parser = PDBParser(QUIET=True)
records = []

for pdbid, chain_id in chains_needed:
    pdb_path = os.path.join(PDB_DIR, f"{pdbid}.pdb")
    if not os.path.isfile(pdb_path):
        print(f"❌ 文件不存在: {pdb_path}")
        continue

    try:
        structure = parser.get_structure(pdbid, pdb_path)
        model = structure[0]
        chain = model[chain_id]
        seq = "".join([seq1(res.get_resname()) for res in chain if res.id[0] == ' '])
        record = SeqRecord(Seq(seq), id=f"{pdbid}_{chain_id}", description="")
        records.append(record)
    except Exception as e:
        print(f"⚠️ 读取 {pdbid} 链 {chain_id} 失败: {e}")

# 写入FASTA
SeqIO.write(records, OUTPUT_FASTA, "fasta")
print(f"✅ 已提取 {len(records)} 条序列到 {OUTPUT_FASTA}")
