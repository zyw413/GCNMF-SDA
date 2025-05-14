import os
import json
import requests
import pandas as pd
import numpy as np
from subprocess import run

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import ttest_ind, ranksums
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + r"C:\Users\Mr Zhang\Desktop\gdc-client"
# 1. 查询 GDC/files 端点获取文件信息
def query_files(project):
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}}
            ]
        },
        "fields": "file_id,file_name,cases.samples.sample_id,cases.samples.sample_type",
        "format": "JSON",
        "size": 10000
    }
    r = requests.post("https://api.gdc.cancer.gov/files", json=payload)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    info = []
    for h in hits:
        fid = h["file_id"]
        fn = h["file_name"]
        sid = h["cases"][0]["samples"][0]["sample_id"]
        st = h["cases"][0]["samples"][0]["sample_type"]
        info.append((fid, fn, sid, st))
    return info


# 2. （可选）生成清单并通过 gdc-client 下载
def download_manifest(info, project):
    out = f"GDCdata/{project}"
    os.makedirs(out, exist_ok=True)
    manifest = os.path.join(out, f"{project}_manifest.txt")
    with open(manifest, "w") as mf:
        mf.write("id\n" + "\n".join([fid for fid, _, _, _ in info]))
    run(["gdc-client", "download", "-m", manifest, "--dir", out])


# 3. 构建基因×样本计数矩阵（不进行过滤）
def build_matrix(info, project):
    base = f"GDCdata/{project}"
    dfs, barcodes, types = [], [], {}
    for fid, fn, sid, st in info:
        path = os.path.join(base, fid, fn)
        if not os.path.exists(path):
            print(f"Warning: Missing {path}, skipped.")
            continue
        df = pd.read_csv(path, sep="\t", comment='#')

        # 确保使用不带版本的 gene_id 作为索引
        df['gene_id_clean'] = df['gene_id'].str.split('.').str[0]
        count_col = 'unstranded' if 'unstranded' in df.columns else df.columns[1]
        df2 = df[['gene_id_clean', count_col]].rename(columns={'gene_id_clean': 'gene_name', count_col: 'count'})
        df2 = df2.set_index('gene_name')

        dfs.append(df2['count'])
        barcodes.append(sid)
        types[sid] = st

    if not dfs:
        raise RuntimeError("No count files found under " + base)

    mat = pd.concat(dfs, axis=1)
    mat.columns = barcodes
    return mat, types


def deg_analysis_wilcoxon(mat, types):
    grp = pd.Series({s: ('Tumor' if 'Tumor' in types[s] else 'Normal') for s in mat.columns})
    counts = grp.value_counts()
    print(f"Sample counts:\n{counts}")
    # if counts.get('Tumor', 0) < 2 or counts.get('Normal', 0) < 2:
    #     print("Insufficient samples.")
    #     return None

    t_idx = grp[grp == 'Tumor'].index
    n_idx = grp[grp == 'Normal'].index
    ps, fcs = [], []
    for gene, row in mat.iterrows():
        _, p = ranksums(row[t_idx], row[n_idx])
        fc = np.log2(row[t_idx].mean() + 1) - np.log2(row[n_idx].mean() + 1)
        ps.append(p)
        fcs.append(fc)

    res = pd.DataFrame({'log2FC': fcs, 'pvalue': ps}, index=mat.index)
    _, padj, _, _ = multipletests(res['pvalue'], method='fdr_bh')
    res['padj'] = padj
    return res
# 5. 火山图可视化
# def plot_volcano(res, proj, targets):
#     # 只去掉 padj 为 NaN 的行，保留 log2FC 为 NaN 或 0 的行
#     res = res.dropna(subset=['padj'])
#
#     # 如果删完之后没有点，给出提示并提前返回
#     if res.empty:
#         print(f"No data to plot for {proj} after dropping NaNs in 'padj'.")
#         return
#
#     # 准备坐标
#     x = res['log2FC']
#     y = -np.log10(res['padj'] + 1e-300)
#
#     # 开画布
#     plt.figure(figsize=(7, 5))
#
#     # 背景散点
#     plt.scatter(x, y, s=8, alpha=0.5, label='_nolegend_')
#
#     # 高亮 targets
#     mask = res.index.isin(targets)
#     if mask.any():
#         plt.scatter(x[mask], y[mask], s=30, color='orange', label='Targets')
#         # 标注
#         for g in res.index[mask]:
#             plt.text(res.at[g, 'log2FC'],
#                      y.at[g],
#                      g,
#                      fontsize=8,
#                      ha='right',
#                      va='bottom')
#
#     # 美化
#     plt.axvline(0, linestyle='--', linewidth=1)
#     plt.xlabel('log₂ Fold Change')
#     plt.ylabel('-log₁₀ Adjusted p-value')
#     plt.title(f"{proj} Volcano Plot")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def deg_analysis_pydeseq2(mat, types):
    # Prepare the count matrix and sample metadata
    count_data = mat.values.T
    col_data = pd.DataFrame({
        'condition': ['Tumor' if 'Tumor' in types[s] else 'Normal' for s in mat.columns]
    }, index=mat.columns)

    # Create a DeseqDataSet object
    dds = DeseqDataSet(
        counts=count_data,
        metadata=col_data,
        design_factors='condition',  # Specify the design factor
        refit_cooks=True,
        n_cpus=8  # Adjust based on your system's capabilities
    )

    # Perform differential expression analysis
    dds.deseq2()
    stats = DeseqStats(dds, contrast=['condition', 'Tumor', 'Normal'])
    stats.summary()  # Run the statistical analysis

    # Access the results DataFrame
    res = stats.results_df

    # Convert results to DataFrame
    res['gene_name'] = mat.index
    res.set_index('gene_name', inplace=True)

    return res

def plot_volcano(res, proj, targets):
    # Filter out rows with NaN adjusted p-values
    res = res.dropna(subset=['padj'])

    # Prepare coordinates for the plot
    x = res['log2FoldChange']
    y = -np.log10(res['padj'] + 1e-100)

    # Create the plot
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=0.5, label='_nolegend_')

    # Highlight target genes
    mask = res.index.isin(targets)
    if mask.any():
        plt.scatter(x[mask], y[mask], s=30, color='orange', label='Targets')
        for g in res.index[mask]:
            plt.text(res.at[g, 'log2FoldChange'], y.at[g], g, fontsize=8, ha='right', va='bottom')

    # Customize the plot
    plt.axvline(0, linestyle='--', linewidth=1)
    plt.xlabel('log₂ Fold Change')
    plt.ylabel('-log₁₀ Adjusted p-value')
    plt.title(f"{proj} Volcano Plot")
    plt.legend()
    plt.tight_layout()
    plt.show()
# def plot_volcano(res, proj, targets):
#     # Replace infinite values and NaNs with zeros or remove them
#     res = res.replace([np.inf, -np.inf], np.nan)
#     res = res.dropna(subset=['log2FoldChange', 'padj'])
#
#     # Prepare coordinates for plotting
#     x = res['log2FoldChange']
#     y = -np.log10(res['padj'] + 1e-300)
#
#     # Create volcano plot
#     plt.figure(figsize=(7, 5))
#     plt.scatter(x, y, s=8, alpha=0.5, label='_nolegend_')
#
#     # Highlight target genes
#     mask = res.index.isin(targets)
#     if mask.any():
#         plt.scatter(x[mask], y[mask], s=30, color='orange', label='Targets')
#         for g in res.index[mask]:
#             plt.text(res.at[g, 'log2FoldChange'], y.at[g], g, fontsize=8, ha='right', va='bottom')
#
#     plt.axvline(0, linestyle='--', linewidth=1)
#     plt.xlabel('log₂ Fold Change')
#     plt.ylabel('-log₁₀ Adjusted p-value')
#     plt.title(f"{proj} Volcano Plot")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# 主流程
def main():

    targets = [
                # 'ENSG00000234741',  # SNORD76
                # 'ENSG00000208772',  # SNORD94
                # 'ENSG00000263934', # SNORD3A
                'ENSG00000221102', # SNORA11B
                'ENSG00000265185',#snoRD3B-1

        ]
    proj = 'TCGA-BRCA'
    # proj = 'TCGA-LUAD'
    print(f"\n=== {proj} ===")
    info = query_files(proj)
    # download_manifest(info, proj)  # Uncomment to download
    mat, types = build_matrix(info, proj)
    res = deg_analysis_pydeseq2(mat, types)
    if res is not None:
        missing = [g for g in targets if g not in res.index]
        if missing:
            print(f"Warning: these genes were not found in {proj}: {missing}")
        print(res.loc[[g for g in targets if g in res.index]])
        plot_volcano(res, proj, targets)


if __name__ == '__main__':
    main()
