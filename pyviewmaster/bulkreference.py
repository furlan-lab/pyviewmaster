import numpy as np
import pandas as pd
import anndata
import scipy.sparse
from scipy.stats import gaussian_kde
from multiprocessing import Pool
import warnings

def get_counts_adata(adata, layer=None):
    if layer is None:
        counts = adata.X
    else:
        counts = adata.layers[layer]
    return counts

def sum_counts(counts, axis=0):
    if scipy.sparse.issparse(counts):
        return np.array(counts.sum(axis=axis)).flatten()
    else:
        return counts.sum(axis=axis)

def splat_bulk_reference(query=None, 
                         ref=None, N=2, assay=None, 
                         bulk_feature_row="gene_short_name", 
                         bulk_assay_name=None, 
                         dist="sc-direct"):
    if dist not in ["sc-direct", "sc-mimic", "bulk"]:
        raise ValueError("dist must be one of 'sc-direct', 'sc-mimic', 'bulk'")

    if dist == "sc-mimic" or dist == "sc-direct":
        print("Finding count distribution of query")
        counts_query = get_counts_adata(query, layer=assay)
        sizes = sum_counts(counts_query, axis=0)
        n = len(sizes)
        sd_sizes = np.std(sizes, ddof=1)
        IQR_sizes = np.subtract(*np.percentile(sizes, [75, 25]))
        bw = 0.9 * min(sd_sizes, IQR_sizes / 1.34) * n ** (-1/5)
        replace_counts = False
    else:
        counts_ref_full = get_counts_adata(ref, layer=bulk_assay_name)
        sizes = sum_counts(counts_ref_full, axis=0)
        n = len(sizes)
        sd_sizes = np.std(sizes, ddof=1)
        IQR_sizes = np.subtract(*np.percentile(sizes, [75, 25]))
        bw = 0.9 * min(sd_sizes, IQR_sizes / 1.34) * n ** (-1/5)
        replace_counts = True

    print("Finding common features between ref and query")
    genes_query = query.var_names
    genes_ref = ref.var[bulk_feature_row].values
    universe = np.intersect1d(genes_ref, genes_query)

    if len(universe) == 0:
        raise ValueError("No common genes found between ref and query.")

    print(f"Simulating {N} single cells for every bulk dataset case")

    # Create mapping from bulk_feature_row to index
    bulk_feature_row_values = ref.var[bulk_feature_row].values
    gene_to_idx = {gene: idx for idx, gene in enumerate(bulk_feature_row_values)}
    indices = [gene_to_idx[gene] for gene in universe]

    counts_ref_full = get_counts_adata(ref, layer=bulk_assay_name)
    counts = counts_ref_full[indices, :]
    gene_names = universe

    min_size = sizes.min()
    max_size = sizes.max()

    def process_sample(args):
        n, counts_n = args
        rsums = counts_n
        rsums = pd.Series(rsums, index=gene_names)
        counts_gt0 = rsums[rsums > 0]
        if len(counts_gt0) == 0:
            warnings.warn(f"Sample {n} has no counts > 0.")
            return None
        splat_genes = counts_gt0.index.values
        splat_counts = counts_gt0.values.astype(int)
        splat = np.repeat(splat_genes, splat_counts)
        newsizes = np.random.choice(sizes, N, replace=True) + np.random.normal(0, bw, N)
        newsizes = newsizes[(newsizes > min_size) & (newsizes < max_size)]
        if len(newsizes) == 0:
            warnings.warn(f"No valid newsizes for sample {n}.")
            return None
        final_newsizes = np.random.choice(np.round(newsizes), N)
        dl = []
        for i in final_newsizes:
            sample_counts = np.random.choice(splat, int(i), replace=replace_counts)
            tab = pd.value_counts(sample_counts)
            all_counts = tab.reindex(gene_names, fill_value=0)
            dl.append(all_counts.values)
        return np.column_stack(dl)

    args_list = []
    for n in range(counts.shape[1]):
        counts_n = counts[:, n].toarray().flatten() if scipy.sparse.issparse(counts) else counts[:, n]
        args_list.append((n, counts_n))

    with Pool() as pool:
        results = pool.map(process_sample, args_list)

    # Filter out None results
    good_results = [res for res in results if res is not None]

    if not good_results:
        raise ValueError("No valid data generated.")

    # Concatenate data
    data = np.concatenate(good_results, axis=1)

    # Create metadata
    metai = np.repeat(np.arange(counts.shape[1]), N)
    meta = ref.obs.reset_index(drop=True)
    newmeta = meta.iloc[metai, :].reset_index(drop=True)

    # Create anndata object
    adata_new = anndata.AnnData(X=data.T, var=pd.DataFrame(index=gene_names), obs=newmeta)

    return adata_new
