import numpy as np
import pandas as pd
import anndata
import scipy.sparse
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import warnings
from warnings import warn


def get_counts_adata(adata, layer=None):
    return adata.layers[layer] if layer else adata.X


def sum_counts(counts, axis=0):
    return np.ravel(counts.sum(axis=axis))


def splat_bulk_reference(query=None,
                         ref=None, N=2, assay=None,
                         bulk_feature_row="gene_short_name",
                         bulk_assay_name=None,
                         dist="sc-direct"):
    if dist not in ["sc-direct", "sc-mimic", "bulk"]:
        raise ValueError("dist must be one of 'sc-direct', 'sc-mimic', 'bulk'")

    # Determine the sizes and bandwidth based on the distribution
    if dist in ["sc-mimic", "sc-direct"]:
        print("Finding count distribution of query")
        counts_query = get_counts_adata(query, layer=assay)
        sizes = sum_counts(counts_query, axis=0)
        replace_counts = False
    else:
        counts_ref_full = get_counts_adata(ref, layer=bulk_assay_name)
        sizes = sum_counts(counts_ref_full, axis=0)
        replace_counts = True

    n = len(sizes)
    sd_sizes = np.std(sizes, ddof=1)
    IQR_sizes = np.subtract(*np.percentile(sizes, [75, 25]))
    bw = 0.9 * min(sd_sizes, IQR_sizes / 1.34) * n ** (-1 / 5)

    print("Finding common features between ref and query")
    genes_query = query.var_names
    genes_ref = ref.var[bulk_feature_row].values
    universe = np.intersect1d(genes_ref, genes_query)

    if len(universe) == 0:
        raise ValueError("No common genes found between ref and query.")

    print(f"Simulating {N} single cells for every bulk dataset case")

    # Map genes to indices for efficient lookup
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes_ref)}
    indices = np.array([gene_to_idx[gene] for gene in universe])

    counts_ref_full = get_counts_adata(ref, layer=bulk_assay_name)
    counts = counts_ref_full[indices, :]
    gene_names = universe

    min_size = sizes.min()
    max_size = sizes.max()

    # Precompute the sizes and noise outside the function to avoid redundancy
    def process_sample(n):
        counts_n = counts[:, n].toarray().flatten() if scipy.sparse.issparse(counts) else counts[:, n]
        counts_gt0 = counts_n > 0
        if not np.any(counts_gt0):
            warnings.warn(f"Sample {n} has no counts > 0.")
            return None

        splat_genes = np.where(counts_gt0)[0]
        splat_counts = counts_n[counts_gt0]
        total_counts = splat_counts.sum()
        probs = splat_counts / total_counts

        # Generate newsizes with added noise
        newsizes = np.random.choice(sizes, N, replace=True) + np.random.normal(0, bw, N)
        newsizes = newsizes[(newsizes > min_size) & (newsizes < max_size)]
        if len(newsizes) == 0:
            warnings.warn(f"No valid newsizes for sample {n}.")
            return None
        final_newsizes = np.random.choice(np.round(newsizes).astype(int), N)

        # Sample counts using multinomial distribution
        dl = []
        for size in final_newsizes:
            sample_counts = np.random.multinomial(int(size), probs)
            counts_full = np.zeros(len(gene_names), dtype=int)
            counts_full[splat_genes] = sample_counts
            dl.append(counts_full)

        return np.column_stack(dl)

    num_cores = -1  # Use all available cores
    results = Parallel(n_jobs=num_cores)(
        delayed(process_sample)(n) for n in range(counts.shape[1])
    )

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

REGISTRY = {
    # typed vectors
    "integer_vector": "rds2py.read_atomic.read_integer_vector",
    "boolean_vector": "rds2py.read_atomic.read_boolean_vector",
    "string_vector": "rds2py.read_atomic.read_string_vector",
    "double_vector": "rds2py.read_atomic.read_double_vector",
    # dictionary
    "vector": "rds2py.read_dict.read_dict",
    # factors
    "factor": "rds2py.read_factor.read_factor",
    # Rle
    "Rle": "rds2py.read_rle.read_rle",
    # matrices
    "dgCMatrix": "rds2py.read_matrix.read_dgcmatrix",
    "dgRMatrix": "rds2py.read_matrix.read_dgrmatrix",
    "dgTMatrix": "rds2py.read_matrix.read_dgtmatrix",
    "ndarray": "rds2py.read_matrix.read_ndarray",
    # data frames
    "data.frame": "rds2py.read_frame.read_data_frame",
    "DFrame": "rds2py.read_frame.read_dframe",
    # genomic ranges
    "GRanges": "rds2py.read_granges.read_genomic_ranges",
    "GenomicRanges": "rds2py.read_granges.read_genomic_ranges",
    "CompressedGRangesList": "rds2py.read_granges.read_granges_list",
    "GRangesList": "rds2py.read_granges.read_granges_list",
    # summarized experiment
    "SummarizedExperiment": "rds2py.read_se.read_summarized_experiment",
    "RangedSummarizedExperiment": "rds2py.read_se.read_ranged_summarized_experiment",
    # single-cell experiment
    "SingleCellExperiment": "rds2py.read_sce.read_single_cell_experiment",
    "SummarizedExperimentByColumn": "rds2py.read_sce.read_alts_summarized_experiment_by_column",
    # multi assay experiment
    "MultiAssayExperiment": "rds2py.read_mae.read_multi_assay_experiment",
    "ExperimentList": "rds2py.read_dict.read_dict",
    # delayed matrices
    "H5SparseMatrix": "rds2py.read_delayed_matrix.read_hdf5_sparse",
}

def _dispatcher(robject: dict, **kwargs):
    _class_name = get_class(robject)

    if _class_name is None:
        return None

    # if a class is registered, coerce the object
    # to the representation.
    if _class_name in REGISTRY:
        try:
            command = REGISTRY[_class_name]
            if isinstance(command, str):
                last_period = command.rfind(".")
                mod = import_module(command[:last_period])
                command = getattr(mod, command[last_period + 1 :])
                REGISTRY[_class_name] = command

            return command(robject, **kwargs)
        except Exception as e:
            warn(
                f"Failed to coerce RDS object to class: '{_class_name}', returning the dictionary, {str(e)}",
                RuntimeWarning,
            )
    else:
        warn(
            f"RDS file contains an unknown class: '{_class_name}', returning the dictionary",
            RuntimeWarning,
        )

    return robject

def get_counts_rds_obj(robj):
    ints = robj["attributes"]["assays"]["attributes"]["data"]["attributes"]["listData"]["data"][0]['data']
    dims = robj["attributes"]["assays"]["attributes"]["data"]["attributes"]["listData"]["data"][0]['attributes']['dim']['data']
    return csr_matrix(np.reshape(ints, (-1, dims[0])), dtype=np.int32)

def get_coldata_rds_obj(robj):
    data = {}
    robject = robj["attributes"]["colData"]
    col_names = _dispatcher(robject["attributes"]["listData"]["attributes"]["names"])
    for idx, colname in enumerate(col_names):
        data[colname] = _dispatcher(robject["attributes"]["listData"]["data"][idx])

    index = None
    if robject["attributes"]["rownames"]["data"]:
        index = _dispatcher(robject["attributes"]["rownames"])

    nrows = None
    if robject["attributes"]["nrows"]["data"]:
        nrows = list(_dispatcher(robject["attributes"]["nrows"]))[0]

    df = BiocFrame(
        data,
        # column_names=col_names,
        row_names=index,
        number_of_rows=nrows,
    )
    meta = df.to_pandas()
    meta.set_index("rownames")  
    return meta

def get_rowdata_rds_obj(robj):
    data = {}
    robject = robj["attributes"]["elementMetadata"]
    row_names = _dispatcher(robject["attributes"]["listData"]["attributes"]["names"])
    for idx, colname in enumerate(row_names):
        data[colname] = _dispatcher(robject["attributes"]["listData"]["data"][idx])

    index = None
    if robject["attributes"]["rownames"]["data"]:
        index = _dispatcher(robject["attributes"]["rownames"])

    nrows = None
    if robject["attributes"]["nrows"]["data"]:
        nrows = list(_dispatcher(robject["attributes"]["nrows"]))[0]

    df = BiocFrame(
        data,
        # column_names=col_names,
        row_names=index,
        number_of_rows=nrows,
    )
    var = df.to_pandas()
    var.index = var['gene_short_name']
    return var