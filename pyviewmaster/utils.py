import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import anndata as ad
from scipy.sparse import csr_matrix
from rds2py.rdsutils import get_class
from biocframe import BiocFrame
from warnings import warn
from importlib import import_module

def add_souporcell_anndata(
    adata,
    souporcell_file,
    prefix=None,
    assay_name="GENO",
    key="gpca_",
    meta_data_label="geno",
    rd_label="gpca",
    rename_assignments=True
):
    """
    Add Souporcell Clustering Output to an AnnData Object

    This function adds an assay to an AnnData object based on Souporcell clustering results.
    The assay contains log-transformed and normalized cluster probabilities from the Souporcell `clusters.tsv` file.
    Principal Component Analysis (PCA) is performed on these probabilities, and the resulting components are added as a dimensionality reduction object.
    The Souporcell 'assignment' is added to the AnnData object's metadata under the specified label.
    Optionally, assignments can be renamed to be 1-indexed and multiplets collapsed into a single category.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    souporcell_file : str
        Path to the Souporcell `clusters.tsv` file.
    prefix : str, optional
        Optional prefix to prepend to cell barcodes.
    assay_name : str, optional
        Name of the assay to add to the AnnData object. Default is "GENO".
    key : str, optional
        Key for the dimensionality reduction. Default is "gpca_".
    meta_data_label : str, optional
        Name of the metadata column to store Souporcell assignments. Default is "geno".
    rd_label : str, optional
        Name of the dimensionality reduction object to store PCA results. Default is "gpca".
    rename_assignments : bool, optional
        Indicates whether to rename Souporcell assignments to be 1-indexed and collapse multiplets. Default is True.

    Returns
    -------
    None
        The function modifies the AnnData object in place.
    """
    # Check if the file exists
    if isinstance(souporcell_file, (list, tuple)):
        raise ValueError("Only supports one file addition at a time.")
    if not os.path.exists(souporcell_file):
        raise FileNotFoundError("Souporcell file does not exist!")

    # Check that adata is an AnnData object
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input object is not an AnnData object.")

    # Read Souporcell data
    souporcell_data = pd.read_csv(souporcell_file, sep='\t', header=0, dtype=str)

    # Check for required columns
    required_cols = ["barcode", "status", "assignment", "log_prob_singleton", "log_prob_doublet"]
    if not all(col in souporcell_data.columns for col in required_cols):
        raise ValueError(
            f"Souporcell file {souporcell_file} does not have the required columns: {', '.join(required_cols)}"
        )

    # Set row names
    if prefix is None:
        souporcell_data.index = souporcell_data['barcode']
    else:
        souporcell_data.index = prefix + souporcell_data['barcode']

    # Check that all cells in adata are in souporcell_data
    missing_cells = set(adata.obs_names) - set(souporcell_data.index)
    if len(missing_cells) > 0:
        raise ValueError("Not all cells in the AnnData object are found in the Souporcell data.")

    # Subset souporcell_data to only include cells in adata
    souporcell_data = souporcell_data.loc[adata.obs_names]

    # Extract cluster probabilities
    cluster_cols = [col for col in souporcell_data.columns if col.startswith('cluster')]
    if len(cluster_cols) == 0:
        raise ValueError("No cluster columns found in Souporcell data.")
    cluster_probs = souporcell_data[cluster_cols].astype(float).values

    # Normalize and log-transform cluster probabilities
    col_means = cluster_probs.mean(axis=0)
    cluster_probs_normalized = cluster_probs / col_means[np.newaxis, :]

    # Handle zeros and negative values before log transformation
    epsilon = np.finfo(float).eps
    cluster_probs_normalized[cluster_probs_normalized <= 0] = epsilon
    cluster_probs_log = np.log10(cluster_probs_normalized)

    # Perform PCA
    pca = PCA()
    pca_scores = pca.fit_transform(cluster_probs_log)

    # Prepare PCA scores DataFrame
    pca_scores_df = pd.DataFrame(
        pca_scores,
        index=souporcell_data.index,
        columns=[f"PC{i+1}" for i in range(pca_scores.shape[1])]
    )

    # Prepare assignments
    assignments = souporcell_data['assignment']
    if rename_assignments:
        fixed_assignments = fix_assignment(assignments)
    else:
        fixed_assignments = assignments

    # Create assay (store cluster_probs_log)
    adata.obsm[assay_name] = pd.DataFrame(
        cluster_probs_log,
        index=souporcell_data.index,
        columns=cluster_cols
    )

    # Create dimensionality reduction object (store PCA results)
    adata.obsm[rd_label] = pca_scores_df.values

    # Optionally, store PCA components in adata.uns
    adata.uns[key + 'variance_ratio'] = pca.explained_variance_ratio_

    # Update metadata
    adata.obs[meta_data_label] = fixed_assignments.loc[adata.obs_names].values

def fix_assignment(vector):
    """
    Helper function to fix assignments by collapsing multiplets and 1-indexing.

    Parameters
    ----------
    vector : pd.Series
        Series containing Souporcell assignments.

    Returns
    -------
    pd.Series
        Fixed assignments.
    """
    vector = vector.copy()
    vector[vector.str.contains("/")] = "Multiplet"
    is_not_multiplet = vector != "Multiplet"
    vector[is_not_multiplet] = (vector[is_not_multiplet].astype(int) + 1).astype(str)
    return vector

def sfc(n, scramble=False):
    """
    Generate a Color Palette

    Creates a color palette of specified length, optionally scrambled.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    scramble : bool, optional
        Indicates whether to randomly shuffle the colors. Default is False.

    Returns
    -------
    List[str]
        A list of hex color codes.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Please input a positive integer for 'n'.")
    base_colors = [
        "#16482A", "#1C7C40", "#45AC49", "#69BC9A", "#FBD43F",
        "#E77A2B", "#DC3F32", "#932528", "#50191E", "#96C4D9",
        "#2394C4", "#4575AD", "#8681B0", "#6C5492", "#8C4A8D",
        "#9E2563", "#492C74", "#E9E52F", "#F8C566", "#D85191"
    ]
    # Extend the palette if n is larger than the base_colors list
    palette_func = lambda x: [base_colors[i % len(base_colors)] for i in range(x)]
    colors = palette_func(n)
    if scramble:
        np.random.shuffle(colors)
    return colors

def sfcolors(n, scramble=False):
    """
    Generate a Color Palette

    Creates a color palette of specified length, optionally scrambled.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    scramble : bool, optional
        Indicates whether to randomly shuffle the colors. Default is False.

    Returns
    -------
    List[str]
        A list of hex color codes.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Please input a positive integer for 'n'.")
    base_colors = [
        "#16482A", "#1C7C40", "#45AC49", "#69BC9A", "#FBD43F",
        "#E77A2B", "#DC3F32", "#932528", "#50191E", "#96C4D9",
        "#2394C4", "#4575AD", "#8681B0", "#6C5492", "#8C4A8D",
        "#9E2563", "#492C74", "#E9E52F", "#F8C566", "#D85191"
    ]
    palette_func = lambda x: [base_colors[i % len(base_colors)] for i in range(x)]
    colors = palette_func(n)
    if scramble:
        np.random.shuffle(colors)
    return colors


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