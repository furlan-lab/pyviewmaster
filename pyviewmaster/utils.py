import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import anndata as ad

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
    AnnData
        An AnnData object with the added assay, dimensionality reduction, and updated metadata.
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

    # Return updated AnnData object
    return adata

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
