import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

def simulate_single_cells(
    ref,
    query,
    N,
    dist='sc-direct',
    sc_assay_name='counts',
    bulk_assay_name='counts',
    bulk_feature_row='gene_short_name'
):
    """
    Simulates N single cells for every bulk dataset case in `ref`,
    using the count distribution from `query`.
    
    Parameters:
    - ref (AnnData): The bulk reference AnnData object.
    - query (AnnData): The single-cell query AnnData object.
    - N (int): Number of single cells to generate per sample in `ref`.
    - dist (str): Distribution to use for generating total counts ('sc-model' or 'sc-direct').
    - sc_assay_name (str): Name of the layer in `query` containing scRNA-seq counts.
    - bulk_feature_row (str): Column name in `ref.var` containing gene identifiers.
    
    Returns:
    - expanded_adata (AnnData): An AnnData object with simulated single-cell data.
    """
    def get_counts_adata(adata, layer=None):
        """
        Retrieves the counts matrix from an AnnData object.
        If layer is specified, uses that layer; otherwise, uses adata.X.
        """
        if layer is not None:
            counts = adata.layers[layer]
        else:
            counts = adata.X
        return counts

    # Get counts from the query dataset
    counts_query = get_counts_adata(query, layer=sc_assay_name)
    if sparse.issparse(counts_query):
        sizes = counts_query.sum(axis=1).A1  # Convert to 1D array
    else:
        sizes = counts_query.sum(axis=1)
    # min_size = sizes.min()
    # max_size = sizes.max()
    
    # Number of single cells to generate
    ss_cells = N * ref.shape[0]
    
    # Generate total counts for downsampled cells
    if dist == "sc-model":
        print("Modeling count distribution of query using Empirical CDF")
        # Sort sizes to create the CDF
        sorted_sizes = np.sort(sizes)
        cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        # Draw random samples from a uniform distribution
        uniform_samples = np.random.rand(ss_cells)
        # Use inverse transform sampling to get values that match the empirical CDF
        final_newsizes = np.interp(uniform_samples, cdf, sorted_sizes).astype(int)
    else:
        # Use the direct sampling approach for "sc-direct"
        final_newsizes = np.random.choice(sizes, ss_cells, replace=True).astype(int)
    
    print("Finding common features between ref and query")
    genes_query = query.var_names
    genes_ref = ref.var[bulk_feature_row].values
    universe = np.intersect1d(genes_ref, genes_query)
    
    if len(universe) == 0:
        raise ValueError("No common genes found between ref and query.")
    
    print(f"Simulating {N} single cells for every bulk dataset case")
    
    # Prepare the reference counts data
    counts_ref_full = get_counts_adata(ref, layer=bulk_assay_name)
    if sparse.issparse(counts_ref_full):
        counts_ref_full = counts_ref_full.todense()
    counts_ref_full = pd.DataFrame(
        counts_ref_full.T,
        index=ref.var_names,
        columns=ref.obs_names
    )
    # Ensure genes_ref are aligned with counts_ref_full
    counts_ref_full = counts_ref_full.loc[genes_ref]
    # Keep only the common genes
    counts_ref_full = counts_ref_full.loc[universe]
    # Create an AnnData object with the filtered counts
    fdata = ad.AnnData(counts_ref_full.T)
    fdata.obs = ref.obs.copy()
    
    # Ensure the number of cells matches
    assert ss_cells == fdata.n_obs * N, "Mismatch in the number of single cells to generate."
    
    # Define the downsampling function
    def downsample_counts_vectorized(counts_matrix, new_total_counts):
        """
        Downsamples counts for multiple cells using vectorized multinomial sampling.
        """
        # Compute total counts per cell
        original_total_counts = counts_matrix.sum(axis=1)
        # Avoid division by zero
        nonzero_mask = original_total_counts > 0
        # Compute probabilities
        probabilities = np.zeros_like(counts_matrix, dtype=float)
        probabilities[nonzero_mask] = (
            counts_matrix[nonzero_mask] /
            original_total_counts[nonzero_mask, np.newaxis]
        )
        # Initialize the random number generator
        rng = np.random.default_rng()
        # Prepare an array to hold the downsampled counts
        downsampled_counts = np.zeros_like(counts_matrix, dtype=int)
        # Perform multinomial sampling where valid
        valid_mask = nonzero_mask & (new_total_counts > 0)
        if np.any(valid_mask):
            downsampled_counts[valid_mask] = rng.multinomial(
                n=new_total_counts[valid_mask],
                pvals=probabilities[valid_mask]
            )
        return downsampled_counts

    # Define the expand function
    def expand_anndata(adata, fold=10, total_counts_vector=None):
        """
        Expands an AnnData object by downsampling counts using vectorized sampling.
        The `obs` DataFrame is similarly expanded.
        """
        # Convert counts to dense matrix if sparse
        if sparse.issparse(adata.X):
            original_counts = adata.X.toarray()
        else:
            original_counts = adata.X.copy()
        
        num_cells, num_genes = original_counts.shape
        
        # Validate the total_counts_vector
        if total_counts_vector is not None:
            expected_length = num_cells * fold
            if len(total_counts_vector) != expected_length:
                raise ValueError(
                    f"The length of total_counts_vector ({len(total_counts_vector)}) "
                    f"must be equal to the number of expanded cells ({expected_length})."
                )
            new_total_counts = np.maximum(np.array(total_counts_vector, dtype=int), 0)
        else:
            # Use the original total counts repeated `fold` times
            original_total_counts = original_counts.sum(axis=1)
            new_total_counts = np.tile(original_total_counts, fold)
        
        # Expand counts by repeating each cell `fold` times
        expanded_counts = np.repeat(original_counts, fold, axis=0)
        
        # Downsample counts using vectorized multinomial sampling
        downsampled_counts = downsample_counts_vectorized(expanded_counts, new_total_counts)
        
        # Expand the obs DataFrame
        expanded_obs = pd.DataFrame(
            np.repeat(adata.obs.values, fold, axis=0),
            columns=adata.obs.columns
        )
        
        # Generate replicate numbers and new observation names
        replicate_numbers = np.tile(np.arange(1, fold + 1), num_cells)
        repeated_indices = np.repeat(adata.obs_names.values, fold)
        expanded_obs_names = [
            f"{obs}_rep{rep}" for obs, rep in zip(repeated_indices, replicate_numbers)
        ]
        expanded_obs.index = expanded_obs_names
        
        # Create a new AnnData object with the downsampled counts and expanded obs
        expanded_adata = ad.AnnData(
            X=sparse.csr_matrix(downsampled_counts),
            obs=expanded_obs,
            var=adata.var.copy()
        )
        
        return expanded_adata
    
    # Expand the data
    expanded_adata = expand_anndata(
        fdata,
        fold=N,
        total_counts_vector=final_newsizes
    )
    
    return expanded_adata