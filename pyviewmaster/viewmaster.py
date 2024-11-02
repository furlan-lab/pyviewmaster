import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
# import warnings

def viewmaster(
    query_cds,
    ref_cds,
    ref_celldata_col,
    query_celldata_col="viewmaster_pred",
    FUNC="mlr",
    norm_method="log",
    selected_genes=None,
    train_frac=0.8,
    tf_idf=False,
    scale=False,
    hidden_layers=(500, 100),
    learning_rate=1e-3,
    max_epochs=200,
    LSImethod=1,
    verbose=True,
    return_probs=False,
    debug=False,
    **kwargs
):
    """
    Predict cell types in the query dataset using the reference dataset.
    The query_cds is modified in place, adding predictions to query_cds.obs.
    """
    if not isinstance(query_cds, sc.AnnData) or not isinstance(ref_cds, sc.AnnData):
        raise TypeError("query_cds and ref_cds must be AnnData objects.")

    if debug:
        print("Dimension check:")
        print(f"\tQuery shape: {query_cds.shape}")
        print(f"\tReference shape: {ref_cds.shape}")
        print(f"\tSelected genes: {len(selected_genes) if selected_genes else 'None'}")

    # Setup training data
    training_data = setup_training(
        query_cds,
        ref_cds,
        ref_celldata_col=ref_celldata_col,
        norm_method=norm_method,
        selected_genes=selected_genes,
        train_frac=train_frac,
        tf_idf=tf_idf,
        scale=scale,
        verbose=verbose,
        debug=debug,
    )

    # Choose the machine learning model
    FUNC = FUNC.lower()
    if FUNC == "mlr":
        export_list = process_learning_obj_mlr(
            X_train=training_data["X_train"],
            y_train=training_data["y_train"],
            X_query=training_data["X_query"],
            max_epochs=max_epochs,
            verbose=verbose,
        )
    elif FUNC == "nn":
        export_list = process_learning_obj_ann(
            X_train=training_data["X_train"],
            y_train=training_data["y_train"],
            X_query=training_data["X_query"],
            hidden_size=hidden_layers,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            verbose=verbose,
        )
    elif FUNC == "nb":
        export_list = process_learning_obj_nb(
            X_train=training_data["X_train"],
            y_train=training_data["y_train"],
            X_query=training_data["X_query"],
            verbose=verbose,
        )
    else:
        raise ValueError("FUNC must be one of 'mlr', 'nn', or 'nb'.")

    # Process predictions
    if FUNC == "nb":
        predictions = export_list["predictions"]
        query_cds.obs[query_celldata_col] = training_data["labels"][predictions]
    else:
        probabilities = export_list["probs"]
        predictions = np.argmax(probabilities, axis=1)
        query_cds.obs[query_celldata_col] = training_data["labels"][predictions]

        if return_probs:
            for i, label in enumerate(training_data["labels"]):
                query_cds.obs[f"prob_{label}"] = probabilities[:, i]

    if verbose:
        print(f"Predictions added to query_cds.obs['{query_celldata_col}']")

def setup_training(
    query_cds,
    ref_cds,
    ref_celldata_col,
    norm_method="log",
    selected_genes=None,
    train_frac=0.8,
    tf_idf=False,
    scale=False,
    LSImethod=1,
    verbose=True,
    debug=False,
):
    """
    Prepare training and query datasets.
    """
    if verbose:
        print("Preparing data...")

    # Find common features
    ref_cds, query_cds = common_features(ref_cds, query_cds)

    # Subset to selected genes if provided
    if selected_genes is not None:
        if verbose:
            print("Subsetting to selected genes.")
        selected_common = ref_cds.var_names.intersection(selected_genes)
        ref_cds = ref_cds[:, selected_common]
        query_cds = query_cds[:, selected_common]

    # Normalize data
    if verbose:
        print("Normalizing data...")
    ref_cds_norm = ref_cds.copy()
    query_cds_norm = query_cds.copy()
    normalize_data(ref_cds_norm, norm_method=norm_method)
    normalize_data(query_cds_norm, norm_method=norm_method)

    # Apply TF-IDF or scaling if specified
    if tf_idf:
        if verbose:
            print("Applying TF-IDF transformation.")
        apply_tfidf(ref_cds_norm)
        apply_tfidf(query_cds_norm)
    elif scale:
        if verbose:
            print("Scaling data.")
        sc.pp.scale(ref_cds_norm)
        sc.pp.scale(query_cds_norm)

    # Prepare labels
    labels = ref_cds_norm.obs[ref_celldata_col].values
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(labels)
    label_classes = label_encoder.classes_

    # Get data matrices
    X = ref_cds_norm.X
    X_query = query_cds_norm.X

    # Split train indices
    if train_frac < 1.0:
        train_idx, _ = train_test_split(
            np.arange(X.shape[0]), train_size=train_frac, stratify=y_labels
        )
        X_train = X[train_idx]
        y_train = y_labels[train_idx]
    else:
        X_train = X
        y_train = y_labels

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_query": X_query,
        "labels": label_classes,
    }

def common_features(ref_cds, query_cds):
    """
    Find and retain common genes between ref_cds and query_cds.
    """
    common_genes = ref_cds.var_names.intersection(query_cds.var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between reference and query datasets.")
    ref_cds = ref_cds[:, common_genes]
    query_cds = query_cds[:, common_genes]
    return ref_cds, query_cds

def normalize_data(adata, norm_method="log"):
    """
    Normalize counts in an AnnData object in-place.
    """
    if norm_method == "log":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif norm_method == "binary":
        adata.X = (adata.X != 0).astype(float)
    elif norm_method == "size_only":
        sc.pp.normalize_total(adata, target_sum=1e4)
    elif norm_method == "none":
        pass  # Do nothing
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

def apply_tfidf(adata):
    """
    Apply TF-IDF transformation to an AnnData object in-place.
    """
    X = adata.X
    transformer = TfidfTransformer(
        norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False
    )
    if issparse(X):
        X = transformer.fit_transform(X.T).T  # Transpose for TF-IDF, then transpose back
    else:
        X = transformer.fit_transform(X.T).T
    adata.X = X

def process_learning_obj_mlr(
    X_train, y_train, X_query, max_epochs=200, verbose=True
):
    """
    Train and predict using multinomial logistic regression.
    """
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=max_epochs
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    probs = clf.predict_proba(X_query)
    return {"predictions": predictions, "probs": probs}

def process_learning_obj_ann(
    X_train, y_train, X_query, hidden_size=(500, 100), learning_rate=1e-3, max_epochs=200, verbose=True
):
    """
    Train and predict using a neural network.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=learning_rate,
        max_iter=max_epochs,
        verbose=verbose,
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    probs = clf.predict_proba(X_query)
    return {"predictions": predictions, "probs": probs}

def process_learning_obj_nb(X_train, y_train, X_query, verbose=True):
    """
    Train and predict using Multinomial Naive Bayes.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    return {"predictions": predictions}
