import numpy as np
import scanpy as sc
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
# import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


def viewmaster(
    query_cds,
    ref_cds,
    ref_celldata_col,
    query_celldata_col=None,
    FUNC="mlr",
    norm_method="log",
    selected_genes=None,
    train_frac=0.8,
    tf_idf=False,
    scale=False,
    hidden_layers=[500, 100],
    learning_rate=1e-3,
    max_epochs=10,
    LSImethod=1,
    verbose=True,
    backend="wgpu",
    threshold=None,
    keras_model=None,
    dir="/tmp/sc_local",
    return_probs=False,
    return_type="object",
    debug=False,
    **kwargs
):
    """
    Run viewmastR using the new Rust implementation

    This function runs viewmastR using the new Rust implementation to generate various forms of training and test data from the given query and reference cell data sets.

    Parameters
    ----------
    query_cds : AnnData
        A cell data set (cds) to query.
    ref_cds : AnnData
        A reference cell data set (cds).
    ref_celldata_col : str
        The column in the reference cell data set containing cell data.
    query_celldata_col : str, optional
        The column in the query cell data set containing cell data. Defaults to "viewmastR_pred" if not provided.
    FUNC : str, optional
        The machine learning model to use (multinomial linear regression - 'mlr' or multilayer neural network - 'nn'). Default is "mlr".
    norm_method : str, optional
        The normalization method to use. Options are "log", "binary", "size_only", or "none".
    selected_genes : list, optional
        A list of pre-selected genes for analysis. If None, common features between reference and query are used.
    train_frac : float, optional
        The fraction of data to use for training. Default is 0.8.
    tf_idf : bool, optional
        Boolean indicating whether to perform TF-IDF transformation. Default is False.
    scale : bool, optional
        Boolean indicating whether to scale the data. Default is False.
    hidden_layers : list, optional
        A list specifying the number of neurons in hidden layers for neural network models. Default is [500, 100]. To use one hidden layer supply a list of length 1.
    learning_rate : float, optional
        The learning rate for training the model. Default is 1e-3.
    max_epochs : int, optional
        The maximum number of epochs for training the model. Default is 10.
    LSImethod : int, optional
        The method for Latent Semantic Indexing. Default is 1.
    verbose : bool, optional
        Boolean indicating whether to display verbose output. Default is True.
    dir : str, optional
        The directory to save output files. Default is "/tmp/sc_local".
    return_probs : bool, optional
        Return probabilities along with predictions. Default is False.
    return_type : str, optional
        The type of output to return. Options are "object" or "list". Default is "object".
    debug : bool, optional
        Run in debug mode.
    **kwargs
        Additional arguments.

    Returns
    -------
    AnnData or dict
        If return_type is "object", returns the modified query AnnData object. If return_type is "list", returns a dictionary containing the query AnnData object and training output.
    """
    # Validate arguments
    return_type = return_type.lower()
    if return_type not in ["object", "list"]:
        raise ValueError("return_type must be 'object' or 'list'")
    backend = backend.lower()
    if backend not in ["wgpu", "nd", "candle"]:
        raise ValueError("backend must be one of 'wgpu', 'nd', 'candle'")
    FUNC = FUNC.lower()
    if FUNC not in ["mlr", "nn", "nb"]:
        raise ValueError("FUNC must be one of 'mlr', 'nn', 'nb'")
    if len(hidden_layers) not in [1, 2]:
        raise ValueError("Only 1 or 2 hidden layers are allowed.")
    if not isinstance(query_cds, sc.AnnData) or not isinstance(ref_cds, sc.AnnData):
        raise TypeError("query_cds and ref_cds must be AnnData objects.")

    if debug:
        print("Dimension check:")
        print(f"\tQuery shape: {query_cds.shape}")
        print(f"\tReference shape: {ref_cds.shape}")
        print(f"\tSelected genes: {len(selected_genes) if selected_genes else 'None'}")

    training_list = setup_training(
        query_cds,
        ref_cds,
        ref_celldata_col=ref_celldata_col,
        norm_method=norm_method,
        selected_genes=selected_genes,
        train_frac=train_frac,
        tf_idf=tf_idf,
        scale=scale,
        LSImethod=LSImethod,
        verbose=verbose,
        return_type="list",
    )

    if not os.path.exists(dir):
        os.makedirs(dir)

    if query_celldata_col is None:
        query_celldata_col = "viewmastR_pred"

    if FUNC == "mlr":
        export_list = process_learning_obj_mlr(
            train=training_list["train"],
            test=training_list["test"],
            query=training_list["query"],
            labels=training_list["labels"],
            learning_rate=learning_rate,
            num_epochs=max_epochs,
            directory=dir,
            verbose=verbose,
            backend=backend,
        )
    elif FUNC == "nn":
        export_list = process_learning_obj_ann(
            train=training_list["train"],
            test=training_list["test"],
            query=training_list["query"],
            labels=training_list["labels"],
            hidden_size=hidden_layers,
            learning_rate=learning_rate,
            num_epochs=max_epochs,
            directory=dir,
            verbose=verbose,
            backend=backend,
        )
    elif FUNC == "nb":
        export_list = process_learning_obj_nb(
            train=training_list["train"],
            test=training_list["test"],
            query=training_list["query"],
        )
        if return_type == "probs":
            print("Probabilities from multinomial naive bayes not implemented yet")
        query_cds.obs[query_celldata_col] = [
            training_list["labels"][idx] for idx in export_list["predictions"]
        ]
        if return_type == "object":
            return query_cds
        else:
            return {"object": query_cds, "training_output": export_list}

    # Process probabilities and predictions
    probabilities = export_list["probs"]  # Shape: (n_samples, n_classes)
    predictions = np.argmax(probabilities, axis=1)
    query_cds.obs[query_celldata_col] = [
        training_list["labels"][idx] for idx in predictions
    ]

    if return_probs:
        for i, label in enumerate(training_list["labels"]):
            query_cds.obs[f"prob_{label}"] = probabilities[:, i]

    if return_type == "object":
        return query_cds
    else:
        return {"object": query_cds, "training_output": export_list}


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
    addbias=False,
    return_type="list",
    debug=False,
    **kwargs
):
    """
    Setup training datasets

    This function sets up training datasets for use in machine learning models.

    Parameters
    ----------
    query_cds : AnnData
        A cell data set (cds) to query.
    ref_cds : AnnData
        A reference cell data set (cds).
    ref_celldata_col : str
        The column in the reference cell data set containing cell data.
    norm_method : str, optional
        The normalization method to use. Options are "log", "binary", "size_only", or "none".
    selected_genes : list, optional
        A list of pre-selected genes for analysis.
    train_frac : float, optional
        The fraction of data to use for training.
    tf_idf : bool, optional
        Boolean indicating whether to perform TF-IDF transformation.
    scale : bool, optional
        Boolean indicating whether to scale the data.
    LSImethod : int, optional
        The method for Latent Semantic Indexing.
    verbose : bool, optional
        Boolean indicating whether to display verbose output.
    addbias : bool, optional
        Boolean indicating whether to add bias.
    return_type : str, optional
        The type of output to return. Options are "list" or "matrix".
    debug : bool, optional
        Debug mode.
    **kwargs
        Additional arguments.

    Returns
    -------
    dict
        A dictionary containing the training datasets.
    """
    import warnings

    if verbose:
        print("Checking arguments and input")
    if tf_idf and scale:
        warnings.warn(
            "Both tf_idf and scale selected. Cannot do this as they are both scaling methods. Using tf_idf alone"
        )
        scale = False
    norm_method = norm_method.lower()
    if norm_method not in ["log", "binary", "size_only", "none"]:
        raise ValueError(
            "norm_method must be one of 'log', 'binary', 'size_only', 'none'"
        )
    return_type = return_type.lower()
    if return_type not in ["list", "matrix"]:
        raise ValueError("return_type must be one of 'list' or 'matrix'")

    # Find common features
    if verbose:
        print("Finding common features between reference and query")
    ref_cds, query_cds = common_features(ref_cds, query_cds)

    if selected_genes is None:
        selected_common = ref_cds.var_names
    else:
        if verbose:
            print("Subsetting by pre-selected features")
        selected_common = ref_cds.var_names.intersection(selected_genes)
    ref_cds = ref_cds[:, selected_common].copy()
    query_cds = query_cds[:, selected_common].copy()

    if verbose:
        print("Calculated normalized counts")
    # Apply normalization
    ref_cds_norm = get_norm_counts(ref_cds, norm_method=norm_method)
    query_cds_norm = get_norm_counts(query_cds, norm_method=norm_method)

    # Apply TF-IDF or scaling
    if tf_idf:
        if verbose:
            print("Performing TF-IDF")
        ref_cds_norm = tf_idf_transform(ref_cds_norm, LSImethod=LSImethod)
        query_cds_norm = tf_idf_transform(query_cds_norm, LSImethod=LSImethod)
    elif scale:
        if verbose:
            print("Scaling data")
        sc.pp.scale(ref_cds_norm)
        sc.pp.scale(query_cds_norm)

    # Prepare labels
    labels = ref_cds_norm.obs[ref_celldata_col]
    label_encoder = LabelEncoder()
    Ylab = label_encoder.fit_transform(labels)
    label_classes = label_encoder.classes_
    Y = OneHotEncoder(sparse=False).fit_transform(Ylab.reshape(-1, 1))
    features = ref_cds_norm.var_names

    # Get data matrices
    X = ref_cds_norm.X
    query = query_cds_norm.X
    if issparse(X):
        X = X.toarray()
    if issparse(query):
        query = query.toarray()

    # Split train and test
    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]), train_size=train_frac, stratify=Ylab
    )

    if return_type == "matrix":
        return {
            "Xtrain_data": X[train_idx, :],
            "Xtest_data": X[test_idx, :],
            "Ytrain_label": Y[train_idx, :],
            "Ytest_label": Y[test_idx, :],
            "query": query,
            "label_text": label_classes,
            "features": features,
        }
    elif return_type == "list":
        # Prepare list of dicts as in R code
        train = [
            {"data": X[idx, :], "target": Ylab[idx]} for idx in train_idx
        ]
        test = [
            {"data": X[idx, :], "target": Ylab[idx]} for idx in test_idx
        ]
        query_list = [{"data": query[idx, :]} for idx in range(query.shape[0])]
        return {
            "train": train,
            "test": test,
            "query": query_list,
            "labels": label_classes,
            "features": features,
        }


def common_features(ref_cds, query_cds):
    common_genes = ref_cds.var_names.intersection(query_cds.var_names)
    ref_cds = ref_cds[:, common_genes].copy()
    query_cds = query_cds[:, common_genes].copy()
    return ref_cds, query_cds


def get_norm_counts(adata, norm_method="log"):
    adata = adata.copy()
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
    return adata


def tf_idf_transform(adata, LSImethod=1):
    """
    Apply TF-IDF transformation to an AnnData object.
    """
    from sklearn.feature_extraction.text import TfidfTransformer

    X = adata.X
    if issparse(X):
        X = X.toarray()
    transformer = TfidfTransformer(
        norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False
    )
    X_tfidf = transformer.fit_transform(X.T).T  # Transpose to have features as terms
    adata.X = X_tfidf
    return adata


def process_learning_obj_mlr(
    train, test, query, labels, learning_rate, num_epochs, directory, verbose, backend
):
    """
    Process learning object using multinomial logistic regression.
    """
    X_train = np.array([item["data"] for item in train])
    y_train = np.array([item["target"] for item in train])
    X_query = np.array([item["data"] for item in query])

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=num_epochs
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    probs = clf.predict_proba(X_query)
    return {"predictions": predictions, "probs": probs}


def process_learning_obj_ann(
    train,
    test,
    query,
    labels,
    hidden_size,
    learning_rate,
    num_epochs,
    directory,
    verbose,
    backend,
):
    """
    Process learning object using neural network.
    """
    X_train = np.array([item["data"] for item in train])
    y_train = np.array([item["target"] for item in train])
    X_query = np.array([item["data"] for item in query])

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=learning_rate,
        max_iter=num_epochs,
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    probs = clf.predict_proba(X_query)
    return {"predictions": predictions, "probs": probs}


def process_learning_obj_nb(train, test, query):
    """
    Process learning object using Naive Bayes.
    """
    X_train = np.array([item["data"] for item in train])
    y_train = np.array([item["target"] for item in train])
    X_query = np.array([item["data"] for item in query])

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_query)
    return {"predictions": predictions}
