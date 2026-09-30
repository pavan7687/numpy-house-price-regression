"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    mean = np.nanmean(X, axis = 0)
    mean = np.nan_to_num(mean , nan=0.0)
    row , col = np.where(np.isnan(X))
    X[row, col] = mean[col]
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.quantile(X, 0.25,axis=0)
    q3 = np.quantile(X, 0.75, axis = 0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower,upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X,col))

# Step 6 - one_hot_encode
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    encoder = OneHotEncoder(sparse_output = False)
    return encoder.fit_transform(np.asarray(labels).reshape(-1,1))

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X,axis = 0)
    std = np.std(X,axis = 0)
    std[std==0] = 1.0
    return mean , std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    n = X.shape[0]
    ones = np.ones(n,dtype=float).reshape(-1)
    return np.column_stack((ones,X))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)
    arr = np.arange(n_samples)
    rng.shuffle(arr)
    return arr

# Step 11 - partition_indices
import numpy as np

def partition_indices(indices, train_ratio, val_ratio):
    n = len(indices)

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices],y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true-y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return (np.sum((y_pred-y_true)**2)/len(y_pred))**0.5

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred

    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "median_abs": float(np.median(np.abs(r)))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    low,up = compute_iqr_bounds(X ,iqr_k)
    return clip_columns(X,low,up)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]

    ratio = make_ratio_feature(numerator, denominator)

    X = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)
        X = np.hstack((X, cat_block))

    return X

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    n = len(X)

    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx]
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean, std = fit_standardizer(splits["X_train"])

    X_train = apply_standardizer(splits["X_train"], mean, std)
    X_val = apply_standardizer(splits["X_val"], mean, std)
    X_test = apply_standardizer(splits["X_test"], mean, std)

    X_train = add_bias_column(X_train)
    X_val = add_bias_column(X_val)
    X_test = add_bias_column(X_test)

    std_splits = {
        "X_train": X_train,
        "y_train": splits["y_train"],
        "X_val": X_val,
        "y_val": splits["y_val"],
        "X_test": X_test,
        "y_test": splits["y_test"]
    }

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline
import numpy as np


# ============================================================
# 1. DATA CLEANING
# ============================================================

def impute_nan_with_mean(X):
    X = np.asarray(X, dtype=float).copy()

    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])

        if np.all(mask):
            X[:, j] = 0.0
        else:
            mean = np.nanmean(X[:, j])
            X[mask, j] = mean

    return X


def compute_iqr_bounds(X, k=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper


def clip_columns(X, lower, upper):
    return np.clip(X, lower, upper)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def append_column(X, col):
    return np.column_stack((X, col))


def one_hot_encode(labels):
    labels = np.asarray(labels)

    categories = np.unique(labels)

    result = np.zeros(
        (len(labels), len(categories)),
        dtype=float
    )

    for i, label in enumerate(labels):
        j = np.searchsorted(categories, label)
        result[i, j] = 1.0

    return result


def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Replace zero standard deviations with 1
    std = np.where(std == 0, 1.0, std)

    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def add_bias_column(X):
    ones = np.ones(X.shape[0], dtype=float)

    return np.column_stack((ones, X))


def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)


def assemble_feature_matrix(
    X_num,
    ratio_num_idx,
    ratio_den_idx,
    cat_labels=None
):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]

    ratio = make_ratio_feature(
        numerator,
        denominator
    )

    X_out = append_column(
        X_num,
        ratio
    )

    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)

        X_out = np.column_stack(
            (X_out, cat_block)
        )

    return X_out


# ============================================================
# 3. REPRODUCIBLE SPLITS
# ============================================================

def make_shuffled_indices(n_samples, seed):
    rng = np.random.default_rng(seed)

    indices = np.arange(n_samples)

    rng.shuffle(indices)

    return indices


def partition_indices(indices, train_ratio, val_ratio):
    n = len(indices)

    train_n = int(np.floor(n * train_ratio))
    val_n = int(np.floor(n * val_ratio))

    train_idx = indices[:train_n]

    val_idx = indices[
        train_n:train_n + val_n
    ]

    test_idx = indices[
        train_n + val_n:
    ]

    return train_idx, val_idx, test_idx


def make_train_val_test(
    X,
    y,
    train_ratio,
    val_ratio,
    seed
):
    indices = make_shuffled_indices(
        len(X),
        seed
    )

    train_idx, val_idx, test_idx = partition_indices(
        indices,
        train_ratio,
        val_ratio
    )

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],

        "X_val": X[val_idx],
        "y_val": y[val_idx],

        "X_test": X[test_idx],
        "y_test": y[test_idx]
    }


# ============================================================
# 4. STANDARDIZATION + BIAS
# ============================================================

def standardize_and_add_bias(splits):

    # Fit ONLY on training data
    mean, std = fit_standardizer(
        splits["X_train"]
    )

    # Use training statistics for every split
    X_train = apply_standardizer(
        splits["X_train"],
        mean,
        std
    )

    X_val = apply_standardizer(
        splits["X_val"],
        mean,
        std
    )

    X_test = apply_standardizer(
        splits["X_test"],
        mean,
        std
    )

    # Add bias/intercept column
    X_train = add_bias_column(X_train)
    X_val = add_bias_column(X_val)
    X_test = add_bias_column(X_test)

    std_splits = {
        "X_train": X_train,
        "y_train": splits["y_train"],

        "X_val": X_val,
        "y_val": splits["y_val"],

        "X_test": X_test,
        "y_test": splits["y_test"]
    }

    return std_splits, mean, std


# ============================================================
# 5. ORDINARY LEAST SQUARES
# ============================================================

def ols_fit(X, y):
    # Use least-squares directly.
    # This works even when X.T @ X is singular.
    theta = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0]

    return theta


# ============================================================
# 6. METRICS
# ============================================================

def mean_absolute_error(y_true, y_pred):
    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def root_mean_squared_error(y_true, y_pred):
    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def r_squared(y_true, y_pred):
    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    # IMPORTANT:
    # Assignment explicitly requires 0.0
    # when SS_tot is zero.
    if ss_tot == 0:
        return 0.0

    return float(
        1.0 - ss_res / ss_tot
    )


def residual_summary(y_true, y_pred):
    r = y_true - y_pred

    return {
        "mean": float(
            np.mean(r)
        ),

        "std": float(
            np.std(r)
        ),

        "median_abs": float(
            np.median(np.abs(r))
        )
    }


def evaluate_predictions(y_true, y_pred):

    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred
        ),

        "rmse": root_mean_squared_error(
            y_true,
            y_pred
        ),

        "r2": r_squared(
            y_true,
            y_pred
        ),

        "residual_summary": residual_summary(
            y_true,
            y_pred
        )
    }


# ============================================================
# 7. FINAL END-TO-END PIPELINE
# ============================================================

def house_price_pipeline(
    X,
    y,
    ratio_num_idx,
    ratio_den_idx,
    cat_labels=None,
    seed=0
):

    # --------------------------------------------------------
    # Step 1: Clean missing values
    # --------------------------------------------------------

    X = impute_nan_with_mean(X)


    # --------------------------------------------------------
    # Step 2: Compute IQR bounds
    # --------------------------------------------------------

    lower, upper = compute_iqr_bounds(X)


    # --------------------------------------------------------
    # Step 3: Clip outliers
    # --------------------------------------------------------

    X = clip_columns(
        X,
        lower,
        upper
    )


    # --------------------------------------------------------
    # Step 4: Feature engineering
    # --------------------------------------------------------

    X = assemble_feature_matrix(
        X,
        ratio_num_idx,
        ratio_den_idx,
        cat_labels
    )


    # --------------------------------------------------------
    # Step 5: Train / validation / test split
    # --------------------------------------------------------

    splits = make_train_val_test(
        X,
        y,
        0.6,
        0.2,
        seed
    )


    # --------------------------------------------------------
    # Step 6: Standardize + bias
    # --------------------------------------------------------

    std_splits, mean, std = standardize_and_add_bias(
        splits
    )


    # --------------------------------------------------------
    # Step 7: OLS
    # --------------------------------------------------------

    theta = ols_fit(
        std_splits["X_train"],
        std_splits["y_train"]
    )


    # --------------------------------------------------------
    # Step 8: Validation prediction
    # --------------------------------------------------------

    y_val_pred = (
        std_splits["X_val"] @ theta
    )


    # --------------------------------------------------------
    # Step 9: Test prediction
    # --------------------------------------------------------

    y_test_pred = (
        std_splits["X_test"] @ theta
    )


    # --------------------------------------------------------
    # Step 10: Test metrics
    # --------------------------------------------------------

    test_metrics = evaluate_predictions(
        std_splits["y_test"],
        y_test_pred
    )


    # --------------------------------------------------------
    # Step 11: Validation metrics
    # --------------------------------------------------------

    val_metrics = evaluate_predictions(
        std_splits["y_val"],
        y_val_pred
    )


    # --------------------------------------------------------
    # Step 12: Return
    # --------------------------------------------------------

    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics
    }

