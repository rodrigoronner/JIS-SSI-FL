import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_process_mimic(file_path, num_clients, dirichlet_alpha=0.5, seed=42):
    """
    Loads the MIMIC-IV cohort, performs preprocessing, splits it chronologically,
    and partitions the training fold among federated clients using a Dirichlet
    distribution (non-IID), applying SMOTETomek independently to each client's
    local training fold (Sec. 4.1/4.2 of the paper).

    Args:
        file_path (str): Path to the CSV file containing patient features.
                          If extracted via sql/extract_cohort.sql, it includes
                          an 'admittime' column enabling a true chronological
                          split. Older exports without 'admittime' fall back
                          to a documented proxy ordering (see warning below).
        num_clients (int): Number of federated clients (hospitals).
        dirichlet_alpha (float): Concentration parameter for the Dirichlet
                                  partition. Lower values -> more heterogeneous
                                  (non-IID) client data. Paper uses alpha=0.5.
        seed (int): Random seed for partitioning/resampling reproducibility.

    Returns:
        X_train, y_train, X_test, y_test (np.ndarray): Held-out evaluation split.
        client_data (dict): {client_id: (X_client, y_client)} — already
                             SMOTETomek-balanced per client, ready for local training.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    target_col = "hospital_expire_flag"
    if target_col not in df.columns:
        target_col = df.columns[-1]

    # --- Chronological ordering -------------------------------------------
    # A true chronological split requires an admission timestamp. The
    # extraction query in sql/extract_cohort.sql produces one ('admittime').
    # Older/reduced CSVs shipped without it (no timestamp column at all);
    # in that case we fall back to hadm_id ordering and say so explicitly,
    # rather than silently claiming a chronological split that isn't possible.
    id_cols = []
    if "admittime" in df.columns:
        df["admittime"] = pd.to_datetime(df["admittime"])
        df = df.sort_values("admittime").reset_index(drop=True)
        id_cols = [c for c in ("hadm_id", "subject_id", "admittime") if c in df.columns]
    elif "hadm_id" in df.columns:
        warnings.warn(
            "No 'admittime' column found — this dataset does not support a "
            "true chronological split. Falling back to ordering by 'hadm_id' "
            "as a proxy. Re-extract the cohort with sql/extract_cohort.sql to "
            "obtain a genuine chronological split.",
            stacklevel=2,
        )
        df = df.sort_values("hadm_id").reset_index(drop=True)
        id_cols = ["hadm_id"]

    X_df = df.drop(columns=[target_col] + id_cols)
    y = df[target_col].values

    # --- Chronological 90/10 split FIRST (no shuffling: preserves temporal
    # order), so every fitted preprocessing step below (imputer, one-hot
    # category vocabulary, scaler) is fit on the training fold only. Fitting
    # any of them on the full dataset would leak test-set statistics into
    # training, e.g. mean-imputing NaNs using values seen only in the future.
    split_idx = int(len(X_df) * 0.9)
    X_train_df = X_df.iloc[:split_idx].copy()
    X_test_df = X_df.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx], y[split_idx:]

    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="mean")
        X_train_df[num_cols] = imputer_num.fit_transform(X_train_df[num_cols])
        X_test_df[num_cols] = imputer_num.transform(X_test_df[num_cols])

    if len(cat_cols) > 0:
        # Fix the one-hot vocabulary to categories observed in training;
        # unseen test-time categories fall back to the all-zero encoding
        # (equivalent to sklearn's OneHotEncoder(handle_unknown='ignore')).
        for col in cat_cols:
            categories = pd.Categorical(X_train_df[col]).categories
            X_train_df[col] = pd.Categorical(X_train_df[col], categories=categories)
            X_test_df[col] = pd.Categorical(X_test_df[col], categories=categories)
        X_train_df = pd.get_dummies(X_train_df, columns=list(cat_cols), drop_first=True)
        X_test_df = pd.get_dummies(X_test_df, columns=list(cat_cols), drop_first=True)
        X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

    X_train_raw = X_train_df.values.astype(np.float64)
    X_test_raw = X_test_df.values.astype(np.float64)
    print(f"Data processed. Feature matrix shape: train={X_train_raw.shape}, test={X_test_raw.shape}")

    # Fit the scaler on the training fold only to avoid test-set leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- Non-IID partitioning across clients (Dirichlet, alpha=0.5) ---------
    user_groups = dirichlet_partition(y_train, num_clients, alpha=dirichlet_alpha, seed=seed)

    # --- Per-client SMOTETomek (train folds only, applied independently) ----
    client_data = {}
    for client_id, idxs in user_groups.items():
        X_c, y_c = X_train[idxs], y_train[idxs]
        client_data[client_id] = balance_client_fold(X_c, y_c, seed=seed + client_id)

    return X_train, y_train, X_test, y_test, client_data


def dirichlet_partition(y_train, num_clients, alpha=0.5, seed=42):
    """
    Partitions training indices among clients using a Dirichlet distribution
    per class, producing non-IID client data (Sec. 4.2: alpha=0.5).

    Lower alpha yields more skewed (heterogeneous) per-client class
    distributions; alpha -> infinity approaches an IID split.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    client_idxs = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(idx_c, split_points)):
            client_idxs[client_id].extend(split.tolist())

    user_groups = {}
    for client_id in range(num_clients):
        idxs = np.array(client_idxs[client_id])
        rng.shuffle(idxs)
        user_groups[client_id] = idxs

    return user_groups


def balance_client_fold(X_client, y_client, seed=42):
    """
    Applies SMOTETomek exclusively to a single client's local training fold
    (Sec. 4.1). Test/validation data must never pass through this function.

    Falls back to the original (imbalanced) fold if the client has too few
    minority-class samples for SMOTE's k-neighbors requirement — this can
    happen for small/highly-skewed shards produced by the Dirichlet split.
    """
    from imblearn.combine import SMOTETomek

    classes, counts = np.unique(y_client, return_counts=True)
    if len(classes) < 2 or counts.min() < 6:
        warnings.warn(
            f"Skipping SMOTETomek for a client fold with class counts "
            f"{dict(zip(classes.tolist(), counts.tolist()))}: too few minority "
            f"samples for the default k-neighbors. Using the raw fold instead.",
            stacklevel=2,
        )
        return X_client, y_client

    smt = SMOTETomek(random_state=seed)
    X_res, y_res = smt.fit_resample(X_client, y_client)
    return X_res, y_res
