import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from typing import Tuple, Optional


def _load_from_openml():
    # fetch with sklearn (may download the dataset on first run)
    df = fetch_openml(name="adult", version=2, as_frame=True)
    data = df.frame.copy()

    # target column is 'class' or 'income'
    if 'class' in data.columns:
        target_col = 'class'
    elif 'income' in data.columns:
        target_col = 'income'
    else:
        # pick last col
        target_col = data.columns[-1]

    y = data[target_col].astype(str)
    X = data.drop(columns=[target_col])

    # Separate numeric and categorical columns to avoid inserting string 'NA' into numeric columns
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna('NA')

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    X_vals = X.values.astype('float32')

    le = LabelEncoder()
    y_vals = le.fit_transform(y)

    return X_vals, y_vals


def _load_from_local_zip(zip_path: str):
    import zipfile

    # known column names for the UCI Adult dataset
    colnames = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]

    with zipfile.ZipFile(zip_path, 'r') as z:
        # read adult.data
        if 'adult.data' in z.namelist():
            with z.open('adult.data') as f:
                df_train = pd.read_csv(f, header=None, names=colnames, na_values='?', skipinitialspace=True)
        else:
            raise FileNotFoundError('adult.data not found inside zip')

        # read adult.test if present
        if 'adult.test' in z.namelist():
            with z.open('adult.test') as f:
                # adult.test has a header/first row that can be skipped and labels may end with a dot
                df_test = pd.read_csv(f, header=None, names=colnames, na_values='?', skiprows=1, skipinitialspace=True)
                # concatenate
                df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            df = df_train

    # strip periods from labels (e.g., ' >50K.' -> ' >50K')
    df['income'] = df['income'].astype(str).str.replace('.', '', regex=False).str.strip()

    X = df.drop(columns=['income'])
    y = df['income'].astype(str)

    # Separate numeric and categorical columns to avoid inserting string 'NA' into numeric columns
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    # Fill numeric missing values with column median, categorical with 'NA'
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna('NA')

    # One-hot encode categorical columns (if any)
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    # Ensure numeric numpy array (float32)
    X_vals = X.values.astype('float32')

    le = LabelEncoder()
    y_vals = le.fit_transform(y)

    return X_vals, y_vals


def get_adult_loaders(batch_size: int = 128,
                      val_size: float = 0.1,
                      test_size: float = 0.2,
                      max_samples: Optional[int] = None,
                      zip_path: Optional[str] = None,
                      # new preprocessing options
                      feature_selection: Optional[str] = None,  # 'pearson' or None
                      feature_threshold: float = 0.02,
                      aggregate_low_signal: bool = True,
                      one_hot_only: bool = False,
                      target_encode_cols: Optional[list] = None,
                      te_kfold: int = 5,
                      te_prior: float = 20.0,
                      pca_components: Optional[int] = 8,
                      drop_original_target_encoded: bool = True
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train, val, test DataLoaders for the Adult dataset.

    Behavior:
    - If local zip with adult.data and adult.test exists, use adult.data to split train/val and adult.test as final test set.
    - Otherwise, fetch from OpenML and perform a 3-way random split using val_size and test_size.
    """
    # If a local zip is provided or exists in the repo, prefer it to avoid network fetch
    import os
    if zip_path is None:
        import os

        default_zip = os.path.join(os.path.dirname(__file__), '..', 'adult.zip')
        if os.path.exists(default_zip):
            zip_path = default_zip

    if zip_path is not None and os.path.exists(zip_path):
        # _load_from_local_zip will return X, y for combined data when previously used;
        # we need to read train (adult.data) and test (adult.test) separately here.
        import zipfile

        colnames = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]

        with zipfile.ZipFile(zip_path, 'r') as z:
            if 'adult.data' in z.namelist():
                with z.open('adult.data') as f:
                    df_train = pd.read_csv(f, header=None, names=colnames, na_values='?', skipinitialspace=True)
            else:
                raise FileNotFoundError('adult.data not found inside zip')

            if 'adult.test' in z.namelist():
                with z.open('adult.test') as f:
                    df_test = pd.read_csv(f, header=None, names=colnames, na_values='?', skiprows=1, skipinitialspace=True)
            else:
                df_test = None

        # clean labels
        df_train['income'] = df_train['income'].astype(str).str.replace('.', '', regex=False).str.strip()
        if df_test is not None:
            df_test['income'] = df_test['income'].astype(str).str.replace('.', '', regex=False).str.strip()

        # split df_train into train/val first (we will fit preprocessors on train only)
        df_train_part, df_val_part = train_test_split(df_train, test_size=val_size, random_state=42)

        # columns to drop or treat specially
        # Keep all original columns by default (no explicit drops)
        drop_cols = []

        # numeric columns to scale (keep numeric education-num, drop textual education)
        num_cols = [
            'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ]

        # The UCI Adult dataset documents missing values in workclass and occupation.
        # We drop 'native-country' and 'race' as low-signal geographic/demographic features to simplify the model.
        # Treat '?' as missing and create explicit missing-indicator flags for the remaining known missingable categoricals.
        missing_categorical = [c for c in ['workclass', 'occupation'] if c in df_train_part.columns]

        # keep the textual 'education' column as well (we'll let one-hot encoding handle it)

        # normalize and log-transform capital columns to reduce skew before scaling
        for col in ['capital-gain', 'capital-loss']:
            if col in df_train_part.columns:
                df_train_part[col] = pd.to_numeric(df_train_part[col], errors='coerce').fillna(0)
                df_val_part[col] = pd.to_numeric(df_val_part[col], errors='coerce').fillna(0)
                # log1p transform
                df_train_part[col] = np.log1p(df_train_part[col])
                df_val_part[col] = np.log1p(df_val_part[col])

        # create missing-indicator flags for known missingable categoricals
        flag_cols = []
        for c in missing_categorical:
            flag = f"{c}_missing"
            flag_cols.append(flag)
            df_train_part[flag] = df_train_part[c].replace('?', np.nan).isna().astype(int)
            df_val_part[flag] = df_val_part[c].replace('?', np.nan).isna().astype(int)

        # If caller requested pure one-hot preprocessing, disable aggregation/target-encoding/PCA
        if one_hot_only:
            aggregate_low_signal = False
            target_encode_cols = None
            pca_components = None
            feature_selection = None

        # ---------- Target-encoding (k-fold smoothed) for selected categorical columns ----------
        te_mappings = {}
        if aggregate_low_signal and target_encode_cols:
            from sklearn.model_selection import KFold
            global_mean = df_train_part['income'].astype(str).map({'<=50K':0, '>50K':1}).astype(float).mean()
            kf = KFold(n_splits=te_kfold, shuffle=True, random_state=42)
            for col in target_encode_cols:
                if col not in df_train_part.columns:
                    continue
                # create out-of-fold encoded column in train
                te_col = f"{col}_te"
                df_train_part[te_col] = np.nan
                vals = df_train_part[col].replace('?', np.nan).fillna('Missing').values
                yvals = df_train_part['income'].astype(str).map({'<=50K':0, '>50K':1}).astype(float).values
                for train_idx, oof_idx in kf.split(vals):
                    tr_vals = vals[train_idx]
                    tr_y = yvals[train_idx]
                    tr_df = pd.DataFrame({col: tr_vals, 'y': tr_y})
                    stats = tr_df.groupby(col)['y'].agg(['count','mean'])
                    # smoothing
                    counts = stats['count']
                    means = stats['mean']
                    smooth = (counts * means + te_prior * global_mean) / (counts + te_prior)
                    mapping = smooth.to_dict()
                    # fill oof
                    for i in oof_idx:
                        key = vals[i]
                        if key in mapping:
                            df_train_part.at[df_train_part.index[i], te_col] = mapping[key]
                        else:
                            df_train_part.at[df_train_part.index[i], te_col] = global_mean
                # for val/test we'll compute full-train mapping later
                # compute full-train smoothed mapping to use on val/test
                full_df = pd.DataFrame({col: vals, 'y': yvals})
                stats_full = full_df.groupby(col)['y'].agg(['count','mean'])
                counts = stats_full['count']
                means = stats_full['mean']
                smooth_full = (counts * means + te_prior * global_mean) / (counts + te_prior)
                te_mappings[col] = {'mapping': smooth_full.to_dict(), 'global_mean': global_mean}

        # apply target-encoding to validation (use full-train mapping)
        if aggregate_low_signal and target_encode_cols:
            for col in target_encode_cols:
                if col not in df_val_part.columns:
                    continue
                te_col = f"{col}_te"
                mapping = te_mappings.get(col, {}).get('mapping', {})
                gmean = te_mappings.get(col, {}).get('global_mean', 0.0)
                df_val_part[te_col] = df_val_part[col].replace('?', np.nan).fillna('Missing').map(lambda x: mapping.get(x, gmean))

        # handle categorical columns: determine top categories on train and map rare -> 'Other'
        # Treat '?' as missing and fold into 'Missing' token before top-k selection
        cat_cols = [c for c in df_train_part.columns if c not in num_cols + ['income'] + drop_cols + flag_cols]
        top_k = 10
        top_categories = {}
        for c in cat_cols:
            top = df_train_part[c].replace('?', np.nan).fillna('Missing').value_counts().nlargest(top_k).index.tolist()
            top_categories[c] = top
            # if this column was target-encoded, keep original but also prepare for mapping; we'll drop originals later if requested
            df_train_part[c] = df_train_part[c].replace('?', np.nan).fillna('Missing').apply(lambda x: x if x in top else 'Other')
            df_val_part[c] = df_val_part[c].replace('?', np.nan).fillna('Missing').apply(lambda x: x if x in top else 'Other')
            # apply target-encoding to train/val if requested for this column
            if aggregate_low_signal and target_encode_cols and (c in target_encode_cols):
                # train already has te_col filled via OOF above; ensure val has been created earlier
                pass

        # drop unwanted columns
        if drop_cols:
            df_train_part = df_train_part.drop(columns=drop_cols)
            df_val_part = df_val_part.drop(columns=drop_cols)

        # ensure numeric columns are numeric and fill missing with median (fit on train)
        for c in num_cols:
            if c in df_train_part.columns:
                df_train_part[c] = pd.to_numeric(df_train_part[c], errors='coerce')
                df_val_part[c] = pd.to_numeric(df_val_part[c], errors='coerce')
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        present_num_cols = [c for c in num_cols if c in df_train_part.columns]
        if present_num_cols:
            df_train_part[present_num_cols] = df_train_part[present_num_cols].fillna(df_train_part[present_num_cols].median())
            scaler.fit(df_train_part[present_num_cols])
            df_train_part[present_num_cols] = scaler.transform(df_train_part[present_num_cols])
            df_val_part[present_num_cols] = df_val_part[present_num_cols].fillna(df_train_part[present_num_cols].median())
            df_val_part[present_num_cols] = scaler.transform(df_val_part[present_num_cols])

        # one-hot encode categorical columns (keep target-encoded numeric columns in dataframe)
        df_train_enc = pd.get_dummies(df_train_part.drop(columns=['income']))
        df_val_enc = pd.get_dummies(df_val_part.drop(columns=['income']))

        # if we created target-encoded columns for train but they are missing from val due to column ordering, ensure presence
        if aggregate_low_signal and target_encode_cols:
            for col in target_encode_cols:
                te_col = f"{col}_te"
                if te_col in df_train_enc.columns and te_col not in df_val_enc.columns:
                    df_val_enc[te_col] = df_val_part.get(te_col, np.nan).fillna(te_mappings.get(col, {}).get('global_mean', 0.0))

        # align validation columns to train columns
        df_val_enc = df_val_enc.reindex(columns=df_train_enc.columns, fill_value=0)

        # fit a single label encoder on train+val to ensure consistent mapping
        le = LabelEncoder()
        le.fit(pd.concat([df_train_part['income'], df_val_part['income']]).astype(str))
        y_train = le.transform(df_train_part['income'].astype(str))
        y_val = le.transform(df_val_part['income'].astype(str))

        # Optionally compute per-feature Pearson correlation on train and compress low-signal block with PCA
        kept_cols = list(df_train_enc.columns)
        feature_correlations = {}
        if feature_selection == 'pearson' or (aggregate_low_signal and pca_components and pca_components > 0):
            # compute pearson correlations between each column and the binary target
            y_train_numeric = pd.Series(df_train_part['income'].astype(str).map({'<=50K':0, '>50K':1}).astype(float).values, index=df_train_part.index)
            import numpy as _np
            pearson = {}
            for col in kept_cols:
                try:
                    xi = df_train_enc[col].values.astype(float)
                    if _np.std(xi) == 0 or _np.std(y_train_numeric.values) == 0:
                        corr = 0.0
                    else:
                        corr = float(_np.corrcoef(xi, y_train_numeric.values)[0,1])
                except Exception:
                    corr = 0.0
                pearson[col] = corr
            # store correlations
            feature_correlations = pearson

            # select kept columns by threshold if requested
            if feature_selection == 'pearson' and feature_threshold is not None and feature_threshold > 0:
                kept_cols = [c for c, v in pearson.items() if abs(v) >= feature_threshold]

            # compress remaining low-signal columns with PCA if requested
            if aggregate_low_signal and pca_components and pca_components > 0:
                low_signal_cols = [c for c in df_train_enc.columns if c not in kept_cols]
                if len(low_signal_cols) > 0:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(pca_components, len(low_signal_cols)), random_state=42)
                    low_block = df_train_enc[low_signal_cols].values.astype(float)
                    try:
                        pca.fit(low_block)
                        pcs_train = pca.transform(low_block)
                        pcs_val = pca.transform(df_val_enc[low_signal_cols].values.astype(float))
                        # attach PCA components to train/val enc
                        for i in range(pcs_train.shape[1]):
                            name = f'pca_low_{i}'
                            df_train_enc[name] = pcs_train[:, i]
                            df_val_enc[name] = pcs_val[:, i]
                        # drop original low-signal columns from both
                        df_train_enc = df_train_enc.drop(columns=low_signal_cols)
                        df_val_enc = df_val_enc.drop(columns=low_signal_cols, errors='ignore')
                        # add pca names to kept_cols
                        kept_cols = [c for c in df_train_enc.columns]
                    except Exception:
                        # if PCA fails, keep original columns
                        pca = None
                else:
                    pca = None
            else:
                pca = None
        else:
            pca = None

        X_train = df_train_enc.values.astype('float32')
        X_val = df_val_enc.values.astype('float32')

        # prepare test from df_test if available
        if df_test is not None:
            df_test_proc = df_test.copy()
            if drop_cols:
                df_test_proc = df_test_proc.drop(columns=drop_cols)
            # numeric capital columns: coerce, fill, log1p
            for c in ['capital-gain', 'capital-loss']:
                if c in df_test_proc.columns:
                    df_test_proc[c] = pd.to_numeric(df_test_proc[c], errors='coerce').fillna(0)
                    df_test_proc[c] = np.log1p(df_test_proc[c])
            # create missing flags for known missingable categoricals
            for c in missing_categorical:
                flag = f"{c}_missing"
                if c in df_test_proc.columns:
                    df_test_proc[flag] = df_test_proc[c].replace('?', np.nan).isna().astype(int)
            # apply top categories mapping (use 'Missing' token like train)
            for c in cat_cols:
                if c in df_test_proc.columns:
                    tops = top_categories.get(c, [])
                    df_test_proc[c] = df_test_proc[c].replace('?', np.nan).fillna('Missing').apply(lambda x: x if x in tops else 'Other')
            # numeric scaling
            if present_num_cols:
                df_test_proc[present_num_cols] = df_test_proc[present_num_cols].fillna(df_train_part[present_num_cols].median())
                df_test_proc[present_num_cols] = scaler.transform(df_test_proc[present_num_cols])
            df_test_enc = pd.get_dummies(df_test_proc.drop(columns=['income']))
            df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)
            X_test_vals = df_test_enc.values.astype('float32')
            # use same label encoder as used for train/val
            y_test_vals = le.transform(df_test['income'].astype(str))
        else:
            X_test_vals = None
            y_test_vals = None

        # save preprocessing artifacts to data/preproc_adult.pkl
        import pickle
        preproc = {
            'scaler': scaler,
            'train_columns': df_train_enc.columns.tolist(),
            'top_categories': top_categories,
            'num_cols': present_num_cols,
            'label_classes': le.classes_.tolist(),
            'missing_categorical': missing_categorical,
            'flag_cols': flag_cols,
            'dropped_columns': drop_cols,
            'kept_columns': kept_cols,
            'feature_correlations': feature_correlations,
            'target_encode_mappings': te_mappings,
            'pca_object': pca,
        }
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'preproc_adult.pkl'), 'wb') as f:
            pickle.dump(preproc, f)

        # At this point, X_train, X_val, and X_test (if available) have been prepared above.
        # If no separate df_test was present, X_test_vals/y_test_vals remain None and will be
        # set to validation data later to ensure tensors are returned.

        # allow max_samples on training set
        if max_samples is not None:
            n = min(max_samples, X_train.shape[0])
            X_train = X_train[:n]
            y_train = y_train[:n]

        # convert to tensors and loaders
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test_vals, dtype=torch.float32) if X_test_vals is not None else torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test_vals, dtype=torch.long) if y_test_vals is not None else torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        # fetch from openml and do a 3-way split
        X_vals, y_vals = _load_from_openml()
        if max_samples is not None:
            n = min(max_samples, X_vals.shape[0])
            X_vals = X_vals[:n]
            y_vals = y_vals[:n]

        # first split into train and temp (val+test)
        combined = val_size + test_size
        if combined <= 0 or combined >= 1.0:
            raise ValueError('val_size + test_size must be between 0 and 1')
        X_train, X_temp, y_train, y_temp = train_test_split(X_vals, y_vals, test_size=combined, random_state=42)
        # split temp into val and test
        test_prop = test_size / combined
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_prop, random_state=42)

        X_train = torch.tensor(X_train.astype('float32'))
        X_val = torch.tensor(X_val.astype('float32'))
        X_test = torch.tensor(X_test.astype('float32'))
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
