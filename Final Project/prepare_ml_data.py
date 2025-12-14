import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PREFIX_LEN = 20


def _parse_polyline(polyline) -> List[List[float]]:
    """Parse a POLYLINE field into a list of coordinates."""
    if isinstance(polyline, str):
        try:
            return json.loads(polyline)
        except json.JSONDecodeError:
            return []
    return polyline if isinstance(polyline, list) else []


def add_missing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary indicator for rows tagged with missing data."""
    result = df.copy()
    result["missing_flag"] = (
        result["MISSING_DATA"].astype(str).str.lower().apply(lambda value: 0 if value == "false" else 1)
    )
    return result


def add_destination_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive destination coordinates from the full trajectory (train only)."""
    result = df.copy()
    lon_dest: List[float] = []
    lat_dest: List[float] = []

    for poly in result["POLYLINE"]:
        coords = _parse_polyline(poly)
        if coords:
            lon_dest.append(coords[-1][0])
            lat_dest.append(coords[-1][1])
        else:
            lon_dest.append(None)
            lat_dest.append(None)

    result["lon_dest"] = lon_dest
    result["lat_dest"] = lat_dest
    return result


def add_polyline_prefix(df: pd.DataFrame, prefix_len: int = PREFIX_LEN) -> pd.DataFrame:
    """Attach a flattened prefix (lon/lat pairs) derived from initial trajectory points."""
    result = df.copy()
    vectors: List[np.ndarray] = []
    displacements: List[List[float]] = []
    lengths: List[int] = []

    for poly in result["POLYLINE"]:
        coords = _parse_polyline(poly)
        prefix = coords[:prefix_len]

        if prefix:
            last_point = prefix[-1]
            start_point = prefix[0]
        else:
            last_point = [0.0, 0.0]
            start_point = [0.0, 0.0]

        # pad to fixed length using the last observed point (or zeros if none)
        padded = prefix + [last_point] * max(0, prefix_len - len(prefix))
        padded = padded[:prefix_len]

        flat: List[float] = []
        for lon, lat in padded:
            flat.extend([float(lon), float(lat)])

        vectors.append(np.array(flat, dtype=float))
        lengths.append(len(prefix))
        disp_lon = float(last_point[0]) - float(start_point[0])
        disp_lat = float(last_point[1]) - float(start_point[1])
        displacements.append([disp_lon, disp_lat])

    result["polyline_prefix"] = vectors
    result["prefix_length"] = lengths
    result["prefix_disp_lon"] = [disp[0] for disp in displacements]
    result["prefix_disp_lat"] = [disp[1] for disp in displacements]
    return result


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure the requested columns exist, adding zero-filled placeholders if needed."""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            result[col] = 0
    return result[columns]


def prepare_ml_data(
    train_path: str,
    test_path: str,
    sample_size: int = 136000,
    test_split: float = 0.3,
    random_state: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed tables, build supervised datasets, and produce a validation split."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 1) retain only signals available before trip completion
    train = add_missing_flag(train)
    test = add_missing_flag(test)

    # 2) recover destination coordinates for training labels
    train = add_destination_targets(train)
    train = train.dropna(subset=["lon_dest", "lat_dest"])

    # 3) generate fixed-length prefix representations for both splits
    train = add_polyline_prefix(train, PREFIX_LEN)
    test = add_polyline_prefix(test, PREFIX_LEN)

    # optional downsampling for faster experimentation
    ml_train = train.sample(sample_size, random_state=random_state) if sample_size < len(train) else train.copy()
    ml_test = test.copy()

    # 4) select low-collinearity calendar features, call-type indicators, prefix vector, and missing flag
    feature_cols = [
        "call_type_a",
        "call_type_b",
        "call_type_c",
        "month",
        "day",
        "hour",
        "min",
        "weekday",
        "missing_flag",
        "prefix_length",
        "prefix_disp_lon",
        "prefix_disp_lat",
        "polyline_prefix",
    ]
    target_cols = ["lon_dest", "lat_dest"]

    X = _ensure_columns(ml_train, feature_cols)
    y = ml_train[target_cols]
    X_test_full = _ensure_columns(ml_test, feature_cols)

    def expand_prefix(df: pd.DataFrame) -> pd.DataFrame:
        prefix_vectors = [np.asarray(vec, dtype=float) for vec in df.pop("polyline_prefix")]
        prefix_matrix = np.vstack(prefix_vectors)
        prefix_columns = [f"poly_prefix_{i}" for i in range(prefix_matrix.shape[1])]
        prefix_df = pd.DataFrame(prefix_matrix, index=df.index, columns=prefix_columns)
        return pd.concat([df, prefix_df], axis=1)

    X = expand_prefix(X)
    X_test_full = expand_prefix(X_test_full)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    return X_train, X_val, y_train, y_val, X_test_full
