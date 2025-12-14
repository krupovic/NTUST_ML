import random
from pathlib import Path
from typing import List

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_imports import *  # reuse centralized imports (numpy already included)

PREFIX_LEN = 20
SAMPLE_MIN = 5
SAMPLE_MAX = 8
CMAP = plt.get_cmap("tab10")

TRAIN_PATH = Path("train_processed.csv")


def _parse_polyline(poly_str):
    if isinstance(poly_str, str):
        try:
            return json.loads(poly_str)
        except json.JSONDecodeError:
            return []
    return poly_str if isinstance(poly_str, list) else []


def sample_rows(df, seed=42):
    rng = random.Random(seed)
    k = rng.randint(SAMPLE_MIN, SAMPLE_MAX)
    return df.sample(n=min(k, len(df)), random_state=seed).reset_index(drop=True)


def extract_prefix(poly, prefix_len=PREFIX_LEN):
    prefix = poly[:prefix_len]
    if prefix:
        start = np.array(prefix[0], dtype=float)
        end = np.array(prefix[-1], dtype=float)
    else:
        start = np.zeros(2, dtype=float)
        end = np.zeros(2, dtype=float)
    return prefix, start, end


def plot_sampled_trajectories(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax_traj, ax_delta = axes

    colors = [CMAP(i % CMAP.N) for i in range(len(df))]
    lon_vals: List[float] = []
    lat_vals: List[float] = []

    for idx, row in df.iterrows():
        poly = _parse_polyline(row["POLYLINE"])
        if len(poly) < 2:
            continue

        poly_arr = np.array(poly, dtype=float)
        ax_traj.plot(
            poly_arr[:, 0],
            poly_arr[:, 1],
            linewidth=1.0,
            alpha=0.7,
            color=colors[idx],
            label=f"Trip {row['TRIP_ID']}",
        )
        lon_vals.extend(poly_arr[:, 0].tolist())
        lat_vals.extend(poly_arr[:, 1].tolist())

        prefix, start, end = extract_prefix(poly)
        if len(prefix) >= 1:
            prefix_arr = np.array(prefix, dtype=float)
            ax_traj.plot(
                prefix_arr[:, 0],
                prefix_arr[:, 1],
                linewidth=2.0,
                alpha=0.9,
                color=colors[idx],
            )
            ax_traj.scatter(
                start[0],
                start[1],
                color=colors[idx],
                marker="o",
                s=50,
                edgecolor="white",
                zorder=5,
            )
            ax_traj.scatter(
                end[0],
                end[1],
                color=colors[idx],
                marker="X",
                s=70,
                edgecolor="white",
                zorder=5,
            )
            lon_vals.extend(prefix_arr[:, 0].tolist())
            lat_vals.extend(prefix_arr[:, 1].tolist())

            ax_delta.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=colors[idx],
                linewidth=2.0,
                alpha=0.9,
            )
            ax_delta.scatter(
                start[0],
                start[1],
                color=colors[idx],
                marker="o",
                s=50,
                edgecolor="white",
                zorder=5,
            )
            ax_delta.scatter(
                end[0],
                end[1],
                color=colors[idx],
                marker="X",
                s=70,
                edgecolor="white",
                zorder=5,
            )
            ax_delta.arrow(
                start[0],
                start[1],
                (end - start)[0],
                (end - start)[1],
                color=colors[idx],
                width=0.00005,
                length_includes_head=True,
                head_width=0.001,
                head_length=0.001,
                alpha=0.9,
            )

    if lon_vals and lat_vals:
        lon_min, lon_max = min(lon_vals), max(lon_vals)
        lat_min, lat_max = min(lat_vals), max(lat_vals)
        lon_margin = max(1e-3, (lon_max - lon_min) * 0.05)
        lat_margin = max(1e-3, (lat_max - lat_min) * 0.05)
        x_bounds = (lon_min - lon_margin, lon_max + lon_margin)
        y_bounds = (lat_min - lat_margin, lat_max + lat_margin)
    else:
        x_bounds = (-8.75, -8.50)
        y_bounds = (41.10, 41.25)

    for ax in axes:
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax_traj.set_title("Sampled Trajectories (Prefix Highlighted)")
    ax_delta.set_title("Prefix Delta Vectors (20 Points)")
    ax_traj.legend(loc="upper right", fontsize=8)

    output_path = Path("partial_trajectory_sample.png")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}, run data_preparation first.")

    df = pd.read_csv(TRAIN_PATH)
    sampled = sample_rows(df)
    plot_sampled_trajectories(sampled)


if __name__ == "__main__":
    main()
