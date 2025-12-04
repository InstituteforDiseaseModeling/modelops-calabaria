"""Diagnostic report generation for ModelOps optimization results.

This module provides visualization and analysis tools for parameter optimization
results, including loss landscapes, profiles, and MLE summaries.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2
import typer


def detect_param_cols(df: pl.DataFrame) -> List[str]:
    """Detect parameter columns in the DataFrame."""
    # Currently just ignore param_id (special identifier column)
    # TODO: After changing ModelOps to use param___ naming, update this logic

    # Find param_* columns, excluding param_id
    cols = [c for c in df.columns
            if c.startswith("param_") and c != "param_id"]

    # Filter to only numeric columns
    numeric_cols = []
    for col in cols:
        try:
            # Check if column can be cast to float
            df.select(pl.col(col).cast(pl.Float64)).head(1)
            numeric_cols.append(col)
        except:
            # Skip non-numeric columns
            pass

    if not numeric_cols:
        raise ValueError("No numeric parameter columns found (expected param_* columns with numeric values).")

    return sorted(numeric_cols)  # Sort for consistent ordering


def compute_dloss(df: pl.DataFrame) -> Tuple[pl.DataFrame, float]:
    """Compute Δloss relative to minimum loss."""
    loss_min = df.select(pl.col("loss").min()).item()
    df2 = df.with_columns((pl.col("loss") - loss_min).alias("dloss"))
    return df2, float(loss_min)


def mle_row(df: pl.DataFrame, param_cols: List[str]) -> Tuple[np.ndarray, float, dict]:
    """Find minimum loss row and extract parameters."""
    row = df.sort("loss").row(0, named=True)
    x_opt = np.array([row[c] for c in param_cols], dtype=float)
    loss_val = float(row["loss"])
    return x_opt, loss_val, row


def scaled_params(df: pl.DataFrame, param_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale parameters to [0,1] range for distance calculations."""
    X = df.select(param_cols).to_numpy()
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = np.where(maxs > mins, maxs - mins, 1.0)
    Z = (X - mins) / rng
    return Z, mins, rng


def scaled_distance_other_dims(
    X_scaled: np.ndarray,
    x0_scaled: np.ndarray,
    drop_index: int
) -> np.ndarray:
    """Euclidean distance in scaled space, excluding one focal dimension."""
    Z = X_scaled - x0_scaled[None, :]
    if Z.shape[1] == 1:
        return np.zeros(Z.shape[0])
    keep = [k for k in range(Z.shape[1]) if k != drop_index]
    return np.linalg.norm(Z[:, keep], axis=1)


def choose_informative_pairs(
    df: pl.DataFrame,
    param_cols: List[str],
    top_k_params: int = 3
) -> List[Tuple[int, int]]:
    """Pick parameter pairs with highest correlation to Δloss for visualization."""
    y = df["dloss"].to_numpy()
    corrs = []
    for i, p in enumerate(param_cols):
        x = df[p].to_numpy()
        if np.all(x == x[0]):
            corrs.append((i, 0.0))
        else:
            c = np.corrcoef(x, y)[0, 1]
            if not np.isfinite(c):
                c = 0.0
            corrs.append((i, abs(c)))

    corrs.sort(key=lambda t: t[1], reverse=True)
    idxs = [i for i, _ in corrs[:max(2, min(top_k_params, len(param_cols)))]]

    pairs = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            pairs.append((idxs[a], idxs[b]))
    return pairs


def tri_cube(u: np.ndarray) -> np.ndarray:
    """Tri-cube kernel for weighted local regression."""
    u = np.clip(u, 0.0, 1.0)
    return (1 - u**3)**3


def profile_1d_clean(
    df: pl.DataFrame,
    param_cols: List[str],
    focal: str,
    x_mle: np.ndarray,
    mins: np.ndarray,
    rng: np.ndarray,
    knn_other: int = 400,
    nbins: int = 120,
    x_span_pad: float = 0.0,
    min_pts_per_x: int = 12,
    widen_factor: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Robust 1D profile:
    - KNN in other dims to stay near the MLE neighborhood.
    - Adaptive HARD x-window (not quantile-on-weights) so we never 'see' the MLE when x is far.
    - Binwise min, then gentle spline; anchored at (x_MLE, 0).
    """
    j = param_cols.index(focal)
    X = df.select(param_cols).to_numpy()
    y = df["dloss"].to_numpy().astype(float)

    # KNN in other dims
    Xs = (X - mins) / np.where(rng > 0, rng, 1.0)
    z0 = (x_mle - mins) / np.where(rng > 0, rng, 1.0)
    d_other = scaled_distance_other_dims(Xs, z0, drop_index=j)

    k = min(knn_other, len(y))
    idx = np.argsort(d_other)[:k]
    xk, yk = X[idx, j], y[idx]

    # Choose x-grid and a *data-driven* half-window based on median spacing
    ux = np.unique(np.sort(xk))
    if len(ux) > 1:
        dx_med = np.median(np.diff(ux))
    else:
        dx_med = max(np.ptp(xk) / 50, 1e-8)
    half = max(1.5 * dx_med, 0.01 * (ux[-1] - ux[0]) if len(ux) > 0 else 1e-8)

    xs = np.linspace(xk.min(), xk.max(), nbins)
    prof_x, prof_y = [], []

    for xc in xs:
        h = half
        # adapt window until we have enough points locally in x
        for _ in range(6):  # cap expansions
            mask = np.abs(xk - xc) <= h
            if mask.sum() >= min_pts_per_x:
                break
            h *= widen_factor
        if mask.sum() < max(3, min_pts_per_x // 2):
            # insufficient coverage: skip this xc (avoids spurious zeros)
            continue

        # local conditional min
        y_min = yk[mask].min()
        prof_x.append(xc)
        prof_y.append(max(0.0, y_min))

    # Anchor at MLE (x_mle_j, 0)
    x_mle_j = x_mle[j]
    prof_x.append(x_mle_j)
    prof_y.append(0.0)

    prof_x = np.asarray(prof_x)
    prof_y = np.asarray(prof_y)

    # Smooth (fit on sorted x), then evaluate on the same sorted grid
    if len(prof_x) >= 7:
        s = max(len(prof_x) * 0.002, 1e-9)
        order = np.argsort(prof_x)
        x_sorted = prof_x[order]
        y_sorted = prof_y[order]
        try:
            spl = UnivariateSpline(x_sorted, y_sorted, s=s)
            y_smooth = np.maximum(0.0, spl(x_sorted))
            prof_x, prof_y = x_sorted, y_smooth
        except Exception:
            prof_x, prof_y = x_sorted, y_sorted
    else:
        # At least return in ascending x
        order = np.argsort(prof_x)
        prof_x, prof_y = prof_x[order], prof_y[order]

    # Insert NaNs to avoid drawing lines across unsupported gaps
    if len(prof_x) > 2:
        dx = np.diff(prof_x)
        dxm = np.median(dx[dx > 0]) if np.any(dx > 0) else 0.0
        if dxm > 0:
            gap = 4 * dxm
            xs, ys = [prof_x[0]], [prof_y[0]]
            for i in range(1, len(prof_x)):
                if prof_x[i] - prof_x[i-1] > gap:
                    xs.extend([np.nan])
                    ys.extend([np.nan])  # break the polyline
                xs.append(prof_x[i])
                ys.append(prof_y[i])
            prof_x, prof_y = np.array(xs), np.array(ys)

    # Return (sorted, gap-broken) profile and the local KNN points used
    return prof_x, prof_y, (xk, yk)


def try_pivot_pair_to_grid(
    df: pl.DataFrame,
    a: str,
    b: str,
    v: str = "dloss",
):
    """Attempt to pivot data to 2D grid for clean contour plots."""
    sub = df.select([a, b, v])

    # Get unique values for each axis
    xa_vals = sub[a].unique().sort()
    xb_vals = sub[b].unique().sort()
    xa = xa_vals.to_numpy()
    xb = xb_vals.to_numpy()

    # Try to pivot using Polars
    try:
        piv = sub.pivot(values=v, index=b, columns=a, aggregate_function="first")

        # Check if we have a complete grid
        if len(piv) == len(xb):
            # Get column names (excluding index column)
            value_cols = [c for c in piv.columns if c != b]

            if len(value_cols) == len(xa):
                # Extract the values matrix, ensuring proper ordering
                Z = piv.sort(b).select(value_cols).to_numpy()

                # Check for NaNs
                if not np.isnan(Z).any():
                    return xa, xb, Z, True
    except:
        pass  # Fall back to scatter if pivot fails

    # Fall back to returning raw data for scatter plot
    return sub[a].to_numpy(), sub[b].to_numpy(), sub[v].to_numpy(), False


def page_overview(pdf: PdfPages,
                  df: pl.DataFrame,
                  param_cols: List[str],
                  x_opt: np.ndarray,
                  loss_min: float):
    """Generate overview page with loss rank, distance plot, and optimum summary."""
    y = df["dloss"].to_numpy()
    X_scaled, mins, rng = scaled_params(df, param_cols)
    z_opt = (x_opt - mins) / rng
    d = np.linalg.norm(X_scaled - z_opt[None, :], axis=1)

    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0], wspace=0.25)

    # Loss rank plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.sort(y), lw=1, color='darkblue')
    ax1.set_title("Δloss Rank", fontsize=12, fontweight='bold', loc='center')
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Δloss")
    ax1.grid(True, alpha=0.3)

    # Distance from optimum plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(d, y, s=8, alpha=0.4, c=y, cmap='viridis')
    ax2.set_title("Δloss vs Distance from Optimum", fontsize=12, fontweight='bold', loc='center')
    ax2.set_xlabel("Scaled Euclidean Distance")
    ax2.set_ylabel("Δloss")
    ax2.grid(True, alpha=0.3)

    # Optimum summary table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    # Format parameter names for display
    rows = [("Min Loss", f"{loss_min:,.3f}")]
    for i, p in enumerate(param_cols):
        param_name = p.replace("param_", "")
        rows.append((param_name, f"{x_opt[i]:.4g}"))

    tbl = ax3.table(cellText=rows,
                   colLabels=["Parameter", "Optimum Value"],
                   loc="center",
                   cellLoc='left')
    tbl.scale(1.0, 1.3)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax3.set_title("Optimum Summary", fontsize=12, fontweight='bold', loc='center')

    fig.suptitle("Optimization Overview", fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_profiles(pdf: PdfPages,
                  df: pl.DataFrame,
                  param_cols: List[str],
                  x_opt: np.ndarray,
                  mins: np.ndarray,
                  rng: np.ndarray):
    """Generate grid of 1D parameter profiles."""
    n = len(param_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4))
    if n == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).reshape(-1)

    for k, p in enumerate(param_cols):
        ax = axes[k]
        gx, gy, (rx, ry) = profile_1d_clean(df, param_cols, p, x_opt, mins, rng)

        # Plot profile and raw points
        ax.plot(gx, gy, lw=2, color='darkblue', label='Profile')
        ax.scatter(rx, ry, s=8, alpha=0.2, color='gray', label='Data')
        ax.axvline(x_opt[param_cols.index(p)], ls="--", lw=1, color='red', alpha=0.7, label='Optimum')

        # χ² reference line for 95% CI (Δloss = 1.92 for 1 param)
        ax.axhline(1.92, ls=":", lw=1, color='black', alpha=0.5, label='95% CI')

        param_name = p.replace("param_", "")
        ax.set_title(param_name, fontsize=11, fontweight='bold', loc='center')
        ax.set_xlabel("Value")
        ax.set_ylabel("Δloss")
        ax.grid(True, alpha=0.3)

        if k == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Remove empty subplots
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle("1D Parameter Profiles (conditioned at optimum)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_pair_contours(pdf: PdfPages,
                       df: pl.DataFrame,
                       param_cols: List[str],
                       x_opt: np.ndarray,
                       max_pairs: int = 6,
                       threshold: float = 0.0):
    """Generate 2D contour plots for informative parameter pairs."""
    pairs = choose_informative_pairs(df, param_cols, top_k_params=3)
    pairs = pairs[:max_pairs]

    if not pairs:
        return

    chunks = [pairs[i:i + 6] for i in range(0, len(pairs), 6)]
    for chunk in chunks:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        last_cs = None

        for ax_idx, pair in enumerate(chunk):
            ax = axes[ax_idx]
            i, j = pair

            a, b = param_cols[i], param_cols[j]
            xa, xb, Z, is_grid = try_pivot_pair_to_grid(df, a, b, "dloss")

            if is_grid:
                cs = ax.contourf(xa, xb, Z, levels=15, cmap='viridis', alpha=0.9)
                ax.contour(xa, xb, Z, levels=15, colors='black', alpha=0.2, linewidths=0.5)

                # Add threshold contour if specified
                if threshold > 0:
                    # Plot special contour at threshold
                    ax.contour(xa, xb, Z, levels=[threshold], colors='red',
                              linewidths=1.5, linestyles='solid', alpha=0.8)
                    # Also shade the region below threshold
                    ax.contourf(xa, xb, Z, levels=[0, threshold],
                               colors=['red'], alpha=0.2)
            else:
                cs = ax.tricontourf(xa, xb, Z, levels=15, cmap='viridis')

                if threshold > 0:
                    ax.tricontour(xa, xb, Z, levels=[threshold], colors='red',
                                 linewidths=1.5, linestyles='solid', alpha=0.8)

            last_cs = cs

            # Mark optimum with smaller star
            ax.scatter([x_opt[i]], [x_opt[j]], marker="*", s=150,
                      color='red', edgecolor="white", linewidth=1, zorder=10)

            ax.set_xlabel(a.replace("param_", ""), fontsize=10)
            ax.set_ylabel(b.replace("param_", ""), fontsize=10)
            ax.grid(True, alpha=0.2)

        # Remove unused axes
        for k in range(len(chunk), len(axes)):
            fig.delaxes(axes[k])

        title = "2D Loss Surfaces (optimum marked with star)"
        if threshold > 0:
            title += f"\nRed region: Δloss < {threshold:.1e}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if last_cs is not None:
            cbar = fig.colorbar(last_cs, ax=axes[:len(chunk)], fraction=0.02, pad=0.02)
            cbar.set_label('Δloss', rotation=270, labelpad=15)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def generate_report(
    input_path: Path,
    output_path: Optional[Path] = None,
    threshold: float = 0.0,
) -> Path:
    """
    Generate diagnostic report from ModelOps results.

    Args:
        input_path: Path to ModelOps results parquet file
        output_path: Optional output path for PDF (defaults to input_diagnostic_report.pdf)

    Returns:
        Path to generated PDF report
    """
    input_path = Path(input_path)

    # Default output name based on input
    if output_path is None:
        output_path = input_path.with_suffix('').with_name(
            f"{input_path.stem}_diagnostic_report.pdf"
        )
    else:
        output_path = Path(output_path)

    # Load data
    typer.echo(f"Loading data from {input_path}...")
    df = pl.read_parquet(str(input_path))

    # Validate required columns
    if "loss" not in df.columns:
        raise ValueError("Missing 'loss' column in parquet file")

    param_cols = detect_param_cols(df)
    typer.echo(f"Found {len(param_cols)} parameters: {', '.join(c.replace('param_', '') for c in param_cols[:5])}" +
               ("..." if len(param_cols) > 5 else ""))

    # Compute derived quantities
    df, loss_min = compute_dloss(df)
    x_opt, loss_val, opt_as_dict = mle_row(df, param_cols)
    X_scaled, mins, rng = scaled_params(df, param_cols)

    typer.echo(f"Minimum loss: {loss_min:.3f}")
    typer.echo(f"Generating diagnostic report...")

    # Create PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        # Add metadata
        d = pdf.infodict()
        d['Title'] = f'Diagnostic Report: {input_path.name}'
        d['Author'] = 'ModelOps Calabaria'
        d['Subject'] = 'Parameter Optimization Diagnostics'
        d['Creator'] = 'cb diagnostics report'

        # Generate pages
        page_overview(pdf, df, param_cols, x_opt, loss_min)
        page_profiles(pdf, df, param_cols, x_opt, mins, rng)
        page_pair_contours(pdf, df, param_cols, x_opt, max_pairs=6, threshold=threshold)

    file_size_kb = output_path.stat().st_size / 1024
    typer.echo(f"✓ Report saved to {output_path} ({file_size_kb:.1f} KB)")

    return output_path


def report_command(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ModelOps results parquet file"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output PDF path (defaults to <input>_diagnostic_report.pdf)"
    ),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Threshold for highlighting low-loss regions in red (e.g., 1e-5 for numeric error)"
    ),
) -> None:
    """Generate diagnostic report from ModelOps optimization results.

    Creates a comprehensive PDF report including:
    - Overview with optimum summary and loss landscape
    - 1D parameter profiles showing loss vs each parameter
    - 2D contour plots for informative parameter pairs

    The input should be a standard ModelOps results parquet file with
    'loss' column and 'param_*' parameter columns.
    """
    try:
        generate_report(input_file, output, threshold=threshold)
    except Exception as e:
        typer.echo(f"Error generating report: {e}", err=True)
        raise typer.Exit(1)