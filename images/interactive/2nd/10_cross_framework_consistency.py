"""Cross-engine state consistency between Isaac Sim and MuJoCo.

Paired scatter plot showing joint position consistency between the two
simulation engines, with perfect agreement along the diagonal and
annotated drift regions where state divergence exceeds thresholds.
"""

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_cross_framework_consistency(dark_mode=False):
    """Create paired scatter plot for cross-engine state consistency.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with consistency scatter plot.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    n_samples = 600
    joint_names = ["J1 Base", "J2 Shoulder", "J3 Elbow", "J4 Wrist1", "J5 Wrist2", "J6 Wrist3"]
    joint_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#FF6692"]

    fig = go.Figure()

    # Perfect agreement diagonal line
    diag_range = np.linspace(-3.14, 3.14, 100)
    fig.add_trace(
        go.Scatter(
            x=diag_range,
            y=diag_range,
            mode="lines",
            name="Perfect Agreement",
            line=dict(color="gray", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Drift tolerance bands
    fig.add_trace(
        go.Scatter(
            x=diag_range,
            y=diag_range + 0.05,
            mode="lines",
            line=dict(color="rgba(255,161,90,0.3)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=diag_range,
            y=diag_range - 0.05,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,161,90,0.08)",
            line=dict(color="rgba(255,161,90,0.3)", width=0),
            name="\u00b10.05 rad tolerance",
            hoverinfo="skip",
        )
    )

    all_isaac = []
    all_mujoco = []
    all_drifts = []

    for i, (joint, color) in enumerate(zip(joint_names, joint_colors)):
        n_per_joint = n_samples // len(joint_names)

        # Isaac Sim joint positions (ground truth)
        isaac_pos = np.random.uniform(-2.5, 2.5, n_per_joint)

        # MuJoCo positions: mostly consistent with some drift
        # Drift increases at extreme joint positions
        base_noise = np.random.normal(0, 0.015, n_per_joint)
        position_dependent_drift = 0.02 * np.abs(isaac_pos) * np.random.normal(0, 1, n_per_joint)

        # Add systematic drift for some joints
        systematic_offset = 0.0
        if joint in ["J3 Elbow", "J4 Wrist1"]:
            systematic_offset = 0.03 * np.sign(isaac_pos)

        mujoco_pos = isaac_pos + base_noise + position_dependent_drift + systematic_offset

        # Inject a few outlier drift points
        n_outliers = max(2, int(n_per_joint * 0.03))
        outlier_idx = np.random.choice(n_per_joint, n_outliers, replace=False)
        mujoco_pos[outlier_idx] += np.random.normal(0, 0.15, n_outliers)

        drift = np.abs(mujoco_pos - isaac_pos)
        all_isaac.extend(isaac_pos)
        all_mujoco.extend(mujoco_pos)
        all_drifts.extend(drift)

        fig.add_trace(
            go.Scatter(
                x=isaac_pos,
                y=mujoco_pos,
                mode="markers",
                name=joint,
                marker=dict(
                    color=color,
                    size=5,
                    opacity=0.6,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"<b>{joint}</b><br>"
                    "Isaac: %{x:.4f} rad<br>"
                    "MuJoCo: %{y:.4f} rad<br>"
                    f"Drift: %{{customdata:.4f}} rad<extra></extra>"
                ),
                customdata=drift,
            )
        )

    all_isaac = np.array(all_isaac)
    all_mujoco = np.array(all_mujoco)
    all_drifts = np.array(all_drifts)

    # Highlight drift regions where error > 0.1 rad
    high_drift_mask = all_drifts > 0.1
    if np.any(high_drift_mask):
        fig.add_trace(
            go.Scatter(
                x=all_isaac[high_drift_mask],
                y=all_mujoco[high_drift_mask],
                mode="markers",
                name="High Drift (>0.1 rad)",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=12,
                    line=dict(color="red", width=2),
                    symbol="circle-open",
                ),
                hovertemplate=("<b>HIGH DRIFT</b><br>Isaac: %{x:.4f} rad<br>MuJoCo: %{y:.4f} rad<br><extra></extra>"),
            )
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="Cross-Engine State Consistency: Isaac Sim vs MuJoCo Joint Positions",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Isaac Sim Joint Position (rad)",
            range=[-3.2, 3.2],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="MuJoCo Joint Position (rad)",
            range=[-3.2, 3.2],
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        width=1920,
        height=1080,
        margin=dict(l=80, r=60, t=100, b=100),
    )

    # Summary statistics
    rmse = np.sqrt(np.mean(all_drifts**2))
    mae = np.mean(all_drifts)
    max_drift = np.max(all_drifts)
    pct_within = (all_drifts <= 0.05).sum() / len(all_drifts) * 100

    fig.add_annotation(
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        text=(
            f"RMSE: {rmse:.4f} rad | MAE: {mae:.4f} rad | "
            f"Max Drift: {max_drift:.4f} rad | Within \u00b10.05 rad: {pct_within:.1f}%"
        ),
        showarrow=False,
        font=dict(size=12, color="gray"),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_cross_framework_consistency(dark_mode=False)
    fig.write_html(str(output_dir / "10_cross_framework_consistency.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "10_cross_framework_consistency.png"), width=1920, height=1080, scale=2)
    print("Saved 10_cross_framework_consistency.html and 10_cross_framework_consistency.png")
