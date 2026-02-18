"""Sim-to-real transfer metrics for surgical robotics.

Dual-axis chart showing task success rate and position error across
transfer iterations, comparing direct transfer, fine-tuned, and
domain-adapted approaches for the oncology surgical robotics module.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_sim_to_real_transfer(dark_mode=False):
    """Create dual-axis chart showing sim-to-real transfer metrics.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with dual-axis transfer metrics.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    iterations = np.arange(0, 51)

    # Direct transfer: limited improvement, plateaus early
    direct_success = 45 + 20 * (1 - np.exp(-0.08 * iterations)) + np.random.normal(0, 1.5, len(iterations))
    direct_error = 12.0 * np.exp(-0.03 * iterations) + 5.5 + np.random.normal(0, 0.3, len(iterations))
    direct_success = np.clip(direct_success, 30, 100)

    # Fine-tuned: moderate improvement trajectory
    finetune_success = 40 + 40 * (1 - np.exp(-0.06 * iterations)) + np.random.normal(0, 1.2, len(iterations))
    finetune_error = 12.0 * np.exp(-0.05 * iterations) + 3.0 + np.random.normal(0, 0.25, len(iterations))
    finetune_success = np.clip(finetune_success, 30, 100)

    # Domain-adapted: best performance, strong convergence
    domain_success = 38 + 55 * (1 - np.exp(-0.07 * iterations)) + np.random.normal(0, 1.0, len(iterations))
    domain_error = 12.0 * np.exp(-0.08 * iterations) + 1.2 + np.random.normal(0, 0.2, len(iterations))
    domain_success = np.clip(domain_success, 30, 100)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Success rate traces (left Y-axis)
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=direct_success,
            mode="lines",
            name="Direct Transfer - Success %",
            line=dict(color="#636EFA", width=2.5),
            hovertemplate="Iter: %{x}<br>Success: %{y:.1f}%<extra>Direct</extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=finetune_success,
            mode="lines",
            name="Fine-Tuned - Success %",
            line=dict(color="#EF553B", width=2.5),
            hovertemplate="Iter: %{x}<br>Success: %{y:.1f}%<extra>Fine-Tuned</extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=domain_success,
            mode="lines",
            name="Domain-Adapted - Success %",
            line=dict(color="#00CC96", width=2.5),
            hovertemplate="Iter: %{x}<br>Success: %{y:.1f}%<extra>Domain-Adapted</extra>",
        ),
        secondary_y=False,
    )

    # Position error traces (right Y-axis) - dashed lines
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=direct_error,
            mode="lines",
            name="Direct Transfer - Error (mm)",
            line=dict(color="#636EFA", width=2, dash="dash"),
            hovertemplate="Iter: %{x}<br>Error: %{y:.2f} mm<extra>Direct</extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=finetune_error,
            mode="lines",
            name="Fine-Tuned - Error (mm)",
            line=dict(color="#EF553B", width=2, dash="dash"),
            hovertemplate="Iter: %{x}<br>Error: %{y:.2f} mm<extra>Fine-Tuned</extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=domain_error,
            mode="lines",
            name="Domain-Adapted - Error (mm)",
            line=dict(color="#00CC96", width=2, dash="dash"),
            hovertemplate="Iter: %{x}<br>Error: %{y:.2f} mm<extra>Domain-Adapted</extra>",
        ),
        secondary_y=True,
    )

    # Add clinical acceptability threshold
    fig.add_hline(
        y=90,
        line_dash="dot",
        line_color="#FFA15A",
        opacity=0.7,
        annotation_text="Clinical Acceptability Threshold (90%)",
        annotation_position="bottom right",
        secondary_y=False,
    )

    # Add position error threshold
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#FF6692",
        opacity=0.7,
        annotation_text="Precision Requirement (2mm)",
        annotation_position="top right",
        secondary_y=True,
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Sim-to-Real Transfer: Surgical Task Performance Across Strategies",
            font=dict(size=20),
        ),
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        width=1920,
        height=1080,
        margin=dict(l=80, r=80, t=100, b=80),
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Transfer Iteration", dtick=5)
    fig.update_yaxes(title_text="Task Success Rate (%)", range=[25, 100], secondary_y=False)
    fig.update_yaxes(title_text="Position Error (mm)", range=[0, 18], secondary_y=True)

    # Subtitle
    fig.add_annotation(
        x=0.5,
        y=1.06,
        xref="paper",
        yref="paper",
        text="Solid lines = success rate (left axis) | Dashed lines = position error (right axis)",
        showarrow=False,
        font=dict(size=13, color="gray"),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_sim_to_real_transfer(dark_mode=False)
    fig.write_html(str(output_dir / "06_sim_to_real_transfer.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "06_sim_to_real_transfer.png"), width=1920, height=1080, scale=2)
    print("Saved 06_sim_to_real_transfer.html and 06_sim_to_real_transfer.png")
    fig_dark = create_sim_to_real_transfer(dark_mode=True)
    fig_dark.write_html(str(output_dir / "06_sim_to_real_transfer_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "06_sim_to_real_transfer_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 06_sim_to_real_transfer_dark.html and _dark.png")
