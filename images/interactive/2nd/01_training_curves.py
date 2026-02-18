"""Federated training convergence curves for oncology trial FL platform.

Visualizes loss convergence across communication rounds for three federated
learning algorithms: FedAvg, FedProx (mu=0.01), and SCAFFOLD, demonstrating
how each strategy handles non-IID clinical data distributions.
"""

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_training_curves(dark_mode=False):
    """Create multi-line chart showing federated training convergence.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with training convergence curves.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    rounds = np.arange(0, 101)

    # FedAvg: standard convergence, slower on non-IID oncology data
    fedavg_base = 2.8 * np.exp(-0.025 * rounds) + 0.42
    fedavg_noise = np.random.normal(0, 0.03, len(rounds))
    fedavg_loss = fedavg_base + fedavg_noise
    fedavg_loss[0] = 2.8

    # FedProx (mu=0.01): better non-IID handling, converges to lower loss
    fedprox_base = 2.8 * np.exp(-0.032 * rounds) + 0.31
    fedprox_noise = np.random.normal(0, 0.025, len(rounds))
    fedprox_loss = fedprox_base + fedprox_noise
    fedprox_loss[0] = 2.8

    # SCAFFOLD: variance-reduced, fastest convergence on heterogeneous data
    scaffold_base = 2.8 * np.exp(-0.045 * rounds) + 0.24
    scaffold_noise = np.random.normal(0, 0.02, len(rounds))
    scaffold_loss = scaffold_base + scaffold_noise
    scaffold_loss[0] = 2.8

    fig = go.Figure()

    # Color scheme for the three algorithms
    colors = {
        "FedAvg": "#636EFA",
        "FedProx (mu=0.01)": "#EF553B",
        "SCAFFOLD": "#00CC96",
    }

    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=fedavg_loss,
            mode="lines",
            name="FedAvg",
            line=dict(color=colors["FedAvg"], width=2.5),
            hovertemplate="Round: %{x}<br>Loss: %{y:.4f}<extra>FedAvg</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=fedprox_loss,
            mode="lines",
            name="FedProx (\u03bc=0.01)",
            line=dict(color=colors["FedProx (mu=0.01)"], width=2.5),
            hovertemplate="Round: %{x}<br>Loss: %{y:.4f}<extra>FedProx</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=scaffold_loss,
            mode="lines",
            name="SCAFFOLD",
            line=dict(color=colors["SCAFFOLD"], width=2.5),
            hovertemplate="Round: %{x}<br>Loss: %{y:.4f}<extra>SCAFFOLD</extra>",
        )
    )

    # Add convergence region annotation
    fig.add_hrect(
        y0=0.2,
        y1=0.5,
        fillcolor="gray",
        opacity=0.1,
        line_width=0,
        annotation_text="Convergence Region",
        annotation_position="top right",
    )

    # Add milestone annotations
    fig.add_vline(x=20, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_annotation(
        x=20,
        y=2.2,
        text="Early Stopping<br>Checkpoint",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="gray",
        font=dict(size=10),
    )

    fig.add_vline(x=50, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_annotation(
        x=50,
        y=1.6,
        text="Mid-Training<br>Evaluation",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="gray",
        font=dict(size=10),
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Federated Training Convergence: Oncology Tumor Classification",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Communication Round",
            dtick=10,
            range=[0, 100],
            gridwidth=1,
        ),
        yaxis=dict(
            title="Cross-Entropy Loss",
            range=[0, 3.0],
            gridwidth=1,
        ),
        legend=dict(
            x=0.65,
            y=0.95,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=13),
        ),
        width=1920,
        height=1080,
        margin=dict(l=80, r=60, t=100, b=80),
        hovermode="x unified",
    )

    # Add text annotation for final loss values
    final_annotations = [
        ("FedAvg", fedavg_loss[-1], colors["FedAvg"]),
        ("FedProx", fedprox_loss[-1], colors["FedProx (mu=0.01)"]),
        ("SCAFFOLD", scaffold_loss[-1], colors["SCAFFOLD"]),
    ]
    for name, val, color in final_annotations:
        fig.add_annotation(
            x=100,
            y=val,
            text=f"  {name}: {val:.3f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=color),
        )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_training_curves(dark_mode=False)
    fig.write_html(str(output_dir / "01_training_curves.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "01_training_curves.png"), width=1920, height=1080, scale=2)
    print("Saved 01_training_curves.html and 01_training_curves.png")
    fig_dark = create_training_curves(dark_mode=True)
    fig_dark.write_html(str(output_dir / "01_training_curves_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "01_training_curves_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 01_training_curves_dark.html and _dark.png")
