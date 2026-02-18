"""Domain randomization sim-to-real transfer learning chart.

Shows how different levels of domain randomization in the physics simulator
affect real-world task success rates for robotic oncology procedures during
transfer from simulation to physical deployment.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_domain_randomization_transfer(dark_mode=False):
    """Create a multi-line chart showing sim-to-real transfer performance.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    grid_color = "#444444" if dark_mode else "#E0E0E0"

    np.random.seed(42)
    epochs = np.arange(0, 201, 5)
    n_epochs = len(epochs)

    # Sigmoid-like learning curves with different asymptotes and rates
    def learning_curve(epochs, max_rate, midpoint, steepness, noise_std):
        """Generate a noisy sigmoid learning curve."""
        base = max_rate / (1 + np.exp(-steepness * (epochs - midpoint)))
        noise = np.random.normal(0, noise_std, len(epochs))
        # Cumulative smoothing to make noise realistic
        smoothed = np.convolve(noise, np.ones(3) / 3, mode="same")
        return np.clip(base + smoothed, 0, 100)

    # No domain randomization: overfits to sim, poor transfer
    no_dr_sim = learning_curve(epochs, 98, 40, 0.08, 1.5)
    no_dr_real = learning_curve(epochs, 42, 80, 0.04, 3.0)

    # Low domain randomization
    low_dr_sim = learning_curve(epochs, 95, 45, 0.07, 1.8)
    low_dr_real = learning_curve(epochs, 62, 70, 0.05, 2.5)

    # Medium domain randomization (optimal)
    med_dr_sim = learning_curve(epochs, 90, 50, 0.065, 2.0)
    med_dr_real = learning_curve(epochs, 82, 65, 0.055, 2.0)

    # High domain randomization: harder to learn but best transfer
    high_dr_sim = learning_curve(epochs, 84, 60, 0.055, 2.2)
    high_dr_real = learning_curve(epochs, 78, 75, 0.05, 2.2)

    configs = [
        {
            "name": "No Randomization",
            "sim": no_dr_sim,
            "real": no_dr_real,
            "color": "#EF5350",
            "dash": "solid",
        },
        {
            "name": "Low Randomization",
            "sim": low_dr_sim,
            "real": low_dr_real,
            "color": "#FFA726",
            "dash": "solid",
        },
        {
            "name": "Medium Randomization",
            "sim": med_dr_sim,
            "real": med_dr_real,
            "color": "#66BB6A",
            "dash": "solid",
        },
        {
            "name": "High Randomization",
            "sim": high_dr_sim,
            "real": high_dr_real,
            "color": "#42A5F5",
            "dash": "solid",
        },
    ]

    fig = go.Figure()

    for cfg in configs:
        # Simulation performance (dashed lines)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=cfg["sim"],
                mode="lines",
                name=f"{cfg['name']} (Sim)",
                line=dict(color=cfg["color"], width=2, dash="dash"),
                opacity=0.5,
                legendgroup=cfg["name"],
                hovertemplate="Epoch %{x}<br>Sim Success: %{y:.1f}%<extra>" + cfg["name"] + " (Sim)</extra>",
            )
        )

        # Real-world performance (solid lines)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=cfg["real"],
                mode="lines",
                name=f"{cfg['name']} (Real)",
                line=dict(color=cfg["color"], width=3, dash="solid"),
                legendgroup=cfg["name"],
                hovertemplate="Epoch %{x}<br>Real Success: %{y:.1f}%<extra>" + cfg["name"] + " (Real)</extra>",
            )
        )

    # Add shaded region for sim-to-real gap for "No Randomization"
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([epochs, epochs[::-1]]),
            y=np.concatenate([no_dr_sim, no_dr_real[::-1]]),
            fill="toself",
            fillcolor="rgba(239, 83, 80, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Reality Gap (No DR)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Add annotation for optimal region
    fig.add_annotation(
        x=180,
        y=82,
        text="Optimal: Medium DR<br>82% real-world success",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#66BB6A",
        font=dict(size=12, color="#66BB6A"),
        bordercolor="#66BB6A",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        ax=-60,
        ay=-40,
    )

    # Add annotation for reality gap
    fig.add_annotation(
        x=180,
        y=70,
        text="Reality Gap: 56%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#EF5350",
        font=dict(size=11, color="#EF5350"),
        bordercolor="#EF5350",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(0,0,0,0.7)" if dark_mode else "rgba(255,255,255,0.9)",
        ax=60,
        ay=0,
    )

    # Clinical threshold line
    fig.add_hline(
        y=75,
        line_dash="dot",
        line_color="#9E9E9E",
        line_width=1,
        annotation_text="Clinical Deployment Threshold (75%)",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#9E9E9E",
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Sim-to-Real Transfer: Domain Randomization Impact on Robotic Biopsy Success Rate",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        xaxis=dict(title="Training Epochs", gridcolor=grid_color, range=[0, 205]),
        yaxis=dict(title="Task Success Rate (%)", gridcolor=grid_color, range=[0, 105]),
        legend=dict(
            x=0.01,
            y=0.50,
            bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.85)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(t=100, l=80, r=40, b=100),
        annotations=[
            dict(
                text=(
                    "Dashed = simulation performance | Solid = real-world performance | "
                    "Task: CT-guided needle biopsy with 7-DOF arm"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.08,
                showarrow=False,
                font=dict(size=11),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_domain_randomization_transfer(dark_mode=False)
    fig.write_html(str(output_dir / "09_domain_randomization_transfer.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "09_domain_randomization_transfer.png"), width=1920, height=1080, scale=2)
    print("Saved 09_domain_randomization_transfer.html and .png")
    fig_dark = create_domain_randomization_transfer(dark_mode=True)
    fig_dark.write_html(str(output_dir / "09_domain_randomization_transfer_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(
        str(output_dir / "09_domain_randomization_transfer_dark.png"), width=1920, height=1080, scale=2
    )
    print("Saved 09_domain_randomization_transfer_dark.html and _dark.png")
