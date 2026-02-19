"""Computational resource utilization during federated training.

Stacked area chart showing how GPU/CPU resources are allocated across
different phases of a federated learning training session: model training,
secure aggregation, privacy filtering, communication, and idle time.
"""

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_resource_utilization(dark_mode=False):
    """Create stacked area chart of resource utilization over time.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with stacked area chart.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    # Time axis: 120-minute federated training session, sampled every 30 seconds
    n_points = 240
    time_minutes = np.linspace(0, 120, n_points)

    # Create periodic patterns reflecting federated learning rounds
    # Each round ~4 minutes: train -> aggregate -> filter -> communicate
    round_phase = (time_minutes % 4) / 4  # 0-1 within each round

    # Model Training: high during first 60% of each round
    training = np.where(
        round_phase < 0.6,
        55 + 15 * np.sin(np.pi * round_phase / 0.6) + np.random.normal(0, 2, n_points),
        5 + np.random.normal(0, 1, n_points),
    )
    training = np.clip(training, 2, 75)

    # Secure Aggregation: peaks during 50-75% of each round
    aggregation = np.where(
        (round_phase >= 0.5) & (round_phase < 0.75),
        30 + 10 * np.sin(np.pi * (round_phase - 0.5) / 0.25) + np.random.normal(0, 1.5, n_points),
        2 + np.random.normal(0, 0.5, n_points),
    )
    aggregation = np.clip(aggregation, 0, 45)

    # Privacy Filtering: active during 60-85% of each round
    privacy = np.where(
        (round_phase >= 0.6) & (round_phase < 0.85),
        15 + 8 * np.sin(np.pi * (round_phase - 0.6) / 0.25) + np.random.normal(0, 1, n_points),
        1 + np.random.normal(0, 0.3, n_points),
    )
    privacy = np.clip(privacy, 0, 25)

    # Communication: peaks at round boundaries (85-100% and 0-10%)
    communication = np.where(
        (round_phase >= 0.85) | (round_phase < 0.1),
        20 + 8 * np.random.random(n_points) + np.random.normal(0, 1.5, n_points),
        3 + np.random.normal(0, 0.5, n_points),
    )
    communication = np.clip(communication, 0, 30)

    # Idle: whatever is left to reach ~100%
    used = training + aggregation + privacy + communication
    idle = np.clip(100 - used, 0, 100)

    layers = [
        ("Model Training", training, "#636EFA"),
        ("Secure Aggregation", aggregation, "#EF553B"),
        ("Privacy Filtering", privacy, "#00CC96"),
        ("Communication", communication, "#AB63FA"),
        ("Idle", idle, "#B6B6B6"),
    ]

    fig = go.Figure()

    for name, values, color in layers:
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=values,
                mode="lines",
                name=name,
                stackgroup="one",
                fillcolor=color,
                line=dict(width=0.5, color=color),
                hovertemplate=f"<b>{name}</b><br>Time: %{{x:.1f}} min<br>Usage: %{{y:.1f}}%<extra></extra>",
            )
        )

    # Add round boundary markers every 4 minutes
    round_boundaries = np.arange(0, 121, 4)
    for rb in round_boundaries[1:-1]:
        if rb % 20 == 0:
            fig.add_vline(x=rb, line_dash="dot", line_color="gray", opacity=0.3)

    # Phase annotations
    fig.add_annotation(
        x=10,
        y=105,
        text="Warm-up Phase",
        showarrow=False,
        font=dict(size=11, color="gray"),
    )
    fig.add_annotation(
        x=60,
        y=105,
        text="Steady-State Training",
        showarrow=False,
        font=dict(size=11, color="gray"),
    )
    fig.add_annotation(
        x=110,
        y=105,
        text="Final Rounds",
        showarrow=False,
        font=dict(size=11, color="gray"),
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Federated Training Session: Computational Resource Utilization",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Session Time (minutes)",
            range=[0, 120],
            dtick=10,
        ),
        yaxis=dict(
            title="Resource Utilization (%)",
            range=[0, 110],
            dtick=20,
        ),
        legend=dict(
            x=1.01,
            y=0.95,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        width=1920,
        height=1080,
        margin=dict(l=80, r=180, t=100, b=80),
        hovermode="x unified",
    )

    # Average utilization stats
    avg_training = np.mean(training)
    avg_agg = np.mean(aggregation)
    avg_priv = np.mean(privacy)
    avg_comm = np.mean(communication)
    avg_idle = np.mean(idle)

    fig.add_annotation(
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        text=(
            f"Session Averages: Training {avg_training:.1f}% | "
            f"Aggregation {avg_agg:.1f}% | Privacy {avg_priv:.1f}% | "
            f"Communication {avg_comm:.1f}% | Idle {avg_idle:.1f}%"
        ),
        showarrow=False,
        font=dict(size=12, color="gray"),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_resource_utilization(dark_mode=False)
    fig.write_html(str(output_dir / "08_resource_utilization.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "08_resource_utilization.png"), width=1920, height=1080, scale=2)
    print("Saved 08_resource_utilization.html and 08_resource_utilization.png")
    fig_dark = create_resource_utilization(dark_mode=True)
    fig_dark.write_html(str(output_dir / "08_resource_utilization_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "08_resource_utilization_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 08_resource_utilization_dark.html and _dark.png")
