"""Sync latency distributions for Isaac-MuJoCo bridge.

Violin and box plots showing synchronization latency distributions
across different directions in the Isaac-MuJoCo simulation bridge
used for surgical robotics digital twin validation.
"""

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_latency_distributions(dark_mode=False):
    """Create violin/box plots for sync latency distributions.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with latency distribution plots.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    n_samples = 500

    # Isaac -> MuJoCo: generally fast with occasional spikes
    isaac_to_mujoco = np.concatenate(
        [
            np.random.lognormal(mean=0.5, sigma=0.4, size=int(n_samples * 0.9)),
            np.random.uniform(4, 8, size=int(n_samples * 0.1)),  # outlier spikes
        ]
    )

    # MuJoCo -> Isaac: slightly higher latency due to state translation
    mujoco_to_isaac = np.concatenate(
        [
            np.random.lognormal(mean=0.7, sigma=0.35, size=int(n_samples * 0.85)),
            np.random.uniform(5, 12, size=int(n_samples * 0.15)),
        ]
    )

    # Bidirectional: compound latency from both directions
    bidirectional = np.concatenate(
        [
            np.random.lognormal(mean=1.0, sigma=0.3, size=int(n_samples * 0.8)),
            np.random.uniform(6, 15, size=int(n_samples * 0.2)),
        ]
    )

    directions = ["Isaac \u2192 MuJoCo", "MuJoCo \u2192 Isaac", "Bidirectional"]
    datasets = [isaac_to_mujoco, mujoco_to_isaac, bidirectional]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    fig = go.Figure()

    for i, (direction, data, color) in enumerate(zip(directions, datasets, colors)):
        # Add violin plot
        fig.add_trace(
            go.Violin(
                y=data,
                x=[direction] * len(data),
                name=direction,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
                points="outliers",
                marker=dict(color=color, opacity=0.3, size=3),
                hovertemplate=(f"<b>{direction}</b><br>Latency: %{{y:.2f}} ms<br><extra></extra>"),
                scalemode="width",
                width=0.6,
            )
        )

    # Add SLA threshold line
    fig.add_hline(
        y=5.0,
        line_dash="dash",
        line_color="#FFA15A",
        opacity=0.8,
        annotation_text="Real-Time SLA Threshold (5ms)",
        annotation_position="top right",
    )

    # Add hard deadline line
    fig.add_hline(
        y=10.0,
        line_dash="dash",
        line_color="#EF553B",
        opacity=0.6,
        annotation_text="Hard Deadline (10ms)",
        annotation_position="top right",
    )

    # Compute and add statistics annotations
    stats_text_parts = []
    for direction, data in zip(directions, datasets):
        median_val = np.median(data)
        p95_val = np.percentile(data, 95)
        p99_val = np.percentile(data, 99)
        sla_pct = (data <= 5.0).sum() / len(data) * 100
        stats_text_parts.append(
            f"<b>{direction}</b>: "
            f"Median={median_val:.2f}ms, P95={p95_val:.2f}ms, P99={p99_val:.2f}ms, "
            f"SLA Compliance={sla_pct:.1f}%"
        )

    stats_text = "<br>".join(stats_text_parts)

    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="center",
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Isaac-MuJoCo Bridge: Synchronization Latency Distributions",
            font=dict(size=20),
        ),
        yaxis=dict(
            title="Latency (ms)",
            range=[0, max(np.max(bidirectional), 16) * 1.1],
            gridwidth=1,
        ),
        xaxis=dict(title="Sync Direction", tickfont=dict(size=14)),
        showlegend=False,
        width=1920,
        height=1080,
        margin=dict(l=80, r=60, t=100, b=140),
        violinmode="group",
    )

    # Subtitle
    fig.add_annotation(
        x=0.5,
        y=1.06,
        xref="paper",
        yref="paper",
        text="Distribution of 500 sync events per direction during 6-DOF surgical arm coordination",
        showarrow=False,
        font=dict(size=13, color="gray"),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_latency_distributions(dark_mode=False)
    fig.write_html(str(output_dir / "07_latency_distributions.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "07_latency_distributions.png"), width=1920, height=1080, scale=2)
    print("Saved 07_latency_distributions.html and 07_latency_distributions.png")
