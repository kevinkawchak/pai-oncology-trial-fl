"""Radar chart comparing federated aggregation algorithms.

Compares FedAvg, FedProx, and SCAFFOLD across six performance dimensions
reflecting their known trade-offs in federated learning for oncology trials.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_algorithm_comparison_radar(dark_mode=False):
    """Create a radar chart comparing three federated aggregation algorithms.

    Axes represent key performance dimensions:
    - Communication Efficiency: bandwidth usage per round
    - Convergence Speed: rounds to reach target accuracy
    - Non-IID Robustness: performance under heterogeneous data
    - Privacy Preservation: inherent privacy guarantees
    - Computational Cost: per-client compute overhead (inverted: higher = cheaper)
    - Model Quality: final model accuracy on held-out data

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    categories = [
        "Communication\nEfficiency",
        "Convergence\nSpeed",
        "Non-IID\nRobustness",
        "Privacy\nPreservation",
        "Computational\nCost",
        "Model\nQuality",
    ]

    # Scores based on known FL algorithm characteristics (0-100 scale)
    # FedAvg: simple, efficient comms, poor on non-IID, moderate convergence
    fedavg = [92, 65, 48, 60, 95, 72]
    # FedProx: proximal term helps non-IID, slightly slower, more compute
    fedprox = [85, 74, 78, 62, 72, 81]
    # SCAFFOLD: variance reduction, best non-IID, higher communication cost
    scaffold = [58, 88, 92, 65, 55, 90]

    # Close the polygon
    categories_closed = categories + [categories[0]]
    fedavg_closed = fedavg + [fedavg[0]]
    fedprox_closed = fedprox + [fedprox[0]]
    scaffold_closed = scaffold + [scaffold[0]]

    fig = go.Figure()

    algorithms = [
        ("FedAvg", fedavg_closed, "#3498db", "toself"),
        ("FedProx", fedprox_closed, "#e67e22", "toself"),
        ("SCAFFOLD", scaffold_closed, "#2ecc71", "toself"),
    ]

    for name, values, color, fill in algorithms:
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill=fill,
                fillcolor="rgba(0,0,0,0)",
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r}}/100<extra></extra>",
            )
        )

    # Override fill colors properly using rgba
    fig.data[0].fillcolor = "rgba(52, 152, 219, 0.08)"
    fig.data[1].fillcolor = "rgba(230, 126, 34, 0.08)"
    fig.data[2].fillcolor = "rgba(46, 204, 113, 0.08)"

    fig.update_layout(
        template=template,
        title=dict(
            text="Federated Aggregation Algorithm Comparison",
            font=dict(size=20),
            x=0.5,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor="rgba(128,128,128,0.3)",
            ),
            angularaxis=dict(
                gridcolor="rgba(128,128,128,0.3)",
                linecolor="rgba(128,128,128,0.5)",
            ),
        ),
        legend=dict(
            x=0.85,
            y=0.98,
            bgcolor="rgba(0,0,0,0)" if dark_mode else "rgba(255,255,255,0.9)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            font=dict(size=13),
        ),
        width=1000,
        height=800,
        margin=dict(l=100, r=100, t=100, b=80),
        annotations=[
            dict(
                text=(
                    "SCAFFOLD excels on non-IID data at higher communication cost | "
                    "FedAvg is most communication-efficient | "
                    "FedProx balances robustness and cost"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.06,
                showarrow=False,
                font=dict(size=10, color="gray"),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_algorithm_comparison_radar(dark_mode=False)
    fig.write_html(str(output_dir / "03_algorithm_comparison_radar.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "03_algorithm_comparison_radar.png"), width=1920, height=1080, scale=2)
    print("Saved 03_algorithm_comparison_radar.html and 03_algorithm_comparison_radar.png")
    fig_dark = create_algorithm_comparison_radar(dark_mode=True)
    fig_dark.write_html(str(output_dir / "03_algorithm_comparison_radar_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "03_algorithm_comparison_radar_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 03_algorithm_comparison_radar_dark.html and _dark.png")
