"""Platform module performance radar charts.

Overlapping radar charts comparing key performance dimensions across six
core modules of the PAI Oncology Trial FL platform: Federated Learning,
Digital Twins, Privacy Framework, Regulatory Compliance, Clinical Analytics,
and Surgical Robotics.
"""

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_performance_radars(dark_mode=False):
    """Create overlapping radar charts for platform module performance.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with radar charts.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    axes = ["Throughput", "Accuracy", "Coverage", "Compliance", "Scalability"]

    modules = {
        "Federated Learning": {
            "values": [88, 94, 82, 96, 91],
            "color": "#636EFA",
        },
        "Digital Twins": {
            "values": [76, 97, 71, 88, 68],
            "color": "#EF553B",
        },
        "Privacy Framework": {
            "values": [72, 89, 95, 99, 85],
            "color": "#00CC96",
        },
        "Regulatory Compliance": {
            "values": [65, 92, 98, 100, 78],
            "color": "#AB63FA",
        },
        "Clinical Analytics": {
            "values": [83, 96, 87, 94, 80],
            "color": "#FFA15A",
        },
        "Surgical Robotics": {
            "values": [91, 98, 62, 93, 55],
            "color": "#FF6692",
        },
    }

    fig = go.Figure()

    for module_name, module_data in modules.items():
        values = module_data["values"]
        # Close the polygon by repeating the first value
        values_closed = values + [values[0]]
        axes_closed = axes + [axes[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=axes_closed,
                fill="toself",
                fillcolor=module_data["color"].replace(")", ", 0.1)").replace("#", "rgba(")
                if module_data["color"].startswith("rgba")
                else None,
                opacity=0.7,
                name=module_name,
                line=dict(color=module_data["color"], width=2),
                hovertemplate=(f"<b>{module_name}</b><br>%{{theta}}: %{{r}}/100<br><extra></extra>"),
            )
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="PAI Platform Module Performance Comparison",
            font=dict(size=20),
            x=0.5,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor="rgba(128,128,128,0.3)",
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                gridcolor="rgba(128,128,128,0.3)",
            ),
        ),
        legend=dict(
            x=1.05,
            y=0.95,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        width=1920,
        height=1080,
        margin=dict(l=100, r=250, t=100, b=80),
    )

    # Add subtitle
    fig.add_annotation(
        x=0.5,
        y=1.07,
        xref="paper",
        yref="paper",
        text="Scores normalized to 0-100 scale across Throughput, Accuracy, Coverage, Compliance, and Scalability",
        showarrow=False,
        font=dict(size=13, color="gray"),
    )

    # Add performance tier annotations
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Performance Tiers: \u2265 90 Excellent | \u2265 75 Good | < 75 Needs Improvement",
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="left",
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_performance_radars(dark_mode=False)
    fig.write_html(str(output_dir / "04_performance_radars.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "04_performance_radars.png"), width=1920, height=1080, scale=2)
    print("Saved 04_performance_radars.html and 04_performance_radars.png")
    fig_dark = create_performance_radars(dark_mode=True)
    fig_dark.write_html(str(output_dir / "04_performance_radars_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "04_performance_radars_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 04_performance_radars_dark.html and _dark.png")
