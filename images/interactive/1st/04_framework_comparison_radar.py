"""Physics simulation framework comparison radar chart.

Compares Isaac Sim, MuJoCo, Gazebo, and PyBullet across eight performance
and capability axes relevant to medical robotics and oncology simulation
in the PAI platform.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_framework_comparison_radar(dark_mode=False):
    """Create a radar chart comparing four physics simulation frameworks.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    bg_color = "#111111" if dark_mode else "#FFFFFF"

    categories = [
        "Physics Fidelity",
        "Performance",
        "GPU Support",
        "ROS Integration",
        "Rendering Quality",
        "Documentation",
        "Ease of Use",
        "Medical Robotics",
    ]

    # Scores on 1-10 scale based on domain evaluation
    frameworks = {
        "NVIDIA Isaac Sim": {
            "scores": [9.5, 9.0, 10.0, 8.5, 9.5, 7.5, 6.0, 9.0],
            "color": "#76B900",
            "fill": "rgba(118, 185, 0, 0.15)",
        },
        "MuJoCo": {
            "scores": [9.0, 9.5, 7.0, 6.0, 5.5, 8.5, 8.0, 7.5],
            "color": "#2196F3",
            "fill": "rgba(33, 150, 243, 0.15)",
        },
        "Gazebo": {
            "scores": [7.0, 6.5, 3.0, 10.0, 6.0, 8.0, 7.0, 6.0],
            "color": "#FF9800",
            "fill": "rgba(255, 152, 0, 0.15)",
        },
        "PyBullet": {
            "scores": [7.5, 7.0, 5.0, 5.0, 4.0, 6.5, 9.0, 5.5],
            "color": "#E91E63",
            "fill": "rgba(233, 30, 99, 0.15)",
        },
    }

    fig = go.Figure()

    for name, data in frameworks.items():
        # Close the polygon by repeating the first value
        r_values = data["scores"] + [data["scores"][0]]
        theta_values = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                name=name,
                line=dict(color=data["color"], width=2.5),
                fill="toself",
                fillcolor=data["fill"],
                hovertemplate="<b>%{theta}</b><br>" + name + ": %{r}/10<extra></extra>",
            )
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="Physics Simulation Framework Comparison for Medical Robotics",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10.5],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=["2", "4", "6", "8", "10"],
                gridcolor="#444444" if dark_mode else "#DDDDDD",
            ),
            angularaxis=dict(
                gridcolor="#444444" if dark_mode else "#DDDDDD",
                linecolor="#666666" if dark_mode else "#BBBBBB",
            ),
            bgcolor=bg_color,
        ),
        legend=dict(
            font=dict(size=14),
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)" if dark_mode else "rgba(255,255,255,0.8)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
        ),
        margin=dict(t=100, l=100, r=100, b=80),
        annotations=[
            dict(
                text=(
                    "Scores based on PAI platform evaluation criteria for oncology simulation | "
                    "Scale: 1 (lowest) to 10 (highest)"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.04,
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_framework_comparison_radar(dark_mode=False)
    fig.write_html(str(output_dir / "04_framework_comparison_radar.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "04_framework_comparison_radar.png"), width=1920, height=1080, scale=2)
    print("Saved 04_framework_comparison_radar.html and .png")
