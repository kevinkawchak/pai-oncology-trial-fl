"""Safety metrics dashboard for surgical robotics monitoring.

Multi-panel dashboard tracking force/torque monitoring, workspace boundary
violations, emergency stop frequency, and ISO 13482 compliance scores
for the oncology surgical robotics module.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_safety_metrics_dashboard(dark_mode=False):
    """Create multi-subplot safety metrics dashboard with 4 panels.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with 4-panel safety dashboard.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    np.random.seed(42)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Force/Torque Monitoring Over Time",
            "Workspace Boundary Violations by Joint",
            "Emergency Stop Trigger Frequency",
            "ISO 13482 Compliance Scores",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Panel 1: Force/Torque monitoring over time
    time_steps = np.linspace(0, 60, 300)  # 60 seconds of operation
    force_x = 5.0 + 2.0 * np.sin(0.3 * time_steps) + np.random.normal(0, 0.5, 300)
    force_y = 3.0 + 1.5 * np.cos(0.2 * time_steps) + np.random.normal(0, 0.4, 300)
    torque_z = 1.2 + 0.8 * np.sin(0.5 * time_steps) + np.random.normal(0, 0.2, 300)

    fig.add_trace(
        go.Scatter(x=time_steps, y=force_x, mode="lines", name="Force X (N)", line=dict(color="#636EFA", width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=force_y, mode="lines", name="Force Y (N)", line=dict(color="#EF553B", width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=torque_z, mode="lines", name="Torque Z (Nm)", line=dict(color="#00CC96", width=1.5)),
        row=1,
        col=1,
    )
    # Safety threshold line
    fig.add_hline(y=8.0, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)

    # Panel 2: Workspace boundary violations by joint
    joints = ["J1\nBase", "J2\nShoulder", "J3\nElbow", "J4\nWrist1", "J5\nWrist2", "J6\nWrist3"]
    violations = [3, 12, 8, 2, 1, 0]
    bar_colors = ["#00CC96" if v <= 3 else "#FFA15A" if v <= 8 else "#EF553B" for v in violations]

    fig.add_trace(
        go.Bar(x=joints, y=violations, marker_color=bar_colors, name="Violations", showlegend=False),
        row=1,
        col=2,
    )

    # Panel 3: Emergency stop trigger frequency histogram
    estop_times = np.concatenate(
        [
            np.random.exponential(15, 8),
            np.random.normal(30, 5, 5),
            np.random.normal(50, 3, 3),
        ]
    )
    estop_times = np.clip(estop_times, 0, 60)

    fig.add_trace(
        go.Histogram(
            x=estop_times,
            nbinsx=12,
            marker_color="#AB63FA",
            opacity=0.8,
            name="E-Stop Events",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Panel 4: ISO 13482 compliance scores
    categories = [
        "Hazard\nAnalysis",
        "Risk\nAssessment",
        "Protective\nMeasures",
        "Safety\nFunctions",
        "EMC\nRequirements",
    ]
    scores = [96.5, 92.3, 98.1, 94.7, 97.8]
    score_colors = ["#00CC96" if s >= 95 else "#FFA15A" if s >= 90 else "#EF553B" for s in scores]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=scores,
            marker_color=score_colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            name="Compliance %",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Surgical Robotics Safety Metrics Dashboard",
            font=dict(size=20),
        ),
        width=1920,
        height=1080,
        margin=dict(l=70, r=50, t=100, b=70),
        showlegend=True,
        legend=dict(x=0.01, y=0.52, font=dict(size=11)),
    )

    # Axis labels for each subplot
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Force (N) / Torque (Nm)", row=1, col=1)
    fig.update_xaxes(title_text="Joint", row=1, col=2)
    fig.update_yaxes(title_text="Violation Count", row=1, col=2)
    fig.update_xaxes(title_text="Time into Session (s)", row=2, col=1)
    fig.update_yaxes(title_text="Event Count", row=2, col=1)
    fig.update_xaxes(title_text="ISO 13482 Category", row=2, col=2)
    fig.update_yaxes(title_text="Score (%)", range=[80, 102], row=2, col=2)

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_safety_metrics_dashboard(dark_mode=False)
    fig.write_html(str(output_dir / "03_safety_metrics_dashboard.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "03_safety_metrics_dashboard.png"), width=1920, height=1080, scale=2)
    print("Saved 03_safety_metrics_dashboard.html and 03_safety_metrics_dashboard.png")
    fig_dark = create_safety_metrics_dashboard(dark_mode=True)
    fig_dark.write_html(str(output_dir / "03_safety_metrics_dashboard_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "03_safety_metrics_dashboard_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 03_safety_metrics_dashboard_dark.html and _dark.png")
