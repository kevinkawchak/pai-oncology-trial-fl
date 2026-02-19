"""Multi-site clinical trial enrollment dashboard.

A four-panel dashboard displaying: enrollment waterfall by site, screening funnel,
monthly enrollment trend, and geographic site distribution for a multi-center
oncology federated learning trial.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np


def create_multi_site_trial_dashboard(dark_mode=False):
    """Create a multi-subplot clinical trial enrollment dashboard.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    sites = [
        "MD Anderson",
        "Memorial Sloan Kettering",
        "Mayo Clinic",
        "Johns Hopkins",
        "Dana-Farber",
        "UCSF Medical",
        "Cleveland Clinic",
        "Stanford Cancer",
        "Mass General",
        "Mount Sinai",
    ]
    enrollment = [145, 132, 118, 112, 98, 94, 87, 82, 76, 68]
    targets = [150, 140, 120, 120, 110, 100, 90, 90, 80, 75]

    funnel_stages = ["Screened", "Eligible", "Enrolled", "Randomized", "Completed"]
    funnel_counts = [2480, 1860, 1012, 948, 812]

    rng = np.random.default_rng(99)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_actual = [42, 58, 74, 89, 95, 102, 88, 110, 98, 105, 118, 133]
    monthly_target = [50, 60, 70, 80, 90, 100, 100, 110, 110, 110, 120, 130]

    # Geographic bubble data (lat-like, lon-like for US positions)
    site_x = [0.75, 0.85, 0.55, 0.78, 0.88, 0.12, 0.60, 0.15, 0.90, 0.82]
    site_y = [0.40, 0.65, 0.55, 0.55, 0.68, 0.52, 0.60, 0.48, 0.63, 0.62]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Enrollment by Site vs Target",
            "Patient Screening Funnel",
            "Monthly Enrollment Trend",
            "Site Distribution (bubble = enrollment)",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # Panel 1: Waterfall bar chart
    bar_colors = ["#2ecc71" if a >= t else "#e74c3c" for a, t in zip(enrollment, targets)]
    fig.add_trace(
        go.Bar(
            x=sites,
            y=enrollment,
            name="Enrolled",
            marker_color=bar_colors,
            text=enrollment,
            textposition="outside",
            textfont=dict(size=9),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sites,
            y=targets,
            mode="markers",
            name="Target",
            marker=dict(symbol="line-ew", size=14, line=dict(width=2, color="#3498db")),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Panel 2: Screening funnel
    funnel_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
    fig.add_trace(
        go.Funnel(
            y=funnel_stages,
            x=funnel_counts,
            textinfo="value+percent initial",
            marker=dict(color=funnel_colors),
            connector=dict(line=dict(color="gray", width=1)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Panel 3: Monthly enrollment trend
    fig.add_trace(
        go.Scatter(
            x=months,
            y=monthly_actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#2ecc71", width=2.5),
            marker=dict(size=7),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=monthly_target,
            mode="lines",
            name="Target Pace",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # Panel 4: Geographic bubble chart
    fig.add_trace(
        go.Scatter(
            x=site_x,
            y=site_y,
            mode="markers+text",
            marker=dict(
                size=[e / 3 for e in enrollment],
                color=enrollment,
                colorscale="Viridis",
                showscale=False,
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=[s.split()[0] for s in sites],
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate="%{text}<br>Enrolled: %{marker.size}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(title_text="Patients", row=1, col=1)
    fig.update_yaxes(title_text="Patients / Month", row=2, col=1)
    fig.update_xaxes(title_text="Month (2025)", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    fig.update_layout(
        template=template,
        title=dict(
            text="PAI Oncology Trial: Multi-Site Enrollment Dashboard",
            font=dict(size=18),
            x=0.5,
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_multi_site_trial_dashboard(dark_mode=False)
    fig.write_html(str(output_dir / "02_multi_site_trial_dashboard.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "02_multi_site_trial_dashboard.png"), width=1920, height=1080, scale=2)
    print("Saved 02_multi_site_trial_dashboard.html and 02_multi_site_trial_dashboard.png")
    fig_dark = create_multi_site_trial_dashboard(dark_mode=True)
    fig_dark.write_html(str(output_dir / "02_multi_site_trial_dashboard_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "02_multi_site_trial_dashboard_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 02_multi_site_trial_dashboard_dark.html and _dark.png")
