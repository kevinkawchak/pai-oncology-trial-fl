"""Clinical trial KPI tracker with indicator gauges.

Bullet/indicator gauge chart showing key performance indicators for
the oncology clinical trial: enrollment rate, protocol compliance,
data quality, site activation, SAE reporting, and query resolution.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import numpy as np


def create_oncology_kpi_tracker(dark_mode=False):
    """Create bullet/indicator gauge chart for clinical trial KPIs.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        Plotly Figure object with KPI indicator gauges.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    bg_color = "#111111" if dark_mode else "#FFFFFF"
    text_color = "#FFFFFF" if dark_mode else "#333333"

    kpis = [
        {
            "title": "Enrollment Rate",
            "value": 75,
            "target": 100,
            "suffix": " / 100",
            "description": "Patients enrolled vs target",
            "thresholds": [50, 75, 100],
        },
        {
            "title": "Protocol Compliance",
            "value": 94,
            "target": 95,
            "suffix": "%",
            "description": "Sites following protocol",
            "thresholds": [80, 90, 100],
        },
        {
            "title": "Data Quality Score",
            "value": 97,
            "target": 95,
            "suffix": "%",
            "description": "Clean data rate across CRFs",
            "thresholds": [85, 92, 100],
        },
        {
            "title": "Site Activation",
            "value": 8,
            "target": 10,
            "suffix": " / 10",
            "description": "Active clinical sites",
            "thresholds": [4, 7, 10],
        },
        {
            "title": "SAE Reporting",
            "value": 100,
            "target": 100,
            "suffix": "%",
            "description": "Timely SAE report submissions",
            "thresholds": [90, 95, 100],
        },
        {
            "title": "Query Resolution",
            "value": 89,
            "target": 90,
            "suffix": "%",
            "description": "Data queries resolved on time",
            "thresholds": [70, 85, 100],
        },
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for kpi, (row, col) in zip(kpis, positions):
        value = kpi["value"]
        target = kpi["target"]
        thresholds = kpi["thresholds"]

        # Determine color based on performance
        pct = value / target * 100
        if pct >= 95:
            gauge_color = "#00CC96"
        elif pct >= 80:
            gauge_color = "#FFA15A"
        else:
            gauge_color = "#EF553B"

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                title=dict(
                    text=(
                        f"<b>{kpi['title']}</b><br><span style='font-size:11px;color:gray'>{kpi['description']}</span>"
                    ),
                    font=dict(size=16, color=text_color),
                ),
                delta=dict(
                    reference=target,
                    increasing=dict(color="#00CC96"),
                    decreasing=dict(color="#EF553B"),
                    suffix=kpi["suffix"].replace(f" / {target}", ""),
                ),
                number=dict(
                    suffix=kpi["suffix"],
                    font=dict(size=28, color=text_color),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, target],
                        tickwidth=1,
                        tickcolor=text_color,
                    ),
                    bar=dict(color=gauge_color, thickness=0.6),
                    bgcolor="rgba(128,128,128,0.1)",
                    borderwidth=1,
                    bordercolor="rgba(128,128,128,0.3)",
                    steps=[
                        dict(range=[0, thresholds[0]], color="rgba(239,85,59,0.15)"),
                        dict(range=[thresholds[0], thresholds[1]], color="rgba(255,161,90,0.15)"),
                        dict(range=[thresholds[1], thresholds[2]], color="rgba(0,204,150,0.15)"),
                    ],
                    threshold=dict(
                        line=dict(color="#636EFA", width=3),
                        thickness=0.8,
                        value=target,
                    ),
                ),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="Oncology Clinical Trial KPI Dashboard",
            font=dict(size=22, color=text_color),
            x=0.5,
        ),
        width=1920,
        height=1080,
        margin=dict(l=60, r=60, t=120, b=60),
        paper_bgcolor=bg_color,
    )

    # Add overall status annotation
    total_on_target = sum(1 for k in kpis if k["value"] / k["target"] >= 0.95)
    status_text = f"Overall Status: {total_on_target}/{len(kpis)} KPIs on target"
    status_color = "#00CC96" if total_on_target >= 5 else "#FFA15A" if total_on_target >= 3 else "#EF553B"

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text=status_text,
        showarrow=False,
        font=dict(size=15, color=status_color),
    )

    # Add timestamp annotation
    fig.add_annotation(
        x=1.0,
        y=-0.02,
        xref="paper",
        yref="paper",
        text="Last Updated: 2026-02-18 | Trial Phase: III | ClinicalTrials.gov: NCT06012345",
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="right",
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_oncology_kpi_tracker(dark_mode=False)
    fig.write_html(str(output_dir / "09_oncology_kpi_tracker.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "09_oncology_kpi_tracker.png"), width=1920, height=1080, scale=2)
    print("Saved 09_oncology_kpi_tracker.html and 09_oncology_kpi_tracker.png")
    fig_dark = create_oncology_kpi_tracker(dark_mode=True)
    fig_dark.write_html(str(output_dir / "09_oncology_kpi_tracker_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "09_oncology_kpi_tracker_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 09_oncology_kpi_tracker_dark.html and _dark.png")
