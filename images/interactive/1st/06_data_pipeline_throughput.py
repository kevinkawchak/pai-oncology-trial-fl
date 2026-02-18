"""Federated data pipeline throughput bar chart.

Visualizes throughput metrics (records per second) for each stage of the
federated learning data pipeline, broken down by data type: clinical records,
imaging data, and genomic profiles.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_data_pipeline_throughput(dark_mode=False):
    """Create a bar chart of federated data pipeline throughput metrics.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    grid_color = "#444444" if dark_mode else "#E0E0E0"

    stages = [
        "Data Ingestion",
        "Schema\nHarmonization",
        "Quality\nValidation",
        "Privacy\nFiltering",
        "Secure\nAggregation",
        "Model\nUpdate",
    ]

    # Throughput in records/second for each data type at each pipeline stage
    clinical_throughput = [45000, 38000, 32000, 28000, 15000, 12000]
    imaging_throughput = [1200, 950, 800, 720, 380, 310]
    genomic_throughput = [8500, 6800, 5200, 4500, 2400, 1800]

    # Error bars (standard deviation across federated nodes)
    clinical_err = [3200, 2800, 2500, 2200, 1800, 1500]
    imaging_err = [150, 120, 100, 90, 60, 45]
    genomic_err = [600, 500, 420, 380, 250, 180]

    fig = go.Figure()

    # Clinical records
    fig.add_trace(
        go.Bar(
            name="Clinical Records",
            x=stages,
            y=clinical_throughput,
            error_y=dict(type="data", array=clinical_err, visible=True, color="#1565C0"),
            marker=dict(
                color="#42A5F5",
                line=dict(color="#1565C0", width=1),
                pattern=dict(shape=""),
            ),
            hovertemplate="<b>%{x}</b><br>Clinical: %{y:,.0f} rec/s<extra></extra>",
        )
    )

    # Imaging data
    fig.add_trace(
        go.Bar(
            name="Imaging Data (DICOM)",
            x=stages,
            y=imaging_throughput,
            error_y=dict(type="data", array=imaging_err, visible=True, color="#E65100"),
            marker=dict(
                color="#FF9800",
                line=dict(color="#E65100", width=1),
            ),
            hovertemplate="<b>%{x}</b><br>Imaging: %{y:,.0f} rec/s<extra></extra>",
            yaxis="y2",
        )
    )

    # Genomic profiles
    fig.add_trace(
        go.Bar(
            name="Genomic Profiles",
            x=stages,
            y=genomic_throughput,
            error_y=dict(type="data", array=genomic_err, visible=True, color="#1B5E20"),
            marker=dict(
                color="#66BB6A",
                line=dict(color="#1B5E20", width=1),
            ),
            hovertemplate="<b>%{x}</b><br>Genomic: %{y:,.0f} rec/s<extra></extra>",
        )
    )

    # Add throughput reduction line
    total_throughput = [c + g for c, g in zip(clinical_throughput, genomic_throughput)]
    pct_retained = [t / total_throughput[0] * 100 for t in total_throughput]

    fig.add_trace(
        go.Scatter(
            name="Cumulative Retention %",
            x=stages,
            y=pct_retained,
            mode="lines+markers+text",
            text=[f"{p:.0f}%" for p in pct_retained],
            textposition="top center",
            textfont=dict(size=11),
            line=dict(color="#E91E63", width=3, dash="dot"),
            marker=dict(size=10, color="#E91E63", symbol="diamond"),
            yaxis="y3",
            hovertemplate="<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Federated Data Pipeline Throughput by Stage and Data Type",
            font=dict(size=22),
            x=0.5,
        ),
        barmode="group",
        width=1920,
        height=1080,
        xaxis=dict(title="Pipeline Stage", tickangle=0),
        yaxis=dict(
            title="Throughput (records/sec)",
            gridcolor=grid_color,
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="Imaging Throughput (DICOM/sec)",
            overlaying="y",
            side="right",
            gridcolor=grid_color,
            rangemode="tozero",
            showgrid=False,
        ),
        yaxis3=dict(
            title="Cumulative Retention (%)",
            overlaying="y",
            side="right",
            position=0.95,
            range=[0, 110],
            showgrid=False,
            visible=False,
        ),
        legend=dict(
            x=0.65,
            y=0.95,
            bgcolor="rgba(0,0,0,0.5)" if dark_mode else "rgba(255,255,255,0.85)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
        ),
        margin=dict(t=100, l=80, r=120, b=80),
        annotations=[
            dict(
                text=(
                    "8 federated hospital nodes | Privacy filtering applies differential privacy "
                    "(epsilon=1.0) | Secure aggregation uses CKKS homomorphic encryption"
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
    fig = create_data_pipeline_throughput(dark_mode=False)
    fig.write_html(str(output_dir / "06_data_pipeline_throughput.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "06_data_pipeline_throughput.png"), width=1920, height=1080, scale=2)
    print("Saved 06_data_pipeline_throughput.html and .png")
    fig_dark = create_data_pipeline_throughput(dark_mode=True)
    fig_dark.write_html(str(output_dir / "06_data_pipeline_throughput_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "06_data_pipeline_throughput_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 06_data_pipeline_throughput_dark.html and _dark.png")
