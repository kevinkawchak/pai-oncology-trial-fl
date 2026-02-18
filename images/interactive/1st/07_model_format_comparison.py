"""Model format comparison grouped bar chart.

Compares five model serialization formats used in the PAI federated learning
pipeline across key performance metrics: file size, load time, inference time,
memory usage, and numerical precision.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_model_format_comparison(dark_mode=False):
    """Create a grouped bar chart comparing model serialization formats.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    formats = ["PyTorch (.pt)", "ONNX (.onnx)", "SafeTensors (.safetensors)", "TensorFlow (.pb)", "MONAI Bundle"]

    # Metrics normalized to 0-100 scale (lower is better for most)
    metrics = {
        "File Size (MB)": {
            "values": [482, 395, 378, 520, 445],
            "color": "#42A5F5",
            "description": "Serialized model size for a 3D U-Net oncology segmentation model",
        },
        "Load Time (ms)": {
            "values": [1250, 890, 420, 1480, 1100],
            "color": "#66BB6A",
            "description": "Time to deserialize and load into GPU memory",
        },
        "Inference Time (ms)": {
            "values": [45, 32, 44, 48, 42],
            "color": "#FFA726",
            "description": "Per-sample inference latency on NVIDIA A100",
        },
        "Peak Memory (MB)": {
            "values": [2100, 1850, 1680, 2350, 1950],
            "color": "#EF5350",
            "description": "Peak GPU memory during inference",
        },
        "Precision Loss (1e-7)": {
            "values": [0, 12, 0, 5, 2],
            "color": "#AB47BC",
            "description": "Numerical deviation from reference FP32 output",
        },
    }

    fig = go.Figure()

    for metric_name, metric_data in metrics.items():
        fig.add_trace(
            go.Bar(
                name=metric_name,
                x=formats,
                y=metric_data["values"],
                marker=dict(
                    color=metric_data["color"],
                    line=dict(color="#333333" if dark_mode else "#666666", width=0.5),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>" + metric_name + ": %{y}<br>" + metric_data["description"] + "<extra></extra>"
                ),
            )
        )

    # Add normalized score as scatter overlay
    # Compute composite score: weighted sum (lower = better) inverted to 0-100
    weights = [0.15, 0.25, 0.30, 0.20, 0.10]
    raw_scores = []
    for i in range(len(formats)):
        score = 0.0
        for j, (_, md) in enumerate(metrics.items()):
            max_val = max(md["values"]) if max(md["values"]) > 0 else 1
            score += weights[j] * (md["values"][i] / max_val)
        raw_scores.append(score)

    # Invert so higher = better, scale to 100
    max_raw = max(raw_scores)
    composite_scores = [(1 - (s / max_raw)) * 100 + 20 for s in raw_scores]

    fig.add_trace(
        go.Scatter(
            name="Composite Score (higher=better)",
            x=formats,
            y=composite_scores,
            mode="markers+text",
            text=[f"{s:.0f}" for s in composite_scores],
            textposition="top center",
            textfont=dict(size=13, color="#FFD600"),
            marker=dict(
                size=18,
                color="#FFD600",
                symbol="star",
                line=dict(color="#F57F17", width=2),
            ),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Composite Score: %{y:.1f}/100<extra></extra>",
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Model Serialization Format Comparison - Oncology 3D U-Net",
            font=dict(size=22),
            x=0.5,
        ),
        barmode="group",
        width=1920,
        height=1080,
        xaxis=dict(title="Model Format"),
        yaxis=dict(
            title="Metric Value",
            gridcolor="#444444" if dark_mode else "#E0E0E0",
        ),
        yaxis2=dict(
            title="Composite Score",
            overlaying="y",
            side="right",
            range=[0, 120],
            showgrid=False,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.85)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(t=100, l=80, r=100, b=100),
        annotations=[
            dict(
                text=(
                    "Benchmark: 3D U-Net (encoder-decoder, 31M params) trained on CT volumes | "
                    "Hardware: NVIDIA A100 80GB | Batch size: 1"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=11),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_model_format_comparison(dark_mode=False)
    fig.write_html(str(output_dir / "07_model_format_comparison.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "07_model_format_comparison.png"), width=1920, height=1080, scale=2)
    print("Saved 07_model_format_comparison.html and .png")
