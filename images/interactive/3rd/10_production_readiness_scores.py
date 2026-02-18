"""Production readiness scores with bullet-style horizontal bars.

Horizontal bar chart with bullet-style indicators showing production readiness
scores for each major platform module. A target line at 80% marks the minimum
threshold for production deployment.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_production_readiness_scores(dark_mode=False):
    """Create a horizontal bar chart with bullet-style production readiness scores.

    Modules assessed:
        Federated Learning, Physical AI, Privacy Framework, Regulatory,
        Digital Twins, Clinical Analytics, Regulatory Submissions,
        Unification, Tools

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    modules = [
        "Tools",
        "Unification",
        "Regulatory Submissions",
        "Clinical Analytics",
        "Digital Twins",
        "Regulatory",
        "Privacy Framework",
        "Physical AI",
        "Federated Learning",
    ]

    # Actual scores (0-100)
    actual_scores = [72, 68, 78, 85, 74, 82, 91, 65, 88]
    # Maximum possible (background reference bar)
    max_scores = [100] * len(modules)
    # Target threshold
    target = 80

    # Color bars based on whether they meet target
    bar_colors = ["#2ecc71" if s >= target else "#e74c3c" if s < 60 else "#f39c12" for s in actual_scores]

    fig = go.Figure()

    # Background bars (light gray, representing 100% scale)
    fig.add_trace(
        go.Bar(
            y=modules,
            x=max_scores,
            orientation="h",
            marker=dict(color="rgba(189, 195, 199, 0.25)", line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Actual score bars
    fig.add_trace(
        go.Bar(
            y=modules,
            x=actual_scores,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(width=1, color="rgba(255,255,255,0.5)"),
            ),
            text=[f" {s}%" for s in actual_scores],
            textposition="outside",
            textfont=dict(size=12, color=bar_colors),
            name="Actual Score",
            hovertemplate="<b>%{y}</b><br>Score: %{x}%<extra></extra>",
        )
    )

    # Target line at 80% for each module
    fig.add_trace(
        go.Scatter(
            x=[target] * len(modules),
            y=modules,
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=22,
                line=dict(width=3, color="#2c3e50" if not dark_mode else "#ecf0f1"),
                color="rgba(0,0,0,0)",
            ),
            name=f"Target ({target}%)",
            hovertemplate=f"Target: {target}%<extra></extra>",
        )
    )

    # Add vertical target line
    fig.add_vline(
        x=target,
        line_dash="dash",
        line_color="rgba(44, 62, 80, 0.5)" if not dark_mode else "rgba(236, 240, 241, 0.5)",
        line_width=1.5,
        annotation_text=f"Production Target ({target}%)",
        annotation_position="top",
        annotation_font_size=11,
        annotation_font_color="gray",
    )

    # Status summary
    passing = sum(1 for s in actual_scores if s >= target)
    total = len(actual_scores)
    avg_score = np.mean(actual_scores)

    # Add status indicators as annotations
    for i, (module, score) in enumerate(zip(modules, actual_scores)):
        status = "PASS" if score >= target else "AT RISK" if score >= 60 else "FAIL"
        status_color = "#2ecc71" if score >= target else "#f39c12" if score >= 60 else "#e74c3c"
        fig.add_annotation(
            x=105,
            y=module,
            text=f"<b>{status}</b>",
            showarrow=False,
            font=dict(size=10, color=status_color),
            xanchor="left",
        )

    fig.update_layout(
        template=template,
        barmode="overlay",
        title=dict(
            text="PAI Platform: Production Readiness by Module",
            font=dict(size=18),
            x=0.5,
        ),
        xaxis=dict(
            title="Readiness Score (%)",
            range=[0, 115],
            dtick=10,
            gridwidth=1,
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        width=1200,
        height=650,
        margin=dict(l=180, r=80, t=100, b=70),
        annotations=[
            dict(
                text=(
                    f"Modules passing: {passing}/{total} | "
                    f"Average score: {avg_score:.1f}% | "
                    f"Green >= {target}% | Yellow >= 60% | Red < 60%"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.10,
                showarrow=False,
                font=dict(size=10, color="gray"),
            ),
        ]
        + list(fig.layout.annotations),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_production_readiness_scores(dark_mode=False)
    fig.write_html(str(output_dir / "10_production_readiness_scores.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "10_production_readiness_scores.png"), width=1920, height=1080, scale=2)
    print("Saved 10_production_readiness_scores.html and 10_production_readiness_scores.png")
    fig_dark = create_production_readiness_scores(dark_mode=True)
    fig_dark.write_html(str(output_dir / "10_production_readiness_scores_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "10_production_readiness_scores_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 10_production_readiness_scores_dark.html and _dark.png")
