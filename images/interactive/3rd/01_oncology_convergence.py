"""Federated model convergence across oncology subtasks.

Multi-line chart showing how different oncology tasks converge at different rates
during federated learning across communication rounds. Tasks with higher complexity
(e.g., survival estimation) converge more slowly than simpler classification tasks.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_oncology_convergence(dark_mode=False):
    """Create a multi-line convergence chart for oncology FL subtasks.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    rounds = np.arange(1, 101)

    # Simulate convergence curves with different rates and plateaus
    # Tumor classification: fast convergence, high plateau
    tumor_class = 0.96 - 0.52 * np.exp(-0.08 * rounds) + np.random.default_rng(42).normal(0, 0.005, len(rounds))
    tumor_class = np.clip(tumor_class, 0.4, 0.97)

    # Biomarker detection: moderate-fast convergence
    biomarker = 0.93 - 0.50 * np.exp(-0.06 * rounds) + np.random.default_rng(43).normal(0, 0.006, len(rounds))
    biomarker = np.clip(biomarker, 0.38, 0.94)

    # Response prediction: moderate convergence
    response = 0.89 - 0.48 * np.exp(-0.045 * rounds) + np.random.default_rng(44).normal(0, 0.007, len(rounds))
    response = np.clip(response, 0.35, 0.90)

    # Toxicity grading: slower convergence, moderate plateau
    toxicity = 0.85 - 0.46 * np.exp(-0.035 * rounds) + np.random.default_rng(45).normal(0, 0.008, len(rounds))
    toxicity = np.clip(toxicity, 0.32, 0.86)

    # Survival estimation: slowest convergence, lower plateau
    survival = 0.80 - 0.44 * np.exp(-0.025 * rounds) + np.random.default_rng(46).normal(0, 0.009, len(rounds))
    survival = np.clip(survival, 0.30, 0.82)

    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
    tasks = [
        "Tumor Classification",
        "Biomarker Detection",
        "Response Prediction",
        "Toxicity Grading",
        "Survival Estimation",
    ]
    curves = [tumor_class, biomarker, response, toxicity, survival]

    fig = go.Figure()

    for i, (task, curve, color) in enumerate(zip(tasks, curves, colors)):
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=curve,
                mode="lines",
                name=task,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{task}</b><br>Round: %{{x}}<br>AUC-ROC: %{{y:.3f}}<extra></extra>",
            )
        )
        # Add endpoint annotation
        fig.add_annotation(
            x=100,
            y=curve[-1],
            text=f"  {curve[-1]:.3f}",
            showarrow=False,
            xanchor="left",
            font=dict(color=color, size=11),
        )

    # Add target threshold line
    fig.add_hline(
        y=0.85,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="Clinical Threshold (0.85)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="gray",
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Federated Model Convergence Across Oncology Subtasks",
            font=dict(size=20),
            x=0.5,
        ),
        xaxis=dict(
            title="Communication Rounds",
            dtick=10,
            range=[0, 108],
            gridwidth=1,
        ),
        yaxis=dict(
            title="Validation AUC-ROC",
            range=[0.3, 1.0],
            dtick=0.1,
            gridwidth=1,
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0)" if dark_mode else "rgba(255,255,255,0.8)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        width=1200,
        height=700,
        margin=dict(l=80, r=100, t=80, b=60),
        annotations=[
            dict(
                text="Higher complexity tasks converge more slowly in federated settings",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.12,
                showarrow=False,
                font=dict(size=11, color="gray"),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_oncology_convergence(dark_mode=False)
    fig.write_html(str(output_dir / "01_oncology_convergence.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "01_oncology_convergence.png"), width=1920, height=1080, scale=2)
    print("Saved 01_oncology_convergence.html and 01_oncology_convergence.png")
    fig_dark = create_oncology_convergence(dark_mode=True)
    fig_dark.write_html(str(output_dir / "01_oncology_convergence_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "01_oncology_convergence_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 01_oncology_convergence_dark.html and _dark.png")
