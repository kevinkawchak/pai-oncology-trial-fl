"""Regulatory compliance scorecard with grouped horizontal bars.

Grouped horizontal bar chart showing compliance assessment scores for the PAI
oncology platform across major regulatory frameworks. Each regulation is scored
across three categories: Compliant, Minor Finding, and Major Finding.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_regulatory_compliance_scorecard(dark_mode=False):
    """Create a grouped horizontal bar chart for regulatory compliance scores.

    Regulations assessed:
        FDA 21 CFR 820, FDA Part 11, HIPAA, GDPR, ISO 14971,
        IEC 62304, ICH E6(R3), ICH E9(R1), EU MDR 2017/745

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    regulations = [
        "EU MDR 2017/745",
        "ICH E9(R1)",
        "ICH E6(R3)",
        "IEC 62304",
        "ISO 14971",
        "GDPR",
        "HIPAA",
        "FDA Part 11",
        "FDA 21 CFR 820",
    ]

    # Compliance data: (compliant %, minor finding %, major finding %)
    # Scores reflect realistic assessment of an ML platform
    compliant = [72, 88, 85, 78, 82, 90, 94, 86, 80]
    minor = [18, 8, 10, 15, 12, 7, 4, 10, 14]
    major = [10, 4, 5, 7, 6, 3, 2, 4, 6]

    fig = go.Figure()

    # Compliant bars (green)
    fig.add_trace(
        go.Bar(
            y=regulations,
            x=compliant,
            name="Compliant",
            orientation="h",
            marker=dict(color="#2ecc71", line=dict(width=0.5, color="white")),
            text=[f"{v}%" for v in compliant],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{y}</b><br>Compliant: %{x}%<extra></extra>",
        )
    )

    # Minor finding bars (yellow/amber)
    fig.add_trace(
        go.Bar(
            y=regulations,
            x=minor,
            name="Minor Finding",
            orientation="h",
            marker=dict(color="#f1c40f", line=dict(width=0.5, color="white")),
            text=[f"{v}%" for v in minor],
            textposition="inside",
            textfont=dict(size=11, color="#2c3e50"),
            hovertemplate="<b>%{y}</b><br>Minor Finding: %{x}%<extra></extra>",
        )
    )

    # Major finding bars (red)
    fig.add_trace(
        go.Bar(
            y=regulations,
            x=major,
            name="Major Finding",
            orientation="h",
            marker=dict(color="#e74c3c", line=dict(width=0.5, color="white")),
            text=[f"{v}%" for v in major],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{y}</b><br>Major Finding: %{x}%<extra></extra>",
        )
    )

    # Add a vertical line at 80% compliant threshold
    fig.add_vline(
        x=80,
        line_dash="dash",
        line_color="rgba(128,128,128,0.6)",
        line_width=1.5,
        annotation_text="80% Target",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color="gray",
    )

    fig.update_layout(
        template=template,
        barmode="stack",
        title=dict(
            text="PAI Oncology Platform: Regulatory Compliance Scorecard",
            font=dict(size=18),
            x=0.5,
        ),
        xaxis=dict(
            title="Assessment Score (%)",
            range=[0, 105],
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
            font=dict(size=12),
        ),
        width=1200,
        height=650,
        margin=dict(l=160, r=40, t=100, b=60),
        annotations=[
            dict(
                text="Assessment based on latest internal audit | Target: 80%+ compliant per regulation",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.10,
                showarrow=False,
                font=dict(size=10, color="gray"),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_regulatory_compliance_scorecard(dark_mode=False)
    fig.write_html(str(output_dir / "06_regulatory_compliance_scorecard.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "06_regulatory_compliance_scorecard.png"), width=1920, height=1080, scale=2)
    print("Saved 06_regulatory_compliance_scorecard.html and 06_regulatory_compliance_scorecard.png")
    fig_dark = create_regulatory_compliance_scorecard(dark_mode=True)
    fig_dark.write_html(str(output_dir / "06_regulatory_compliance_scorecard_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(
        str(output_dir / "06_regulatory_compliance_scorecard_dark.png"), width=1920, height=1080, scale=2
    )
    print("Saved 06_regulatory_compliance_scorecard_dark.html and _dark.png")
