"""Privacy analytics pipeline Sankey diagram.

Sankey diagram showing the de-identification data flow from raw clinical data
through PHI detection, method selection, de-identification methods, validation,
and audit trail. Record counts illustrate data volume at each stage.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_privacy_analytics_pipeline(dark_mode=False):
    """Create a Sankey diagram of the de-identification data pipeline.

    Flow: Raw Data -> PHI Detection -> Method Selection ->
          [REDACT, HASH, GENERALIZE, SUPPRESS, PSEUDONYMIZE, DATE_SHIFT] ->
          Validated Output -> Audit Trail

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Node definitions
    node_labels = [
        # Stage 0: Raw data sources (0-4)
        "EHR Records (24,500)",
        "DICOM Images (18,200)",
        "Lab Reports (12,800)",
        "Clinical Notes (15,600)",
        "Consent Forms (8,400)",
        # Stage 1: PHI Detection (5)
        "PHI Detection Engine",
        # Stage 2: Method Selection (6)
        "Method Router",
        # Stage 3: De-identification methods (7-12)
        "REDACT",
        "HASH (SHA-256)",
        "GENERALIZE",
        "SUPPRESS",
        "PSEUDONYMIZE",
        "DATE_SHIFT",
        # Stage 4: Validation (13)
        "Validated Output",
        # Stage 5: Audit (14)
        "Audit Trail",
        # Stage 6: No PHI found path (15)
        "Clean Records (Pass-through)",
    ]

    node_colors = [
        # Raw sources - blue shades
        "#3498db",
        "#2980b9",
        "#2471a3",
        "#1f618d",
        "#1a5276",
        # PHI Detection
        "#e67e22",
        # Method Router
        "#9b59b6",
        # Methods - various
        "#e74c3c",
        "#8e44ad",
        "#2ecc71",
        "#f39c12",
        "#1abc9c",
        "#3498db",
        # Validated Output
        "#27ae60",
        # Audit Trail
        "#2c3e50",
        # Clean pass-through
        "#16a085",
    ]

    # Source -> Target links with values (record counts)
    sources = []
    targets = []
    values = []
    link_colors = []

    # Raw data -> PHI Detection
    raw_to_phi = [(0, 5, 24500), (1, 5, 18200), (2, 5, 12800), (3, 5, 15600), (4, 5, 8400)]
    for s, t, v in raw_to_phi:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append("rgba(52, 152, 219, 0.3)")

    # PHI Detection -> Clean Records (no PHI found)
    sources.append(5)
    targets.append(15)
    values.append(31200)
    link_colors.append("rgba(22, 160, 133, 0.3)")

    # PHI Detection -> Method Router (records with PHI)
    sources.append(5)
    targets.append(6)
    values.append(48300)
    link_colors.append("rgba(230, 126, 34, 0.3)")

    # Method Router -> Individual methods
    method_flows = [
        (6, 7, 12800, "rgba(231, 76, 60, 0.3)"),  # REDACT
        (6, 8, 8500, "rgba(142, 68, 173, 0.3)"),  # HASH
        (6, 9, 7200, "rgba(46, 204, 113, 0.3)"),  # GENERALIZE
        (6, 10, 4600, "rgba(243, 156, 18, 0.3)"),  # SUPPRESS
        (6, 11, 9400, "rgba(26, 188, 156, 0.3)"),  # PSEUDONYMIZE
        (6, 12, 5800, "rgba(52, 152, 219, 0.3)"),  # DATE_SHIFT
    ]
    for s, t, v, c in method_flows:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append(c)

    # Methods -> Validated Output
    method_to_valid = [
        (7, 13, 12800),
        (8, 13, 8500),
        (9, 13, 7200),
        (10, 13, 4600),
        (11, 13, 9400),
        (12, 13, 5800),
    ]
    for s, t, v in method_to_valid:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append("rgba(39, 174, 96, 0.3)")

    # Clean Records -> Validated Output
    sources.append(15)
    targets.append(13)
    values.append(31200)
    link_colors.append("rgba(22, 160, 133, 0.3)")

    # Validated Output -> Audit Trail
    sources.append(13)
    targets.append(14)
    values.append(79500)
    link_colors.append("rgba(44, 62, 80, 0.3)")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="rgba(0,0,0,0.3)", width=0.5),
                label=node_labels,
                color=node_colors,
                hovertemplate="<b>%{label}</b><br>Records: %{value:,}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="<b>%{source.label}</b> -> <b>%{target.label}</b><br>Records: %{value:,}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Privacy De-identification Pipeline: Data Flow & Record Counts",
            font=dict(size=18),
            x=0.5,
        ),
        width=1400,
        height=800,
        margin=dict(l=30, r=30, t=80, b=60),
        annotations=[
            dict(
                text=("79,500 total records processed | 39.3% clean pass-through | 60.7% required de-identification"),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                showarrow=False,
                font=dict(size=11, color="gray"),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_privacy_analytics_pipeline(dark_mode=False)
    fig.write_html(str(output_dir / "08_privacy_analytics_pipeline.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "08_privacy_analytics_pipeline.png"), width=1920, height=1080, scale=2)
    print("Saved 08_privacy_analytics_pipeline.html and 08_privacy_analytics_pipeline.png")
    fig_dark = create_privacy_analytics_pipeline(dark_mode=True)
    fig_dark.write_html(str(output_dir / "08_privacy_analytics_pipeline_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "08_privacy_analytics_pipeline_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 08_privacy_analytics_pipeline_dark.html and _dark.png")
