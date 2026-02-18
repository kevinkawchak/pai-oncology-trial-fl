"""Clinical trial workflow Sankey diagram.

Visualizes the end-to-end oncology clinical trial workflow as a Sankey diagram,
showing data flow volumes between major phases from patient enrollment through
regulatory submission.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_clinical_trial_workflow(dark_mode=False):
    """Create a Sankey diagram of clinical trial workflow phases.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    text_color = "#FFFFFF" if dark_mode else "#333333"

    # Node definitions for trial workflow
    node_labels = [
        "Screening (N=2400)",  # 0
        "Enrollment (N=1200)",  # 1
        "Randomization",  # 2
        "Treatment Arm A",  # 3
        "Treatment Arm B",  # 4
        "Control Arm",  # 5
        "Active Monitoring",  # 6
        "Adverse Event Review",  # 7
        "Dose Modification",  # 8
        "Continued Treatment",  # 9
        "Primary Analysis",  # 10
        "Secondary Analysis",  # 11
        "Biomarker Analysis",  # 12
        "Safety Reporting",  # 13
        "Efficacy Reporting",  # 14
        "CSR Generation",  # 15
        "FDA Submission",  # 16
        "EMA Submission",  # 17
        "Screen Failure",  # 18
        "Early Discontinuation",  # 19
        "Lost to Follow-up",  # 20
    ]

    node_colors = [
        "#42A5F5",
        "#1E88E5",
        "#1565C0",
        "#66BB6A",
        "#43A047",
        "#388E3C",
        "#FFA726",
        "#EF5350",
        "#FF7043",
        "#26A69A",
        "#AB47BC",
        "#8E24AA",
        "#7B1FA2",
        "#EF5350",
        "#66BB6A",
        "#5C6BC0",
        "#EC407A",
        "#EC407A",
        "#BDBDBD",
        "#BDBDBD",
        "#BDBDBD",
    ]

    # Source -> Target -> Value
    links = [
        (0, 1, 1200),  # Screening -> Enrollment
        (0, 18, 1200),  # Screening -> Screen Failure
        (1, 2, 1200),  # Enrollment -> Randomization
        (2, 3, 400),  # Randomization -> Arm A
        (2, 4, 400),  # Randomization -> Arm B
        (2, 5, 400),  # Randomization -> Control
        (3, 6, 370),  # Arm A -> Active Monitoring
        (4, 6, 365),  # Arm B -> Active Monitoring
        (5, 6, 380),  # Control -> Active Monitoring
        (3, 19, 30),  # Arm A -> Early Discontinuation
        (4, 19, 35),  # Arm B -> Early Discontinuation
        (5, 19, 20),  # Control -> Early Discontinuation
        (6, 7, 280),  # Monitoring -> AE Review
        (6, 9, 835),  # Monitoring -> Continued Treatment
        (7, 8, 120),  # AE Review -> Dose Modification
        (7, 19, 40),  # AE Review -> Early Discontinuation
        (7, 9, 120),  # AE Review -> Continued Treatment
        (8, 9, 110),  # Dose Modification -> Continued
        (8, 19, 10),  # Dose Modification -> Discontinuation
        (9, 20, 45),  # Continued -> Lost to Follow-up
        (9, 10, 1020),  # Continued -> Primary Analysis
        (10, 11, 1020),  # Primary -> Secondary Analysis
        (10, 12, 850),  # Primary -> Biomarker Analysis
        (11, 13, 510),  # Secondary -> Safety Reporting
        (11, 14, 510),  # Secondary -> Efficacy Reporting
        (12, 14, 850),  # Biomarker -> Efficacy Reporting
        (13, 15, 510),  # Safety -> CSR
        (14, 15, 1360),  # Efficacy -> CSR
        (15, 16, 1870),  # CSR -> FDA
        (15, 17, 1870),  # CSR -> EMA
    ]

    sources = [link[0] for link in links]
    targets = [link[1] for link in links]
    values = [link[2] for link in links]

    link_colors = []
    for src in sources:
        base = node_colors[src]
        link_colors.append(base.replace(")", ", 0.4)").replace("rgb", "rgba") if "rgb" in base else base + "66")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black" if not dark_mode else "#555555", width=0.5),
                label=node_labels,
                color=node_colors,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="<b>%{source.label}</b> -> <b>%{target.label}</b>"
                + "<br>Patients: %{value}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Oncology Clinical Trial Workflow - Patient Flow (Phase III)",
            font=dict(size=22, color=text_color),
            x=0.5,
        ),
        width=1920,
        height=1080,
        margin=dict(t=80, l=40, r=40, b=40),
        font=dict(size=12, color=text_color),
        annotations=[
            dict(
                text="Screening: 2,400 patients | Enrolled: 1,200 | Evaluable: 1,020 | Attrition: 15%",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.02,
                showarrow=False,
                font=dict(size=13, color=text_color),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_clinical_trial_workflow(dark_mode=False)
    fig.write_html(str(output_dir / "02_clinical_trial_workflow.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "02_clinical_trial_workflow.png"), width=1920, height=1080, scale=2)
    print("Saved 02_clinical_trial_workflow.html and .png")
