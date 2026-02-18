"""Oncology treatment simulation pipeline sunburst chart.

Displays a hierarchical sunburst of oncology treatment modalities, their
constituent drug classes, and expected response categories used in the
digital twin simulation engine.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_oncology_process_diagram(dark_mode=False):
    """Create a sunburst chart of the oncology treatment simulation pipeline.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Hierarchy: Root -> Modality -> Drug Class -> Response Category
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    root = "Treatment Simulation"
    ids.append(root)
    labels.append(root)
    parents.append("")
    values.append(0)
    colors.append("#455A64")

    modalities = {
        "Chemotherapy": {
            "color": "#E53935",
            "drugs": {
                "Alkylating Agents": {"CR": 12, "PR": 28, "SD": 35, "PD": 25},
                "Antimetabolites": {"CR": 15, "PR": 30, "SD": 30, "PD": 25},
                "Taxanes": {"CR": 18, "PR": 32, "SD": 28, "PD": 22},
                "Platinum Compounds": {"CR": 20, "PR": 25, "SD": 30, "PD": 25},
            },
        },
        "Radiation": {
            "color": "#FB8C00",
            "drugs": {
                "EBRT (3D-CRT)": {"CR": 22, "PR": 33, "SD": 30, "PD": 15},
                "IMRT": {"CR": 28, "PR": 35, "SD": 25, "PD": 12},
                "SBRT": {"CR": 35, "PR": 30, "SD": 22, "PD": 13},
                "Proton Therapy": {"CR": 30, "PR": 32, "SD": 25, "PD": 13},
            },
        },
        "Immunotherapy": {
            "color": "#43A047",
            "drugs": {
                "PD-1 Inhibitors": {"CR": 15, "PR": 25, "SD": 35, "PD": 25},
                "PD-L1 Inhibitors": {"CR": 12, "PR": 22, "SD": 38, "PD": 28},
                "CTLA-4 Inhibitors": {"CR": 10, "PR": 20, "SD": 30, "PD": 40},
                "CAR-T Cell": {"CR": 40, "PR": 25, "SD": 15, "PD": 20},
            },
        },
        "Targeted Therapy": {
            "color": "#1E88E5",
            "drugs": {
                "Tyrosine Kinase Inh.": {"CR": 18, "PR": 35, "SD": 30, "PD": 17},
                "mTOR Inhibitors": {"CR": 10, "PR": 28, "SD": 37, "PD": 25},
                "CDK4/6 Inhibitors": {"CR": 15, "PR": 38, "SD": 32, "PD": 15},
                "PARP Inhibitors": {"CR": 20, "PR": 32, "SD": 28, "PD": 20},
            },
        },
        "Combination": {
            "color": "#8E24AA",
            "drugs": {
                "Chemo + Immuno": {"CR": 25, "PR": 35, "SD": 25, "PD": 15},
                "Chemo + Targeted": {"CR": 22, "PR": 33, "SD": 28, "PD": 17},
                "Immuno + Targeted": {"CR": 20, "PR": 30, "SD": 30, "PD": 20},
            },
        },
    }

    response_colors = {"CR": "#1B5E20", "PR": "#4CAF50", "SD": "#FFC107", "PD": "#D32F2F"}
    response_labels = {
        "CR": "Complete Response",
        "PR": "Partial Response",
        "SD": "Stable Disease",
        "PD": "Progressive Disease",
    }

    for mod_name, mod_data in modalities.items():
        mod_id = mod_name
        ids.append(mod_id)
        labels.append(mod_name)
        parents.append(root)
        values.append(0)
        colors.append(mod_data["color"])

        for drug_name, responses in mod_data["drugs"].items():
            drug_id = f"{mod_name}/{drug_name}"
            ids.append(drug_id)
            labels.append(drug_name)
            parents.append(mod_id)
            values.append(0)
            colors.append(mod_data["color"] + "CC")

            for resp_code, resp_val in responses.items():
                resp_id = f"{drug_id}/{resp_code}"
                ids.append(resp_id)
                labels.append(response_labels[resp_code])
                parents.append(drug_id)
                values.append(resp_val)
                colors.append(response_colors[resp_code])

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=4,
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Simulated Rate: %{value}%<extra></extra>",
            marker=dict(colors=colors, line=dict(width=1, color="white" if not dark_mode else "#222222")),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Oncology Treatment Simulation Pipeline - Digital Twin Response Modeling",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        margin=dict(t=80, l=20, r=20, b=40),
        annotations=[
            dict(
                text="Inner ring: Treatment modality | Middle ring: Drug class | Outer ring: RECIST response",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.02,
                showarrow=False,
                font=dict(size=13),
            )
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_oncology_process_diagram(dark_mode=False)
    fig.write_html(str(output_dir / "03_oncology_process_diagram.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "03_oncology_process_diagram.png"), width=1920, height=1080, scale=2)
    print("Saved 03_oncology_process_diagram.html and .png")
    fig_dark = create_oncology_process_diagram(dark_mode=True)
    fig_dark.write_html(str(output_dir / "03_oncology_process_diagram_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "03_oncology_process_diagram_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 03_oncology_process_diagram_dark.html and _dark.png")
