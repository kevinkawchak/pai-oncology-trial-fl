"""Repository architecture treemap visualization.

Displays the PAI Oncology Trial FL repository module structure as an interactive
treemap, with module sizes proportional to the number of Python files in each
top-level package.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_repository_architecture_treemap(dark_mode=False):
    """Create a treemap of the repository module structure.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Repository modules and their file counts sourced from actual repo structure
    modules = {
        "federated": {
            "files": 9,
            "children": {
                "fl_aggregation": 2,
                "fl_client": 2,
                "fl_server": 1,
                "fl_privacy": 1,
                "fl_model_registry": 1,
                "fl_training": 1,
                "fl_utils": 1,
            },
        },
        "physical_ai": {
            "files": 6,
            "children": {
                "isaac_sim_bridge": 2,
                "mujoco_bridge": 1,
                "physics_engine": 1,
                "robotics_control": 1,
                "sensor_fusion": 1,
            },
        },
        "privacy": {
            "files": 10,
            "children": {
                "differential_privacy": 2,
                "secure_aggregation": 2,
                "homomorphic_enc": 2,
                "data_anonymization": 1,
                "audit_logging": 1,
                "consent_management": 1,
                "access_control": 1,
            },
        },
        "regulatory": {
            "files": 6,
            "children": {
                "fda_compliance": 2,
                "ema_guidelines": 1,
                "gcp_validation": 1,
                "audit_trail": 1,
                "submission_prep": 1,
            },
        },
        "digital-twins": {
            "files": 3,
            "children": {
                "patient_twin": 1,
                "organ_models": 1,
                "twin_sync": 1,
            },
        },
        "clinical-analytics": {
            "files": 7,
            "children": {
                "survival_analysis": 2,
                "biomarker_analysis": 1,
                "endpoint_evaluation": 1,
                "subgroup_analysis": 1,
                "safety_monitoring": 1,
                "response_assessment": 1,
            },
        },
        "regulatory-submissions": {
            "files": 6,
            "children": {
                "ectd_builder": 2,
                "nda_module": 1,
                "bla_module": 1,
                "ind_module": 1,
                "csr_generator": 1,
            },
        },
        "unification": {
            "files": 5,
            "children": {
                "pipeline_orchestrator": 2,
                "config_manager": 1,
                "api_gateway": 1,
                "schema_registry": 1,
            },
        },
        "tools": {
            "files": 5,
            "children": {
                "data_converters": 2,
                "model_validators": 1,
                "benchmark_suite": 1,
                "deployment_utils": 1,
            },
        },
        "examples": {
            "files": 17,
            "children": {
                "quickstart": 3,
                "federated_training": 3,
                "privacy_demo": 2,
                "digital_twin_demo": 2,
                "regulatory_demo": 2,
                "analytics_demo": 2,
                "physical_ai_demo": 3,
            },
        },
        "tests": {
            "files": 97,
            "children": {
                "unit_tests": 40,
                "integration_tests": 25,
                "e2e_tests": 15,
                "fixtures": 10,
                "conftest": 7,
            },
        },
    }

    # Build treemap hierarchy
    labels = ["PAI Oncology Trial FL"]
    parents = [""]
    values = [0]
    colors = [0.0]

    module_palette = [
        "#2196F3",
        "#4CAF50",
        "#FF9800",
        "#E91E63",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#3F51B5",
        "#8BC34A",
        "#FFC107",
        "#795548",
    ]

    for idx, (mod_name, mod_data) in enumerate(modules.items()):
        labels.append(mod_name)
        parents.append("PAI Oncology Trial FL")
        values.append(mod_data["files"])
        colors.append(float(idx + 1))

        for child_name, child_count in mod_data["children"].items():
            labels.append(child_name)
            parents.append(mod_name)
            values.append(child_count)
            colors.append(float(idx + 1))

    # Compute branch totals bottom-up so kaleido renders the treemap
    _children = {}
    for _i, (_lbl, _par) in enumerate(zip(labels, parents)):
        if _par:
            _children.setdefault(_par, []).append(_i)

    def _total(idx):
        lbl = labels[idx]
        if lbl not in _children:
            return values[idx]
        s = sum(_total(c) for c in _children[lbl])
        values[idx] = s
        return s

    _total(0)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Files: %{value}<extra></extra>",
            marker=dict(
                colors=colors,
                colorscale="Viridis",
                line=dict(width=1, color="white" if not dark_mode else "#333333"),
            ),
            pathbar=dict(visible=True),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="PAI Oncology Trial FL - Repository Architecture",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        margin=dict(t=80, l=20, r=20, b=20),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_repository_architecture_treemap(dark_mode=False)
    fig.write_html(str(output_dir / "01_repository_architecture_treemap.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "01_repository_architecture_treemap.png"), width=1920, height=1080, scale=2)
    print("Saved 01_repository_architecture_treemap.html and .png")
    fig_dark = create_repository_architecture_treemap(dark_mode=True)
    fig_dark.write_html(str(output_dir / "01_repository_architecture_treemap_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(
        str(output_dir / "01_repository_architecture_treemap_dark.png"), width=1920, height=1080, scale=2
    )
    print("Saved 01_repository_architecture_treemap_dark.html and _dark.png")
