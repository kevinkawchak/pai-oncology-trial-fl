"""FDA device classification pathways treemap.

Treemap showing the hierarchy of FDA medical device classification for AI/ML
oncology devices: Device Classes -> Submission Types -> AI/ML Categories,
with realistic counts based on FDA clearance data.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_fda_device_classification_tree(dark_mode=False):
    """Create a treemap of FDA device classification pathways.

    Hierarchy:
      Level 1: Device Class (I, II, III)
      Level 2: Submission Type (510(k), De Novo, PMA, Breakthrough)
      Level 3: AI/ML Category (SaMD, SiMD, PCCP)

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Build hierarchical data for treemap
    labels = [
        # Root
        "FDA AI/ML Device Classifications",
        # Level 1: Device Classes
        "Class I - Exempt",
        "Class II - 510(k)/Special",
        "Class III - PMA",
        # Level 2 under Class I
        "510(k) Exempt",
        "De Novo (Class I)",
        # Level 2 under Class II
        "510(k) Traditional",
        "510(k) Special",
        "De Novo (Class II)",
        "Breakthrough (Class II)",
        # Level 2 under Class III
        "PMA Original",
        "PMA Supplement",
        "Breakthrough (Class III)",
        # Level 3 under 510(k) Exempt
        "SaMD - Exempt",
        "SiMD - Exempt",
        # Level 3 under De Novo (Class I)
        "SaMD - De Novo I",
        # Level 3 under 510(k) Traditional
        "SaMD - 510(k)",
        "SiMD - 510(k)",
        "PCCP - 510(k)",
        # Level 3 under 510(k) Special
        "SaMD - Special",
        "SiMD - Special",
        # Level 3 under De Novo (Class II)
        "SaMD - De Novo II",
        "PCCP - De Novo II",
        # Level 3 under Breakthrough (Class II)
        "SaMD - BT II",
        "PCCP - BT II",
        # Level 3 under PMA Original
        "SaMD - PMA",
        "SiMD - PMA",
        # Level 3 under PMA Supplement
        "SaMD - PMA Supp",
        "PCCP - PMA Supp",
        # Level 3 under Breakthrough (Class III)
        "SaMD - BT III",
        "PCCP - BT III",
    ]

    parents = [
        "",  # Root has no parent
        # Level 1
        "FDA AI/ML Device Classifications",
        "FDA AI/ML Device Classifications",
        "FDA AI/ML Device Classifications",
        # Level 2 under Class I
        "Class I - Exempt",
        "Class I - Exempt",
        # Level 2 under Class II
        "Class II - 510(k)/Special",
        "Class II - 510(k)/Special",
        "Class II - 510(k)/Special",
        "Class II - 510(k)/Special",
        # Level 2 under Class III
        "Class III - PMA",
        "Class III - PMA",
        "Class III - PMA",
        # Level 3 under 510(k) Exempt
        "510(k) Exempt",
        "510(k) Exempt",
        # Level 3 under De Novo (Class I)
        "De Novo (Class I)",
        # Level 3 under 510(k) Traditional
        "510(k) Traditional",
        "510(k) Traditional",
        "510(k) Traditional",
        # Level 3 under 510(k) Special
        "510(k) Special",
        "510(k) Special",
        # Level 3 under De Novo (Class II)
        "De Novo (Class II)",
        "De Novo (Class II)",
        # Level 3 under Breakthrough (Class II)
        "Breakthrough (Class II)",
        "Breakthrough (Class II)",
        # Level 3 under PMA Original
        "PMA Original",
        "PMA Original",
        # Level 3 under PMA Supplement
        "PMA Supplement",
        "PMA Supplement",
        # Level 3 under Breakthrough (Class III)
        "Breakthrough (Class III)",
        "Breakthrough (Class III)",
    ]

    # Values (leaf nodes have real counts; branch nodes are 0 so Plotly sums children)
    values = [
        0,  # Root
        0,
        0,
        0,  # Level 1 classes
        0,
        0,  # Level 2 Class I
        0,
        0,
        0,
        0,  # Level 2 Class II
        0,
        0,
        0,  # Level 2 Class III
        # Level 3 leaves
        18,
        12,  # 510(k) Exempt
        8,  # De Novo Class I
        142,
        65,
        38,  # 510(k) Traditional
        54,
        22,  # 510(k) Special
        35,
        15,  # De Novo Class II
        12,
        8,  # Breakthrough Class II
        24,
        11,  # PMA Original
        18,
        9,  # PMA Supplement
        6,
        4,  # Breakthrough Class III
    ]

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

    # Color mapping by device class
    class_i_color = "#2ecc71"
    class_ii_color = "#3498db"
    class_iii_color = "#e74c3c"

    colors = [
        "#f0f0f0",  # Root
        class_i_color,
        class_ii_color,
        class_iii_color,
        class_i_color,
        class_i_color,
        class_ii_color,
        class_ii_color,
        class_ii_color,
        class_ii_color,
        class_iii_color,
        class_iii_color,
        class_iii_color,
        "#27ae60",
        "#1abc9c",
        "#16a085",
        "#2980b9",
        "#2471a3",
        "#1f618d",
        "#5dade2",
        "#3498db",
        "#2e86c1",
        "#1a5276",
        "#85c1e9",
        "#7fb3d3",
        "#c0392b",
        "#a93226",
        "#e74c3c",
        "#d35400",
        "#cb4335",
        "#b03a2e",
    ]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Devices: %{value}<extra></extra>",
            marker=dict(
                colors=colors,
                line=dict(width=1.5, color="white" if not dark_mode else "#333"),
            ),
            pathbar=dict(visible=True),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="FDA AI/ML Medical Device Classification Pathways (Oncology Focus)",
            font=dict(size=18),
            x=0.5,
        ),
        width=1400,
        height=850,
        margin=dict(l=20, r=20, t=80, b=40),
        annotations=[
            dict(
                text="Size represents number of cleared/approved AI/ML devices per pathway",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.03,
                showarrow=False,
                font=dict(size=11, color="gray"),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_fda_device_classification_tree(dark_mode=False)
    fig.write_html(str(output_dir / "04_fda_device_classification_tree.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "04_fda_device_classification_tree.png"), width=1920, height=1080, scale=2)
    print("Saved 04_fda_device_classification_tree.html and 04_fda_device_classification_tree.png")
    fig_dark = create_fda_device_classification_tree(dark_mode=True)
    fig_dark.write_html(str(output_dir / "04_fda_device_classification_tree_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(
        str(output_dir / "04_fda_device_classification_tree_dark.png"), width=1920, height=1080, scale=2
    )
    print("Saved 04_fda_device_classification_tree_dark.html and _dark.png")
