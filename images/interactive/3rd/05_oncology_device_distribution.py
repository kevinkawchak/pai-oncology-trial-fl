"""Oncology AI/ML device distribution sunburst chart.

Sunburst visualization showing the distribution of oncology AI/ML devices by
category (Radiology, Pathology, Treatment Planning, Monitoring), regulation
status, and device class.
"""

import plotly.graph_objects as go
from pathlib import Path


def create_oncology_device_distribution(dark_mode=False):
    """Create a sunburst chart of oncology AI/ML device distribution.

    Layers:
      Inner ring: Category (Radiology, Pathology, Treatment Planning, Monitoring)
      Middle ring: Regulation Status (Cleared, Pending, Clinical Trial)
      Outer ring: Device Class (I, II, III)

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Build hierarchical data
    labels = []
    parents = []
    values = []
    colors = []

    # Color palettes
    cat_colors = {
        "Radiology": "#3498db",
        "Pathology": "#2ecc71",
        "Treatment Planning": "#e67e22",
        "Monitoring": "#9b59b6",
    }
    status_shades = {
        "Cleared": 1.0,
        "Pending": 0.7,
        "Clinical Trial": 0.45,
    }
    class_labels = ["Class I", "Class II", "Class III"]

    # Data: category -> status -> class -> count
    data = {
        "Radiology": {
            "Cleared": {"Class I": 8, "Class II": 95, "Class III": 12},
            "Pending": {"Class I": 3, "Class II": 28, "Class III": 7},
            "Clinical Trial": {"Class I": 1, "Class II": 15, "Class III": 9},
        },
        "Pathology": {
            "Cleared": {"Class I": 5, "Class II": 42, "Class III": 6},
            "Pending": {"Class I": 2, "Class II": 18, "Class III": 4},
            "Clinical Trial": {"Class I": 1, "Class II": 12, "Class III": 5},
        },
        "Treatment Planning": {
            "Cleared": {"Class I": 2, "Class II": 28, "Class III": 8},
            "Pending": {"Class I": 1, "Class II": 14, "Class III": 6},
            "Clinical Trial": {"Class I": 0, "Class II": 8, "Class III": 11},
        },
        "Monitoring": {
            "Cleared": {"Class I": 6, "Class II": 22, "Class III": 3},
            "Pending": {"Class I": 2, "Class II": 10, "Class III": 2},
            "Clinical Trial": {"Class I": 1, "Class II": 7, "Class III": 4},
        },
    }

    # Root label
    root = "Oncology AI/ML"
    labels.append(root)
    parents.append("")
    values.append(0)
    colors.append("#636efa" if dark_mode else "#2c3e50")

    # Build hierarchy
    for cat, statuses in data.items():
        labels.append(cat)
        parents.append(root)
        values.append(0)
        colors.append(cat_colors[cat])

        for status, classes in statuses.items():
            node_label = f"{cat} - {status}"
            labels.append(node_label)
            parents.append(cat)
            values.append(0)
            base_color = cat_colors[cat]
            # Adjust opacity through hex modification
            shade = status_shades[status]
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)
            r = int(r * shade + 255 * (1 - shade))
            g = int(g * shade + 255 * (1 - shade))
            b = int(b * shade + 255 * (1 - shade))
            colors.append(f"#{r:02x}{g:02x}{b:02x}")

            for cls, count in classes.items():
                if count > 0:
                    leaf_label = f"{cat[:4]}/{status[:5]}/{cls}"
                    labels.append(leaf_label)
                    parents.append(node_label)
                    values.append(count)
                    class_color_map = {"Class I": "#1abc9c", "Class II": "#f39c12", "Class III": "#e74c3c"}
                    colors.append(class_color_map[cls])

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Devices: %{value}<extra></extra>",
            marker=dict(
                colors=colors,
                line=dict(width=1, color="white" if not dark_mode else "#222"),
            ),
            maxdepth=3,
            insidetextorientation="radial",
        )
    )

    # Add donut hole annotation
    total_devices = sum(
        count for statuses in data.values() for classes in statuses.values() for count in classes.values()
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Oncology AI/ML Device Distribution by Category, Status & Class",
            font=dict(size=18),
            x=0.5,
        ),
        width=1100,
        height=900,
        margin=dict(l=40, r=40, t=80, b=60),
        annotations=[
            dict(
                text=f"Total: {total_devices} devices",
                x=0.5,
                y=0.5,
                font=dict(size=16, color="gray"),
                showarrow=False,
                xref="paper",
                yref="paper",
            ),
            dict(
                text="Inner: Category | Middle: Status | Outer: Device Class",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.04,
                showarrow=False,
                font=dict(size=11, color="gray"),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_oncology_device_distribution(dark_mode=False)
    fig.write_html(str(output_dir / "05_oncology_device_distribution.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "05_oncology_device_distribution.png"), width=1920, height=1080, scale=2)
    print("Saved 05_oncology_device_distribution.html and 05_oncology_device_distribution.png")
