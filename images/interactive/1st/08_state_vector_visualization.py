"""Simulation state vector 3D scatter visualization.

Displays joint position, velocity, and torque state vectors from the
Isaac Sim - MuJoCo bridge, highlighting synchronized versus drifted states
during robotic-assisted oncology procedures.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_state_vector_visualization(dark_mode=False):
    """Create a 3D scatter plot of simulation state vectors.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    bg_color = "#111111" if dark_mode else "#FFFFFF"
    grid_color = "#333333" if dark_mode else "#DDDDDD"

    np.random.seed(42)
    n_synced = 300
    n_drifted = 80
    n_corrected = 50

    # Synchronized states: tight cluster around operating trajectory
    # Simulate a 7-DOF robotic arm during biopsy procedure
    t_synced = np.linspace(0, 2 * np.pi, n_synced)
    positions_synced = np.sin(t_synced) * 0.8 + np.random.normal(0, 0.03, n_synced)
    velocities_synced = np.cos(t_synced) * 0.5 + np.random.normal(0, 0.02, n_synced)
    torques_synced = -np.sin(t_synced) * 0.3 + np.random.normal(0, 0.015, n_synced)

    # Drifted states: divergence between Isaac Sim and MuJoCo
    t_drifted = np.linspace(np.pi * 0.8, np.pi * 1.5, n_drifted)
    drift_factor = np.linspace(0.05, 0.4, n_drifted)
    positions_drifted = np.sin(t_drifted) * 0.8 + drift_factor + np.random.normal(0, 0.05, n_drifted)
    velocities_drifted = np.cos(t_drifted) * 0.5 - drift_factor * 0.8 + np.random.normal(0, 0.04, n_drifted)
    torques_drifted = -np.sin(t_drifted) * 0.3 + drift_factor * 1.2 + np.random.normal(0, 0.03, n_drifted)

    # Corrected states: after bridge re-synchronization
    t_corrected = np.linspace(np.pi * 1.5, np.pi * 1.8, n_corrected)
    correction = np.linspace(0.35, 0.0, n_corrected)
    positions_corr = np.sin(t_corrected) * 0.8 + correction + np.random.normal(0, 0.04, n_corrected)
    velocities_corr = np.cos(t_corrected) * 0.5 - correction * 0.5 + np.random.normal(0, 0.03, n_corrected)
    torques_corr = -np.sin(t_corrected) * 0.3 + correction * 0.6 + np.random.normal(0, 0.02, n_corrected)

    # Drift magnitude for color coding
    drift_mag_synced = np.sqrt(
        np.random.normal(0, 0.03, n_synced) ** 2
        + np.random.normal(0, 0.02, n_synced) ** 2
        + np.random.normal(0, 0.015, n_synced) ** 2
    )
    drift_mag_drifted = np.sqrt(drift_factor**2 + (drift_factor * 0.8) ** 2 + (drift_factor * 1.2) ** 2)
    drift_mag_corrected = np.sqrt(correction**2 + (correction * 0.5) ** 2 + (correction * 0.6) ** 2)

    fig = go.Figure()

    # Synchronized states
    fig.add_trace(
        go.Scatter3d(
            x=positions_synced,
            y=velocities_synced,
            z=torques_synced,
            mode="markers",
            name="Synchronized",
            marker=dict(
                size=3,
                color=drift_mag_synced,
                colorscale="Greens",
                cmin=0,
                cmax=0.5,
                opacity=0.7,
            ),
            hovertemplate=("Pos: %{x:.3f} rad<br>Vel: %{y:.3f} rad/s<br>Torque: %{z:.3f} Nm<extra>Synced</extra>"),
        )
    )

    # Drifted states
    fig.add_trace(
        go.Scatter3d(
            x=positions_drifted,
            y=velocities_drifted,
            z=torques_drifted,
            mode="markers",
            name="Drifted",
            marker=dict(
                size=5,
                color=drift_mag_drifted,
                colorscale="Reds",
                cmin=0,
                cmax=0.5,
                opacity=0.9,
                symbol="diamond",
            ),
            hovertemplate=(
                "Pos: %{x:.3f} rad<br>Vel: %{y:.3f} rad/s<br>Torque: %{z:.3f} Nm"
                + "<br>Drift: %{marker.color:.3f}<extra>Drifted</extra>"
            ),
        )
    )

    # Corrected states
    fig.add_trace(
        go.Scatter3d(
            x=positions_corr,
            y=velocities_corr,
            z=torques_corr,
            mode="markers",
            name="Re-synchronized",
            marker=dict(
                size=4,
                color=drift_mag_corrected,
                colorscale="Blues",
                cmin=0,
                cmax=0.5,
                opacity=0.85,
                symbol="cross",
            ),
            hovertemplate=(
                "Pos: %{x:.3f} rad<br>Vel: %{y:.3f} rad/s<br>Torque: %{z:.3f} Nm"
                + "<br>Residual: %{marker.color:.3f}<extra>Corrected</extra>"
            ),
        )
    )

    # Ideal trajectory reference line
    t_ref = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter3d(
            x=np.sin(t_ref) * 0.8,
            y=np.cos(t_ref) * 0.5,
            z=-np.sin(t_ref) * 0.3,
            mode="lines",
            name="Reference Trajectory",
            line=dict(color="#FFD600", width=4, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Isaac Sim - MuJoCo State Vector Synchronization (7-DOF Biopsy Arm)",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        scene=dict(
            xaxis=dict(title="Joint Position (rad)", gridcolor=grid_color, backgroundcolor=bg_color),
            yaxis=dict(title="Joint Velocity (rad/s)", gridcolor=grid_color, backgroundcolor=bg_color),
            zaxis=dict(title="Joint Torque (Nm)", gridcolor=grid_color, backgroundcolor=bg_color),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor=bg_color,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.85)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
        ),
        margin=dict(t=80, l=20, r=20, b=40),
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_state_vector_visualization(dark_mode=False)
    fig.write_html(str(output_dir / "08_state_vector_visualization.html"), include_plotlyjs=True)
    fig.write_image(str(output_dir / "08_state_vector_visualization.png"), width=1920, height=1080, scale=2)
    print("Saved 08_state_vector_visualization.html and .png")
    fig_dark = create_state_vector_visualization(dark_mode=True)
    fig_dark.write_html(str(output_dir / "08_state_vector_visualization_dark.html"), include_plotlyjs=True)
    fig_dark.write_image(str(output_dir / "08_state_vector_visualization_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 08_state_vector_visualization_dark.html and _dark.png")
