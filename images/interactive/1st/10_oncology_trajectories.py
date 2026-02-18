"""Oncology tumor growth trajectories under treatment modalities.

Multi-panel visualization of tumor volume dynamics using mathematical growth
models (exponential, logistic, Gompertz) with treatment intervention points,
simulated in the PAI digital twin engine.
"""

import plotly.graph_objects as go
from pathlib import Path
import numpy as np


def create_oncology_trajectories(dark_mode=False):
    """Create a multi-panel line chart of tumor growth trajectories.

    Args:
        dark_mode: If True, use plotly_dark template; otherwise plotly_white.

    Returns:
        A plotly Figure object.
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    grid_color = "#444444" if dark_mode else "#E0E0E0"

    days = np.arange(0, 365, 1)
    treatment_start = 60  # Day of treatment initiation

    # --- Exponential growth model ---
    def exponential_growth(t, v0, growth_rate, treatment_day, treatment_effect):
        """Exponential tumor growth with treatment-induced decay."""
        v = np.zeros_like(t, dtype=float)
        for i, day in enumerate(t):
            if day < treatment_day:
                v[i] = v0 * np.exp(growth_rate * day)
            else:
                v_at_treat = v0 * np.exp(growth_rate * treatment_day)
                dt = day - treatment_day
                effective_rate = growth_rate - treatment_effect
                v[i] = v_at_treat * np.exp(effective_rate * dt)
        return v

    # --- Logistic growth model ---
    def logistic_growth(t, v0, r, k, treatment_day, treatment_effect):
        """Logistic tumor growth with carrying capacity."""
        v = np.zeros_like(t, dtype=float)
        v[0] = v0
        for i in range(1, len(t)):
            rate = r if t[i] < treatment_day else r - treatment_effect
            dv = rate * v[i - 1] * (1 - v[i - 1] / k)
            v[i] = max(v[i - 1] + dv, 0.1)
        return v

    # --- Gompertz growth model ---
    def gompertz_growth(t, v0, a, b, treatment_day, treatment_effect):
        """Gompertz tumor growth model."""
        v = np.zeros_like(t, dtype=float)
        for i, day in enumerate(t):
            if day < treatment_day:
                v[i] = v0 * np.exp((a / b) * (1 - np.exp(-b * day)))
            else:
                v_at_treat = v0 * np.exp((a / b) * (1 - np.exp(-b * treatment_day)))
                dt = day - treatment_day
                a_eff = a - treatment_effect
                v[i] = v_at_treat * np.exp((a_eff / b) * (1 - np.exp(-b * dt)))
        return v

    # Treatment modalities with their effects
    treatments = {
        "No Treatment": {"color": "#9E9E9E", "dash": "dot", "effect": 0.0},
        "Chemotherapy": {"color": "#E53935", "dash": "solid", "effect": 0.015},
        "Immunotherapy": {"color": "#43A047", "dash": "solid", "effect": 0.012},
        "Targeted Therapy": {"color": "#1E88E5", "dash": "solid", "effect": 0.018},
        "Combination": {"color": "#8E24AA", "dash": "solid", "effect": 0.025},
    }

    # Panel layout: three columns using xaxis domains
    panel_gap = 0.04
    panel_width = (1.0 - 2 * panel_gap) / 3.0
    panels = [
        {"xaxis": "xaxis", "yaxis": "yaxis", "domain": [0, panel_width]},
        {"xaxis": "xaxis2", "yaxis": "yaxis2", "domain": [panel_width + panel_gap, 2 * panel_width + panel_gap]},
        {"xaxis": "xaxis3", "yaxis": "yaxis3", "domain": [2 * (panel_width + panel_gap), 1.0]},
    ]
    panel_titles = [
        "Exponential Growth Model",
        "Logistic Growth Model (K=100 cm\u00b3)",
        "Gompertz Growth Model",
    ]

    fig = go.Figure()
    np.random.seed(42)

    growth_funcs = [
        lambda tx_eff: exponential_growth(days, 2.0, 0.012, treatment_start, tx_eff),
        lambda tx_eff: logistic_growth(days, 2.0, 0.025, 100.0, treatment_start, tx_eff),
        lambda tx_eff: gompertz_growth(days, 2.0, 0.03, 0.008, treatment_start, tx_eff),
    ]

    axis_refs = [("x", "y"), ("x2", "y2"), ("x3", "y3")]

    for panel_idx, (growth_fn, (xref, yref)) in enumerate(zip(growth_funcs, axis_refs)):
        for tx_name, tx_data in treatments.items():
            v = growth_fn(tx_data["effect"])
            noise = np.random.normal(0, 0.2, len(days))
            v_noisy = np.maximum(v + noise * np.sqrt(np.abs(v)) * 0.04, 0.1)

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=v_noisy,
                    mode="lines",
                    name=tx_name,
                    line=dict(color=tx_data["color"], width=2.5, dash=tx_data["dash"]),
                    legendgroup=tx_name,
                    showlegend=(panel_idx == 0),
                    xaxis=xref,
                    yaxis=yref,
                    hovertemplate="Day %{x}<br>Volume: %{y:.1f} cm\u00b3<extra>" + tx_name + "</extra>",
                )
            )

        # Treatment initiation vertical line
        fig.add_shape(
            type="line",
            x0=treatment_start,
            x1=treatment_start,
            y0=0,
            y1=1,
            yref=yref + " domain",
            xref=xref,
            line=dict(color="#FFD600", width=2, dash="dash"),
        )

        # RECIST baseline horizontal line
        fig.add_shape(
            type="line",
            x0=0,
            x1=365,
            y0=20,
            y1=20,
            xref=xref,
            yref=yref,
            line=dict(color="#9E9E9E", width=1, dash="dot"),
        )

    # Configure axes for three panels
    y_range = [0, 120]
    fig.update_layout(
        xaxis=dict(title="Days", domain=panels[0]["domain"], gridcolor=grid_color),
        yaxis=dict(title="Tumor Volume (cm\u00b3)", gridcolor=grid_color, range=y_range),
        xaxis2=dict(title="Days", domain=panels[1]["domain"], gridcolor=grid_color, anchor="y2"),
        yaxis2=dict(gridcolor=grid_color, range=y_range, anchor="x2"),
        xaxis3=dict(title="Days", domain=panels[2]["domain"], gridcolor=grid_color, anchor="y3"),
        yaxis3=dict(gridcolor=grid_color, range=y_range, anchor="x3"),
    )

    # Panel title annotations
    for i, title in enumerate(panel_titles):
        center_x = (panels[i]["domain"][0] + panels[i]["domain"][1]) / 2.0
        fig.add_annotation(
            text=f"<b>{title}</b>",
            xref="paper",
            yref="paper",
            x=center_x,
            y=1.04,
            showarrow=False,
            font=dict(size=14),
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="Digital Twin Tumor Growth Trajectories Under Treatment Modalities",
            font=dict(size=22),
            x=0.5,
        ),
        width=1920,
        height=1080,
        legend=dict(
            x=0.42,
            y=-0.06,
            orientation="h",
            bgcolor="rgba(0,0,0,0.5)" if dark_mode else "rgba(255,255,255,0.85)",
            bordercolor="#555555" if dark_mode else "#CCCCCC",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(t=120, l=80, r=40, b=120),
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text=(
                    "Yellow dashed line = treatment initiation (Day 60) | "
                    "Gray dotted line = RECIST baseline (20 cm\u00b3) | "
                    "Initial tumor volume: 2.0 cm\u00b3"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.12,
                showarrow=False,
                font=dict(size=11),
            ),
        ],
    )

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    fig = create_oncology_trajectories(dark_mode=False)
    fig.write_html(str(output_dir / "10_oncology_trajectories.html"), include_plotlyjs="cdn")
    fig.write_image(str(output_dir / "10_oncology_trajectories.png"), width=1920, height=1080, scale=2)
    print("Saved 10_oncology_trajectories.html and .png")
    fig_dark = create_oncology_trajectories(dark_mode=True)
    fig_dark.write_html(str(output_dir / "10_oncology_trajectories_dark.html"), include_plotlyjs="cdn")
    fig_dark.write_image(str(output_dir / "10_oncology_trajectories_dark.png"), width=1920, height=1080, scale=2)
    print("Saved 10_oncology_trajectories_dark.html and _dark.png")
