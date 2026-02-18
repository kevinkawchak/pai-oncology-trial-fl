"""Generate all HTML and PNG files for all 3 sets of interactive visualizations."""

import importlib.util
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).parent

SETS = {
    "1st": [
        ("01_repository_architecture_treemap", "create_repository_architecture_treemap"),
        ("02_clinical_trial_workflow", "create_clinical_trial_workflow"),
        ("03_oncology_process_diagram", "create_oncology_process_diagram"),
        ("04_framework_comparison_radar", "create_framework_comparison_radar"),
        ("05_parameter_mapping_heatmap", "create_parameter_mapping_heatmap"),
        ("06_data_pipeline_throughput", "create_data_pipeline_throughput"),
        ("07_model_format_comparison", "create_model_format_comparison"),
        ("08_state_vector_visualization", "create_state_vector_visualization"),
        ("09_domain_randomization_transfer", "create_domain_randomization_transfer"),
        ("10_oncology_trajectories", "create_oncology_trajectories"),
    ],
    "2nd": [
        ("01_training_curves", "create_training_curves"),
        ("02_benchmark_comparisons", "create_benchmark_comparisons"),
        ("03_safety_metrics_dashboard", "create_safety_metrics_dashboard"),
        ("04_performance_radars", "create_performance_radars"),
        ("05_accuracy_confusion_matrix", "create_accuracy_confusion_matrix"),
        ("06_sim_to_real_transfer", "create_sim_to_real_transfer"),
        ("07_latency_distributions", "create_latency_distributions"),
        ("08_resource_utilization", "create_resource_utilization"),
        ("09_oncology_kpi_tracker", "create_oncology_kpi_tracker"),
        ("10_cross_framework_consistency", "create_cross_framework_consistency"),
    ],
    "3rd": [
        ("01_oncology_convergence", "create_oncology_convergence"),
        ("02_multi_site_trial_dashboard", "create_multi_site_trial_dashboard"),
        ("03_algorithm_comparison_radar", "create_algorithm_comparison_radar"),
        ("04_fda_device_classification_tree", "create_fda_device_classification_tree"),
        ("05_oncology_device_distribution", "create_oncology_device_distribution"),
        ("06_regulatory_compliance_scorecard", "create_regulatory_compliance_scorecard"),
        ("07_hipaa_phi_detection_matrix", "create_hipaa_phi_detection_matrix"),
        ("08_privacy_analytics_pipeline", "create_privacy_analytics_pipeline"),
        ("09_deployment_readiness_radar", "create_deployment_readiness_radar"),
        ("10_production_readiness_scores", "create_production_readiness_scores"),
    ],
}


def load_module(set_name, module_name):
    """Dynamically import a module from a set directory."""
    path = BASE / set_name / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(output_type="all"):
    """Generate outputs. output_type: 'html', 'png', or 'all'."""
    errors = []
    total = 0
    for set_name, scripts in SETS.items():
        for module_name, func_name in scripts:
            output_dir = BASE / set_name
            try:
                mod = load_module(set_name, module_name)
                create_fn = getattr(mod, func_name)

                for dark_mode, suffix in [(False, ""), (True, "_dark")]:
                    fig = create_fn(dark_mode=dark_mode)
                    fname = f"{module_name}{suffix}"

                    if output_type in ("html", "all"):
                        fig.write_html(
                            str(output_dir / f"{fname}.html"),
                            include_plotlyjs="cdn",
                        )
                        total += 1
                        print(f"  HTML: {set_name}/{fname}.html")

                    if output_type in ("png", "all"):
                        fig.write_image(
                            str(output_dir / f"{fname}.png"),
                            width=1920,
                            height=1080,
                            scale=2,
                        )
                        total += 1
                        print(f"  PNG:  {set_name}/{fname}.png")

            except Exception as e:
                errors.append((set_name, module_name, str(e)))
                traceback.print_exc()
                print(f"  ERROR: {set_name}/{module_name}: {e}")

    print(f"\nGenerated {total} files. Errors: {len(errors)}")
    for s, m, e in errors:
        print(f"  FAILED: {s}/{m}: {e}")
    return len(errors)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("html", "png", "all"):
        print(f"Usage: {sys.argv[0]} [html|png|all]")
        sys.exit(1)
    print(f"Generating {mode} files for all 3 sets...")
    err_count = generate(mode)
    sys.exit(1 if err_count else 0)
