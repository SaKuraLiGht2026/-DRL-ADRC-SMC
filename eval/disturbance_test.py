"""
Disturbance robustness test with all methods.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from controllers.adrc import ADRC
from controllers.pid import PIDController
from controllers.sliding_mode import NonsingularTerminalSMC
from configs.default import DISTURBANCE_CONFIGS
from eval.evaluate import run_classic_controller, run_drl_model, compute_metrics
from utils.plotting import plot_disturbance_bar, plot_tracking_comparison


def main():
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "disturbance"
    )
    os.makedirs(result_dir, exist_ok=True)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define methods
    methods = {}

    methods["PID"] = {"type": "classic", "controller": PIDController(kp=50, ki=5, kd=10, dt=0.005)}
    methods["Fixed-ADRC"] = {"type": "classic", "controller": ADRC(omega_o=20, omega_c=10, dt=0.005)}

    res_path = os.path.join(project_dir, "results/sac_adrc_residual/sac_final")
    if os.path.exists(res_path + ".zip"):
        methods["DRL-ADRC"] = {"type": "drl", "model_path": res_path,
                               "control_mode": "adrc", "use_neural_eso": False, "use_residual": True}

    comp_path = os.path.join(project_dir, "results/sac_composite_residual/sac_final")
    if os.path.exists(comp_path + ".zip"):
        methods["DRL-ADRC-SMC"] = {"type": "drl", "model_path": comp_path,
                                   "control_mode": "composite", "use_neural_eso": False, "use_residual": True}

    all_metrics = {}

    for scenario_name, dist_config in DISTURBANCE_CONFIGS.items():
        print(f"\n=== Scenario: {scenario_name} ===")
        all_metrics[scenario_name] = {}
        scenario_results = {}

        for method_name, cfg in methods.items():
            print(f"  Running {method_name}...")
            if cfg["type"] == "classic":
                cfg["controller"].reset()
                records = run_classic_controller(
                    cfg["controller"], disturbance_config=dist_config)
            else:
                records = run_drl_model(
                    cfg["model_path"], control_mode=cfg["control_mode"],
                    use_neural_eso=cfg["use_neural_eso"],
                    use_residual=cfg["use_residual"],
                    disturbance_config=dist_config)

            metrics = compute_metrics(records)
            all_metrics[scenario_name][method_name] = metrics["RMSE"]
            scenario_results[method_name] = records
            print(f"    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

        plot_tracking_comparison(
            scenario_results,
            os.path.join(result_dir, f"tracking_{scenario_name}.png"),
            title=f"Tracking Under {scenario_name} Disturbance")

    plot_disturbance_bar(all_metrics,
        os.path.join(result_dir, "disturbance_rmse_bar.png"))

    # Print table
    print("\n" + "=" * 90)
    scenarios = list(all_metrics.keys())
    methods_list = list(methods.keys())
    header = f"{'Scenario':<20}" + "".join(f"{m:>15}" for m in methods_list)
    print(header)
    print("-" * 90)
    for scenario in scenarios:
        row = f"{scenario:<20}"
        for method in methods_list:
            val = all_metrics[scenario].get(method, float("nan"))
            row += f"{val:>15.4f}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
