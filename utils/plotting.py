"""Plotting utilities for evaluation results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_tracking_comparison(results_dict, save_path, title="Trajectory Tracking Comparison"):
    """
    Plot tracking comparison for multiple methods.
    results_dict: {method_name: {"time": [...], "ref": [...], "actual": [...], "error": [...]}}
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    # Plot reference (same for all)
    first_key = list(results_dict.keys())[0]
    axes[0].plot(results_dict[first_key]["time"],
                 results_dict[first_key]["ref"],
                 "k--", linewidth=2, label="Reference", alpha=0.7)
    
    for i, (name, data) in enumerate(results_dict.items()):
        axes[0].plot(data["time"], data["actual"],
                     color=colors[i], linewidth=1.5, label=name)
        axes[1].plot(data["time"], data["error"],
                     color=colors[i], linewidth=1.5, label=name)
    
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Tracking Error (rad)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_torque_comparison(results_dict, save_path):
    """Plot torque signals comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (name, data) in enumerate(results_dict.items()):
        ax.plot(data["time"], data["torque"],
                color=colors[i], linewidth=1.0, label=name, alpha=0.8)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Torque (N·m)")
    ax.set_title("Control Torque Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_adrc_params(results_dict, save_path):
    """Plot ADRC parameter adaptation over time (DRL agent output)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    for name, data in results_dict.items():
        if "omega_o" in data and "omega_c" in data:
            axes[0].plot(data["time"], data["omega_o"],
                         linewidth=1.5, label=name)
            axes[1].plot(data["time"], data["omega_c"],
                         linewidth=1.5, label=name)
    
    axes[0].set_ylabel("ω_o (Observer Bandwidth)")
    axes[0].set_title("ADRC Parameter Adaptation by DRL Agent")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ω_c (Controller Bandwidth)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_disturbance_bar(metrics_dict, save_path):
    """
    Bar chart comparing RMSE across methods and disturbance scenarios.
    metrics_dict: {scenario: {method: rmse_value}}
    """
    scenarios = list(metrics_dict.keys())
    methods = list(metrics_dict[scenarios[0]].keys())
    
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        values = [metrics_dict[s][method] for s in scenarios]
        ax.bar(x + i * width - 0.4 + width / 2, values, width,
               label=method, color=colors[i], edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Disturbance Scenario")
    ax.set_ylabel("RMSE (rad)")
    ax.set_title("Robustness Comparison Under Disturbances")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
