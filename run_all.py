"""
Run all experiments sequentially.
4 groups:
  1. Baseline: SAC + ADRC (absolute params, linear ESO)
  2. Residual RL + ADRC (linear ESO)
  3. Residual RL + ADRC (Neural ESO)  <-- novel
  4. Residual RL + ADRC-SMC composite (Neural ESO)  <-- novel
Then evaluate + disturbance test.
"""
import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode


def main():
    # Group 1: Baseline (absolute params, no residual)
    run("python train/train_sac.py --mode adrc --no_residual "
        "--timesteps 200000 --seed 42")

    # Group 2: Residual RL + linear ESO
    run("python train/train_sac.py --mode adrc --residual "
        "--timesteps 200000 --seed 42")

    # Group 3: Residual RL + Neural ESO (novel)
    run("python train/train_sac.py --mode adrc --residual --neural_eso "
        "--timesteps 200000 --seed 42")

    # Group 4: Residual RL + ADRC-SMC + linear ESO
    run("python train/train_sac.py --mode composite --residual "
        "--timesteps 200000 --seed 42")

    # Evaluate
    run("python eval/evaluate.py "
        "--adrc_model results/sac_adrc_residual/sac_final "
        "--composite_model results/sac_composite_residual/sac_final")

    # Disturbance test
    run("python eval/disturbance_test.py "
        "--adrc_model results/sac_adrc_residual/sac_final "
        "--composite_model results/sac_composite_residual/sac_final")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {os.path.join(PROJECT_DIR, 'results')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
