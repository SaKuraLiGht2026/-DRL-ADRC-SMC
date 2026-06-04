"""Run evaluation and disturbance test only (training already done)."""
import subprocess, os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

def run(cmd):
    print(f"\n{'='*60}\nRunning: {cmd}\n{'='*60}")
    subprocess.run(cmd, shell=True)

run("python eval/evaluate.py")
run("python eval/disturbance_test.py")
print("\nAll evaluation completed!")
