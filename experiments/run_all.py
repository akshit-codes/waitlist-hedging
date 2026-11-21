"""Run every experiment in turn and refresh results/. Takes a few minutes."""

import subprocess
import sys

EXPERIMENTS = ["q1_dilemma", "q2_decomposition", "q3_cap", "q4_refund", "q5_chart",
               "ranking", "rule_check", "learn", "equilibrium_check", "chart_release",
               "sensitivity"]


def main():
    for name in EXPERIMENTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, "-m", f"experiments.{name}", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
