import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_script(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_data_creation() -> None:
    _run_script("main.py")
    _run_script("analytics/data_analysis/generate_data_dictionary.py")
    _run_script("analytics/data_analysis/variable_band_charts.py")
    _run_script("analytics/data_analysis/split_train_test.py")
    _run_script("analytics/data_analysis/variable_selection.py")
    _run_script("analytics/data_analysis/woe_encode.py")
    print("\nData creation pipeline completed.")


def run_modeling() -> None:
    use_woe = input("Use WOE datasets for model training? (y/n) [y]: ").strip().lower()
    model_args = ["--use-woe"] if use_woe in ("", "y", "yes") else []

    _run_script("analytics/data_analysis/train_logistic_model.py", model_args)
    _run_script("analytics/data_analysis/train_neural_network_model.py", model_args)
    _run_script("analytics/data_analysis/train_cox_model.py", model_args)
    _run_script("analytics/data_analysis/compare_models.py")
    print("\nModeling pipeline completed.")


def run_archive() -> None:
    dry_run = input("Run archive in dry-run mode first? (y/n) [n]: ").strip().lower()
    archive_args = ["--dry-run"] if dry_run in ("y", "yes") else []
    _run_script("analytics/data_analysis/archive_outputs.py", archive_args)
    print("\nArchive step completed.")


def main() -> None:
    while True:
        print("\nChoose pipeline:")
        print("1) Data creation")
        print("2) Model training + comparison")
        print("3) Archive outputs")
        print("4) Exit")
        choice = input("Enter 1, 2, 3 or 4: ").strip()

        if choice == "1":
            run_data_creation()
        elif choice == "2":
            run_modeling()
        elif choice == "3":
            run_archive()
        elif choice == "4":
            print("Exiting pipeline menu.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
