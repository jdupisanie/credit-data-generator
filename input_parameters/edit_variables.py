import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive editor for input_parameters/variables.json"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("input_parameters") / "variables.json",
        help="Path to variables.json",
    )
    parser.add_argument(
        "--global-path",
        type=Path,
        default=Path("input_parameters") / "global_parameters.json",
        help="Path to global_parameters.json",
    )
    return parser.parse_args()


def _as_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def _validate_variable(var: dict) -> list[str]:
    errors = []
    name = var.get("name", "<unknown>")
    bands = var.get("bands", [])
    if not bands:
        errors.append(f"{name}: no bands configured.")
        return errors

    dist_sum = 0.0
    for band in bands:
        band_name = band.get("band", "<unknown>")
        dist = _as_float(band.get("distribution_pct"), "distribution_pct")
        ratio = _as_float(band.get("bad_rate_ratio"), "bad_rate_ratio")
        if dist < 0:
            errors.append(f"{name}/{band_name}: distribution_pct must be >= 0.")
        if ratio <= 0:
            errors.append(f"{name}/{band_name}: bad_rate_ratio must be > 0.")
        dist_sum += dist

    if abs(dist_sum - 100.0) > 1e-6:
        errors.append(f"{name}: distribution_pct sum is {dist_sum:.6f}, expected 100.0.")

    return errors


def _validate_all(cfg: dict) -> list[str]:
    errors = []
    dataset_spec = cfg.get("dataset_spec", {})
    variables = dataset_spec.get("variables", [])
    if not variables:
        return ["No variables found under dataset_spec.variables."]

    names = [v.get("name") for v in variables]
    if len(names) != len(set(names)):
        errors.append("Duplicate variable names detected.")

    for var in variables:
        errors.extend(_validate_variable(var))

    return errors


def _validate_global_params(global_cfg: dict) -> list[str]:
    errors = []
    if "simulation_population" not in global_cfg:
        errors.append("global_parameters: missing simulation_population.")
    else:
        try:
            n = int(global_cfg["simulation_population"])
            if n <= 0:
                errors.append("global_parameters: simulation_population must be > 0.")
        except (TypeError, ValueError):
            errors.append("global_parameters: simulation_population must be an integer.")

    if "global_bad_rate_pct" not in global_cfg:
        errors.append("global_parameters: missing global_bad_rate_pct.")
    else:
        try:
            br = float(global_cfg["global_bad_rate_pct"])
            if br <= 0 or br >= 100:
                errors.append("global_parameters: global_bad_rate_pct must be between 0 and 100.")
        except (TypeError, ValueError):
            errors.append("global_parameters: global_bad_rate_pct must be numeric.")

    train_pct = global_cfg.get("train_set_pct")
    test_pct = global_cfg.get("test_set_pct")
    if train_pct is None:
        errors.append("global_parameters: missing train_set_pct.")
    if test_pct is None:
        errors.append("global_parameters: missing test_set_pct.")
    if train_pct is not None and test_pct is not None:
        try:
            tr = float(train_pct)
            te = float(test_pct)
            if tr <= 0 or te <= 0:
                errors.append("global_parameters: train_set_pct and test_set_pct must be > 0.")
            if abs((tr + te) - 100.0) > 1e-6:
                errors.append("global_parameters: train_set_pct and test_set_pct must sum to 100.")
        except (TypeError, ValueError):
            errors.append("global_parameters: train_set_pct and test_set_pct must be numeric.")

    return errors


def _edit_global_params(global_cfg: dict) -> bool:
    changed = False
    while True:
        print("\nGlobal parameters:")
        for key, val in global_cfg.items():
            print(f"- {key}: {val}")
        print("\nGlobal edit options:")
        print("1) Edit simulation_population")
        print("2) Edit global_bad_rate_pct")
        print("3) Edit train_set_pct")
        print("4) Edit test_set_pct")
        print("5) Back")
        choice = input("Select option: ").strip()

        if choice == "5":
            return changed
        if choice not in {"1", "2", "3", "4"}:
            print("Invalid option.")
            continue

        key_map = {
            "1": "simulation_population",
            "2": "global_bad_rate_pct",
            "3": "train_set_pct",
            "4": "test_set_pct",
        }
        key = key_map[choice]
        raw = input(f"New value for {key} (current {global_cfg.get(key)}): ").strip()
        try:
            value = int(raw) if key == "simulation_population" else float(raw)
        except ValueError:
            print("Invalid numeric value.")
            continue

        global_cfg[key] = value
        changed = True
        print(f"Updated {key}.")


def _print_variables(variables: list[dict]) -> None:
    print("\nVariables:")
    for idx, var in enumerate(variables, start=1):
        name = var.get("name", "<unknown>")
        var_type = var.get("type", "unknown")
        band_count = len(var.get("bands", []))
        print(f"{idx:>2}. {name} ({var_type}, {band_count} bands)")


def _print_bands(var: dict) -> None:
    name = var.get("name", "<unknown>")
    print(f"\n{name} bands:")
    print(" idx | band                       | distribution_pct | bad_rate_ratio")
    print("-----+----------------------------+------------------+---------------")
    dist_sum = 0.0
    for idx, band in enumerate(var.get("bands", []), start=1):
        band_name = str(band.get("band", ""))[:26]
        dist = float(band.get("distribution_pct", 0.0))
        ratio = float(band.get("bad_rate_ratio", 0.0))
        dist_sum += dist
        print(f"{idx:>4} | {band_name:<26} | {dist:>16.6f} | {ratio:>13.6f}")
    print(f"Total distribution_pct: {dist_sum:.6f}")
    if abs(dist_sum - 100.0) > 1e-6:
        print("WARNING: distribution_pct does not sum to 100.")


def _choose_index(max_value: int, prompt: str) -> int | None:
    raw = input(prompt).strip().lower()
    if raw in {"q", "quit", "back", "b"}:
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("Invalid input. Enter a number or 'q' to go back.")
        return None
    if idx < 1 or idx > max_value:
        print(f"Out of range. Choose between 1 and {max_value}.")
        return None
    return idx - 1


def _edit_variable(var: dict) -> None:
    while True:
        _print_bands(var)
        print("\nEdit options:")
        print("1) Edit distribution_pct")
        print("2) Edit bad_rate_ratio")
        print("3) Back")
        choice = input("Select option: ").strip()

        if choice == "3":
            break
        if choice not in {"1", "2"}:
            print("Invalid option.")
            continue

        bands = var.get("bands", [])
        band_idx = _choose_index(len(bands), "Band index (or q to cancel): ")
        if band_idx is None:
            continue

        field = "distribution_pct" if choice == "1" else "bad_rate_ratio"
        current = bands[band_idx].get(field)
        raw = input(f"New value for {field} (current {current}): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Invalid number.")
            continue

        if field == "distribution_pct" and value < 0:
            print("distribution_pct must be >= 0.")
            continue
        if field == "bad_rate_ratio" and value <= 0:
            print("bad_rate_ratio must be > 0.")
            continue

        bands[band_idx][field] = value
        print(f"Updated {field} for band '{bands[band_idx].get('band')}'.")


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"Missing config file: {args.path}")
    if not args.global_path.exists():
        raise FileNotFoundError(f"Missing global config file: {args.global_path}")

    cfg = json.loads(args.path.read_text(encoding="utf-8"))
    global_cfg = json.loads(args.global_path.read_text(encoding="utf-8"))
    variables = cfg.get("dataset_spec", {}).get("variables", [])
    if not variables:
        raise ValueError("No variables found in dataset_spec.variables")
    has_unsaved_changes = False

    while True:
        print("\nVariable Input Editor")
        print("1) List variables")
        print("2) Edit variable")
        print("3) Edit global parameters")
        print("4) Validate")
        print("5) Save")
        exit_label = "Exit without saving" if has_unsaved_changes else "Exit"
        print(f"6) {exit_label}")
        choice = input("Choose option: ").strip()

        if choice == "1":
            _print_variables(variables)
        elif choice == "2":
            _print_variables(variables)
            var_idx = _choose_index(len(variables), "Variable index (or q to cancel): ")
            if var_idx is not None:
                _edit_variable(variables[var_idx])
                has_unsaved_changes = True
        elif choice == "3":
            changed = _edit_global_params(global_cfg)
            has_unsaved_changes = has_unsaved_changes or changed
        elif choice == "4":
            errors = _validate_all(cfg) + _validate_global_params(global_cfg)
            if errors:
                print("\nValidation errors:")
                for err in errors:
                    print(f"- {err}")
            else:
                print("Validation passed.")
        elif choice == "5":
            errors = _validate_all(cfg) + _validate_global_params(global_cfg)
            if errors:
                print("\nCannot save due to validation errors:")
                for err in errors:
                    print(f"- {err}")
                continue

            backup_path = args.path.with_suffix(args.path.suffix + ".bak")
            backup_path.write_text(args.path.read_text(encoding="utf-8"), encoding="utf-8")
            args.path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            global_backup_path = args.global_path.with_suffix(args.global_path.suffix + ".bak")
            global_backup_path.write_text(
                args.global_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            args.global_path.write_text(json.dumps(global_cfg, indent=2) + "\n", encoding="utf-8")
            print(f"Saved: {args.path}")
            print(f"Backup: {backup_path}")
            print(f"Saved: {args.global_path}")
            print(f"Backup: {global_backup_path}")
            has_unsaved_changes = False
        elif choice == "6":
            if has_unsaved_changes:
                print("Exited without saving.")
            else:
                print("Exited.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
