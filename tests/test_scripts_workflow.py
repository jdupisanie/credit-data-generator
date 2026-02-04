import json
import sys
import importlib.util
from pathlib import Path
from uuid import uuid4

import pandas as pd

def _run_script_main(module_path: Path, args: list[str], monkeypatch, cwd: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"testmod_{uuid4().hex}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", [str(module_path)] + args)
    module.main()


def test_main_generates_dataset_and_metadata(
    tmp_path: Path,
    mini_variables_config: dict,
    mini_global_config: dict,
    write_json,
    monkeypatch,
    project_root: Path,
):
    write_json(tmp_path / "input_parameters" / "variables.json", mini_variables_config)
    write_json(tmp_path / "input_parameters" / "global_parameters.json", mini_global_config)

    spec = importlib.util.spec_from_file_location("main_under_test", project_root / "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(main_mod)
    monkeypatch.chdir(tmp_path)
    main_mod.main()

    dataset_path = tmp_path / "outputs" / "simulator" / "simulated_dataset.csv"
    metadata_path = tmp_path / "outputs" / "simulator" / "simulated_metadata.json"
    assert dataset_path.exists()
    assert metadata_path.exists()

    df = pd.read_csv(dataset_path)
    assert len(df) == mini_global_config["simulation_population"]
    assert set(["age_years", "residential_status", "default"]).issubset(df.columns)
    assert df["age_years"].dtype != object  # numeric variable output should be value, not band

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(meta["variables"]) == 2
    assert "final_default" in meta


def test_split_train_test_uses_global_ratio(
    tmp_path: Path,
    mini_dataset: pd.DataFrame,
    write_json,
    monkeypatch,
    project_root: Path,
):
    input_csv = tmp_path / "sim.csv"
    mini_dataset.to_csv(input_csv, index=False)
    write_json(
        tmp_path / "global.json",
        {
            "simulation_population": len(mini_dataset),
            "global_bad_rate_pct": 10.0,
            "train_set_pct": 75.0,
            "test_set_pct": 25.0,
        },
    )
    out_dir = tmp_path / "out"

    _run_script_main(
        project_root / "analytics" / "data_analysis" / "split_train_test.py",
        ["--input", str(input_csv), "--global-params", str(tmp_path / "global.json"), "--output-dir", str(out_dir)],
        monkeypatch,
        tmp_path,
    )

    train_df = pd.read_csv(out_dir / "simulated_dataset_train.csv")
    test_df = pd.read_csv(out_dir / "simulated_dataset_test.csv")
    assert len(train_df) == 90
    assert len(test_df) == 30


def test_generate_data_dictionary_and_variable_selection(
    tmp_path: Path,
    mini_variables_config: dict,
    mini_dataset: pd.DataFrame,
    write_json,
    monkeypatch,
    project_root: Path,
):
    cfg_path = tmp_path / "variables.json"
    write_json(cfg_path, mini_variables_config)
    train_path = tmp_path / "train.csv"
    mini_dataset.to_csv(train_path, index=False)

    doc_out = tmp_path / "docs"
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "generate_data_dictionary.py",
        ["--variables-config", str(cfg_path), "--output-dir", str(doc_out)],
        monkeypatch,
        tmp_path,
    )
    assert (doc_out / "data_dictionary.md").exists()
    assert (doc_out / "data_dictionary.html").exists()

    sel_out = tmp_path / "selection" / "report.csv"
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "variable_selection.py",
        [
            "--input",
            str(train_path),
            "--variables-config",
            str(cfg_path),
            "--output",
            str(sel_out),
            "--cv-folds",
            "3",
        ],
        monkeypatch,
        tmp_path,
    )
    report = pd.read_csv(sel_out)
    assert {"variable", "wald_chi2", "lr_chi2", "information_value"}.issubset(report.columns)
    assert set(report["variable"]) == {"age_years", "residential_status"}


def test_woe_encode_uses_train_mapping_for_test(
    tmp_path: Path,
    mini_variables_config: dict,
    mini_dataset: pd.DataFrame,
    write_json,
    monkeypatch,
    project_root: Path,
):
    train_df = mini_dataset.iloc[:80].copy()
    test_df = mini_dataset.iloc[80:].copy()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    cfg_path = tmp_path / "variables.json"
    write_json(cfg_path, mini_variables_config)

    out_dir = tmp_path / "woe"
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "woe_encode.py",
        [
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--variables-config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
        ],
        monkeypatch,
        tmp_path,
    )

    train_woe = pd.read_csv(out_dir / "simulated_dataset_train_woe.csv")
    test_woe = pd.read_csv(out_dir / "simulated_dataset_test_woe.csv")
    mapping = json.loads((out_dir / "woe_mappings.json").read_text(encoding="utf-8"))

    assert len(train_woe) == len(train_df)
    assert len(test_woe) == len(test_df)
    assert "variables" in mapping
    age_bands = mapping["variables"]["age_years"]["bands"]
    age_mapping = {item["band"]: item["woe"] for item in age_bands}

    # Verify a test-row WOE value matches a train-derived mapping band.
    sample_age = float(test_df.iloc[0]["age_years"])
    expected_band = "<30" if sample_age < 30 else ("30-50" if sample_age <= 50 else ">50")
    assert test_woe.iloc[0]["age_years"] == age_mapping[expected_band]


def test_training_and_comparison_for_all_model_types(
    tmp_path: Path,
    mini_dataset: pd.DataFrame,
    monkeypatch,
    project_root: Path,
):
    train_df = mini_dataset.iloc[:90].copy()
    test_df = mini_dataset.iloc[90:].copy()
    dataset_dir = tmp_path / "artifacts" / "01_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "simulated_dataset_train.csv"
    test_path = dataset_dir / "simulated_dataset_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    models_dir = tmp_path / "models"
    registry = tmp_path / "comparison" / "model_registry.csv"

    _run_script_main(
        project_root / "analytics" / "data_analysis" / "train_logistic_model.py",
        [
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--output-dir",
            str(models_dir),
            "--results-csv",
            str(registry),
            "--model-name",
            "logit_test",
        ],
        monkeypatch,
        tmp_path,
    )
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "train_neural_network_model.py",
        [
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--output-dir",
            str(models_dir),
            "--results-csv",
            str(registry),
            "--model-name",
            "nn_test",
            "--hidden-layers",
            "8",
        ],
        monkeypatch,
        tmp_path,
    )
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "train_cox_model.py",
        [
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--output-dir",
            str(models_dir),
            "--results-csv",
            str(registry),
            "--model-name",
            "cox_test",
            "--max-iter",
            "100",
        ],
        monkeypatch,
        tmp_path,
    )

    comparison_out = tmp_path / "comparison"
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "compare_models.py",
        [
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(comparison_out),
            "--split",
            "test",
        ],
        monkeypatch,
        tmp_path,
    )

    cmp_df = pd.read_csv(comparison_out / "comparison_test.csv")
    assert {"logit_test", "nn_test", "cox_test"}.issubset(set(cmp_df["model_name"]))
    assert (comparison_out / "roc_curves_test.json").exists()
    assert (comparison_out / "confusion_matrices_test.json").exists()
    assert (comparison_out / "roc_comparison_test.png").exists()


def test_variable_band_charts_creates_pngs(
    tmp_path: Path,
    mini_variables_config: dict,
    write_json,
    monkeypatch,
    project_root: Path,
):
    out_sim = tmp_path / "outputs" / "simulator"
    out_sim.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "age_years": [22, 28, 44, 58, 36, 67],
            "residential_status": ["owner", "rent", "owner", "rent", "owner", "rent"],
            "default": [1, 0, 0, 1, 0, 1],
        }
    )
    df.to_csv(out_sim / "simulated_dataset.csv", index=False)

    metadata = {
        "variables": [
            {"name": "age_years", "bands": ["<30", "30-50", ">50"]},
            {"name": "residential_status", "bands": ["owner", "rent"]},
        ]
    }
    write_json(out_sim / "simulated_metadata.json", metadata)
    write_json(tmp_path / "input_parameters" / "variables.json", mini_variables_config)

    monkeypatch.setenv("MPLBACKEND", "Agg")
    _run_script_main(
        project_root / "analytics" / "data_analysis" / "variable_band_charts.py",
        [],
        monkeypatch,
        tmp_path,
    )

    vis_dir = tmp_path / "analytics" / "data_analysis" / "artifacts" / "06_visualizations"
    assert (vis_dir / "age_years.png").exists()
    assert (vis_dir / "residential_status.png").exists()


def test_archive_outputs_dry_run_and_move(
    tmp_path: Path,
    monkeypatch,
    project_root: Path,
):
    sim_dir = tmp_path / "outputs" / "simulator"
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "simulated_dataset.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    artifacts_dir = tmp_path / "analytics" / "data_analysis" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "x.txt").write_text("x", encoding="utf-8")

    archive_root = tmp_path / "archive"
    script = project_root / "analytics" / "data_analysis" / "archive_outputs.py"

    _run_script_main(script, ["--archive-root", str(archive_root), "--dry-run"], monkeypatch, tmp_path)
    assert sim_dir.exists()

    _run_script_main(script, ["--archive-root", str(archive_root)], monkeypatch, tmp_path)
    assert not sim_dir.exists()
    manifests = list(archive_root.glob("cleanup_*/archive_manifest.json"))
    assert manifests, "Expected archive manifest in timestamped archive folder."
