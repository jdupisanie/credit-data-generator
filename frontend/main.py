from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = FRONTEND_ROOT / "templates"
STATIC_DIR = FRONTEND_ROOT / "static"
VARIABLES_PATH = PROJECT_ROOT / "input_parameters" / "variables.json"
VARIABLES_FACTORY_PATH = PROJECT_ROOT / "input_parameters" / "variables.factory.json"
GLOBAL_PARAMETERS_PATH = PROJECT_ROOT / "input_parameters" / "global_parameters.json"

DATA_CREATION_SCRIPTS = [
    "main.py",
    "analytics/data_analysis/generate_data_dictionary.py",
    "analytics/data_analysis/variable_band_charts.py",
    "analytics/data_analysis/split_train_test.py",
    "analytics/data_analysis/variable_selection.py",
    "analytics/data_analysis/woe_encode.py",
]
MODELING_SCRIPTS = [
    "analytics/data_analysis/train_logistic_model.py",
    "analytics/data_analysis/train_neural_network_model.py",
    "analytics/data_analysis/train_cox_model.py",
    "analytics/data_analysis/compare_models.py",
]
ARCHIVE_SCRIPT = "analytics/data_analysis/archive_outputs.py"


class Band(BaseModel):
    model_config = ConfigDict(extra="allow")

    band: str
    distribution_pct: float
    bad_rate_ratio: float


class Variable(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    bands: list[Band] = Field(default_factory=list)
    include_in_data_creation: bool = True


class ConfigPayload(BaseModel):
    global_parameters: dict[str, Any]
    variables: list[Variable]


class PipelineRequest(BaseModel):
    pipeline: Literal["data_creation", "modeling", "archive"]
    use_woe: bool = True
    dry_run: bool = False


app = FastAPI(title="Credit Risk Modeling Frontend")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_job_store: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_variables_json() -> dict[str, Any]:
    if not VARIABLES_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Missing file: {VARIABLES_PATH}")
    return json.loads(VARIABLES_PATH.read_text(encoding="utf-8"))


def _read_global_json() -> dict[str, Any]:
    if not GLOBAL_PARAMETERS_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Missing file: {GLOBAL_PARAMETERS_PATH}")
    return json.loads(GLOBAL_PARAMETERS_PATH.read_text(encoding="utf-8"))


def _read_factory_variables_json() -> dict[str, Any]:
    if not VARIABLES_FACTORY_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Missing factory file: {VARIABLES_FACTORY_PATH}")
    return json.loads(VARIABLES_FACTORY_PATH.read_text(encoding="utf-8"))


def _write_json_with_backup(path: Path, payload: dict[str, Any]) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def _validate_variable(var: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = str(var.get("name", "<unknown>"))
    bands = var.get("bands", [])
    if not isinstance(bands, list) or not bands:
        return [f"{name}: no bands configured."]

    dist_sum = 0.0
    for band in bands:
        band_name = str(band.get("band", "<unknown>"))
        try:
            dist = _as_float(band.get("distribution_pct"), "distribution_pct")
            ratio = _as_float(band.get("bad_rate_ratio"), "bad_rate_ratio")
        except ValueError as exc:
            errors.append(f"{name}/{band_name}: {exc}")
            continue

        if dist < 0:
            errors.append(f"{name}/{band_name}: distribution_pct must be >= 0.")
        if ratio <= 0:
            errors.append(f"{name}/{band_name}: bad_rate_ratio must be > 0.")
        dist_sum += dist

    if abs(dist_sum - 100.0) > 1e-6:
        errors.append(f"{name}: distribution_pct sum is {dist_sum:.6f}, expected 100.0.")
    return errors


def _validate_all(variables: list[dict[str, Any]], global_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not variables:
        errors.append("No variables found under dataset_spec.variables.")
    else:
        names = [str(v.get("name")) for v in variables]
        if len(names) != len(set(names)):
            errors.append("Duplicate variable names detected.")
        for var in variables:
            errors.extend(_validate_variable(var))

    if "simulation_population" not in global_cfg:
        errors.append("global_parameters: missing simulation_population.")
    else:
        try:
            population = int(global_cfg["simulation_population"])
            if population <= 0:
                errors.append("global_parameters: simulation_population must be > 0.")
        except (TypeError, ValueError):
            errors.append("global_parameters: simulation_population must be an integer.")

    if "global_bad_rate_pct" not in global_cfg:
        errors.append("global_parameters: missing global_bad_rate_pct.")
    else:
        try:
            bad_rate = float(global_cfg["global_bad_rate_pct"])
            if bad_rate <= 0 or bad_rate >= 100:
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
            train = float(train_pct)
            test = float(test_pct)
            if train <= 0 or test <= 0:
                errors.append("global_parameters: train_set_pct and test_set_pct must be > 0.")
            if abs((train + test) - 100.0) > 1e-6:
                errors.append("global_parameters: train_set_pct and test_set_pct must sum to 100.")
        except (TypeError, ValueError):
            errors.append("global_parameters: train_set_pct and test_set_pct must be numeric.")
    return errors


def _is_variable_included(var: dict[str, Any]) -> bool:
    return bool(var.get("include_in_data_creation", True))


def _count_included_variables(variables: list[dict[str, Any]]) -> int:
    return sum(1 for item in variables if _is_variable_included(item))


def _append_job_log(job_id: str, message: str) -> None:
    with _job_lock:
        job = _job_store.get(job_id)
        if not job:
            return
        job["logs"].append(message.rstrip())
        if len(job["logs"]) > 2000:
            job["logs"] = job["logs"][-2000:]


def _set_job_state(job_id: str, **fields: Any) -> None:
    with _job_lock:
        job = _job_store.get(job_id)
        if not job:
            return
        job.update(fields)


def _build_commands(payload: PipelineRequest) -> list[list[str]]:
    commands: list[list[str]] = []
    if payload.pipeline == "data_creation":
        for script in DATA_CREATION_SCRIPTS:
            commands.append([sys.executable, str(PROJECT_ROOT / script)])
    elif payload.pipeline == "modeling":
        extra_args = ["--use-woe"] if payload.use_woe else []
        for script in MODELING_SCRIPTS:
            script_args = [] if script.endswith("compare_models.py") else extra_args
            commands.append([sys.executable, str(PROJECT_ROOT / script), *script_args])
    else:
        extra_args = ["--dry-run"] if payload.dry_run else []
        commands.append([sys.executable, str(PROJECT_ROOT / ARCHIVE_SCRIPT), *extra_args])
    return commands


def _run_job(job_id: str, payload: PipelineRequest) -> None:
    _set_job_state(job_id, status="running", started_at=_utc_now())
    _append_job_log(job_id, f"Starting '{payload.pipeline}' pipeline...")

    try:
        for command in _build_commands(payload):
            _append_job_log(job_id, "")
            _append_job_log(job_id, f"$ {subprocess.list2cmdline(command)}")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if process.stdout:
                for line in process.stdout:
                    _append_job_log(job_id, line.rstrip("\n"))

            return_code = process.wait()
            if return_code != 0:
                _set_job_state(
                    job_id,
                    status="failed",
                    completed_at=_utc_now(),
                    exit_code=return_code,
                )
                _append_job_log(job_id, f"Pipeline failed with exit code {return_code}.")
                return

        _set_job_state(job_id, status="completed", completed_at=_utc_now(), exit_code=0)
        _append_job_log(job_id, "Pipeline completed successfully.")
    except Exception as exc:  # pragma: no cover - defensive handling
        _set_job_state(job_id, status="failed", completed_at=_utc_now(), exit_code=-1)
        _append_job_log(job_id, f"Execution error: {exc}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    variables_cfg = _read_variables_json()
    global_cfg = _read_global_json()
    return {
        "global_parameters": global_cfg,
        "variables": variables_cfg.get("dataset_spec", {}).get("variables", []),
        "has_factory_defaults": VARIABLES_FACTORY_PATH.exists(),
    }


@app.post("/api/config/validate")
def validate_config(payload: ConfigPayload) -> dict[str, Any]:
    variables = [item.model_dump() for item in payload.variables]
    errors = _validate_all(variables=variables, global_cfg=payload.global_parameters)
    return {"valid": not errors, "errors": errors}


@app.put("/api/config")
def save_config(payload: ConfigPayload) -> dict[str, Any]:
    variables = [item.model_dump() for item in payload.variables]
    global_parameters = payload.global_parameters

    errors = _validate_all(variables=variables, global_cfg=global_parameters)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    variables_wrapper = _read_variables_json()
    variables_wrapper.setdefault("dataset_spec", {})
    variables_wrapper["dataset_spec"]["variables"] = variables

    _write_json_with_backup(VARIABLES_PATH, variables_wrapper)
    _write_json_with_backup(GLOBAL_PARAMETERS_PATH, global_parameters)

    return {"saved": True, "variables_path": str(VARIABLES_PATH), "global_path": str(GLOBAL_PARAMETERS_PATH)}


@app.post("/api/pipeline/jobs")
def run_pipeline(payload: PipelineRequest) -> dict[str, Any]:
    if payload.pipeline == "data_creation":
        variables_cfg = _read_variables_json()
        variables = variables_cfg.get("dataset_spec", {}).get("variables", [])
        if _count_included_variables(variables) == 0:
            raise HTTPException(
                status_code=400,
                detail="No variables are enabled for data creation. Tick at least one variable.",
            )

    job_id = uuid.uuid4().hex[:12]
    job_record: dict[str, Any] = {
        "id": job_id,
        "pipeline": payload.pipeline,
        "status": "queued",
        "created_at": _utc_now(),
        "started_at": None,
        "completed_at": None,
        "exit_code": None,
        "logs": [],
        "options": {
            "use_woe": payload.use_woe,
            "dry_run": payload.dry_run,
        },
    }

    with _job_lock:
        _job_store[job_id] = job_record

    thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/pipeline/jobs")
def list_jobs() -> list[dict[str, Any]]:
    with _job_lock:
        jobs = [deepcopy(record) for record in _job_store.values()]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return jobs


@app.get("/api/pipeline/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _job_lock:
        job = _job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return deepcopy(job)


@app.post("/api/config/variables/factory-reset")
def factory_reset_variables() -> dict[str, Any]:
    factory_payload = _read_factory_variables_json()
    _write_json_with_backup(VARIABLES_PATH, factory_payload)
    variables_cfg = _read_variables_json()
    return {
        "reset": True,
        "variables": variables_cfg.get("dataset_spec", {}).get("variables", []),
        "factory_path": str(VARIABLES_FACTORY_PATH),
    }
