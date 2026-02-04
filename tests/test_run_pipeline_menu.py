import run_pipeline


def test_run_pipeline_menu_loops_until_exit(monkeypatch):
    calls = []

    monkeypatch.setattr(run_pipeline, "run_data_creation", lambda: calls.append("data"))
    monkeypatch.setattr(run_pipeline, "run_modeling", lambda: calls.append("model"))
    monkeypatch.setattr(run_pipeline, "run_archive", lambda: calls.append("archive"))

    inputs = iter(["1", "2", "3", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_pipeline.main()

    assert calls == ["data", "model", "archive"]
