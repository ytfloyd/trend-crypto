from scripts.research import strategy_registry_v0 as reg


def test_pythonpath_env_injected_for_python_steps(monkeypatch):
    calls = []

    def fake_run(cmd, shell, cwd, env):
        calls.append({"cmd": cmd, "env": env})
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(reg.subprocess, "run", fake_run)
    monkeypatch.setattr(reg, "REPO_ROOT", ".")
    monkeypatch.setattr(
        reg,
        "find_strategy",
        lambda _: {
            "id": "dummy",
            "run_recipe": [
                "python scripts/run_backtest.py --config foo.yaml",
                "echo done",
            ],
        },
    )

    args = type(
        "Args",
        (),
        {
            "id": "dummy",
            "no_tearsheet": False,
            "tearsheet": False,
            "tearsheet_only_top": None,
            "tearsheet_dir": None,
            "python": "/usr/bin/python",
            "no_pythonpath": False,
        },
    )()

    reg.cmd_run(args)
    assert len(calls) == 2
    assert calls[0]["env"] is not None
    assert calls[0]["env"]["PYTHONPATH"].startswith(f".{reg.os.pathsep}src")
    assert calls[1]["env"] is None


def test_run_plan_includes_tearsheet_by_default():
    recipe = [
        "echo prepare",
        "python scripts/research/demo_tearsheet.py --out_pdf artifacts/demo.pdf",
    ]
    plan = reg.build_run_plan(
        recipe,
        tearsheet_pdf="artifacts/demo.pdf",
        no_tearsheet=False,
        tearsheet_only_top=None,
        tearsheet_dir=None,
    )
    assert recipe[1] in plan


def test_run_plan_skips_tearsheet_when_disabled():
    recipe = [
        "echo prepare",
        "python scripts/research/demo_tearsheet.py --out_pdf artifacts/demo.pdf",
    ]
    plan = reg.build_run_plan(
        recipe,
        tearsheet_pdf="artifacts/demo.pdf",
        no_tearsheet=True,
        tearsheet_only_top=None,
        tearsheet_dir=None,
    )
    assert recipe[1] not in plan


def test_run_plan_tearsheet_only_top():
    recipe = [
        "echo prepare",
        "python scripts/research/demo_tearsheet.py --out_pdf artifacts/a.pdf",
        "python scripts/research/demo_tearsheet.py --out_pdf artifacts/b.pdf",
    ]
    plan = reg.build_run_plan(
        recipe,
        tearsheet_pdf=None,
        no_tearsheet=False,
        tearsheet_only_top=1,
        tearsheet_dir=None,
    )
    assert recipe[1] in plan
    assert recipe[2] not in plan
