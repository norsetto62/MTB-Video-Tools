from autocut.cli import build_parser, main


def test_help_lists_commands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for command in ("run", "audit", "train", "validate", "test", "lovo", "models"):
        assert command in output


def test_bare_annotation_path_defaults_to_run():
    args = build_parser().parse_args(["run", "annotations.txt"])
    assert args.command == "run"
    assert str(args.annotations) == "annotations.txt"


def test_test_command_forwards_pytest_arguments(monkeypatch):
    from autocut import cli
    captured = {}

    class Result:
        returncode = 7

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert main(["test", "-q", "tests/test_config.py"]) == 7
    assert captured["command"][-2:] == ["-q", "tests/test_config.py"]
    assert captured["check"] is False
