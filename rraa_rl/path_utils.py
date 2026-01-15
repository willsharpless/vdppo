import pathlib


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_runs_dir() -> pathlib.Path:
    d = get_root_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
