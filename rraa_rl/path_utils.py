import pathlib


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_runs_dir(debug: bool) -> pathlib.Path:
    if debug:
        d = get_root_dir() / "runs_debug"
    else:
        d = get_root_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_paper_plot_dir() -> pathlib.Path:
    d = get_root_dir() / "paper_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d
