import pathlib
import pickle
from typing import Annotated

import cyclopts
import ipdb
import jax
from cyclopts import Parameter
from loguru import logger

app = cyclopts.App()


@app.default
def main(run_paths: Annotated[list[pathlib.Path], Parameter(consume_multiple=True)], step: int | None = None):
    for run_path in run_paths:
        update_ckpt(run_path, step)


def update_ckpt(run_path: pathlib.Path, step: int | None = None):
    ckpts_path = run_path / "ckpts"
    if step is None:
        latest_ckpt = sorted(ckpts_path.glob("params_*.pkl"))
        assert latest_ckpt, f"No checkpoints found in {ckpts_path}"

        load_path = latest_ckpt[-1]
    else:
        load_path = ckpts_path / f"params_{step:09}.pkl"
        if not load_path.exists():
            available = sorted(ckpts_path.glob("params_*.pkl"))
            raise FileNotFoundError(f"Checkpoint not found: {load_path}. Available: {available}")
    logger.info(f"Restoring from {load_path}")

    with load_path.open("rb") as f:
        load_dict = pickle.load(f)

    # Convert to numpy, then save it
    load_dict_np = jax.device_get(load_dict)

    load_path_np = ckpts_path.with_name(load_path.stem + "_np.pkl")
    logger.info(f"Saving numpy checkpoint to {load_path_np}")
    with load_path_np.open("wb") as f:
        pickle.dump(load_dict_np, f)
    logger.success(f"Saved numpy checkpoint to {load_path_np}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
