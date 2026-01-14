import cyclopts
import ipdb

from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer

app = cyclopts.App()


@app.default()
def main():
    env = HerdOs()
    trainer = Trainer()
    trainer.train(env)



if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
