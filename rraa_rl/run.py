import datetime
import pathlib
import random

import nltk
from attrs import define
from wordfreq import top_n_list

from rraa_rl.path_utils import get_runs_dir


@define
class Run:
    start_time: datetime.datetime

    noun: str
    name: str
    folder_suffix: str
    env_name: str
    run_dir: pathlib.Path

    @property
    def wandb_name(self):
        stamp = self.start_time.strftime("%y%m%d-%H%M%S")
        return f"{stamp}_{self.folder_suffix}"

    @staticmethod
    def create(env_name: str, name: str | None = None):
        now = datetime.datetime.now()
        runs_dir = get_runs_dir()

        noun = get_random_noun(n_letters=5)
        if name is None:
            folder_suffix = noun
        else:
            folder_suffix = f"{noun}_{name}"

        stamp = now.strftime("%Y%m%d-%H%M%S")
        folder_name = f"{stamp}_{folder_suffix}"

        run_dir = runs_dir / env_name / folder_name
        return Run(
            start_time=now,
            noun=noun,
            name=name,
            folder_suffix=folder_suffix,
            env_name=env_name,
            run_dir=run_dir,
        )

    @property
    def plots_dir(self) -> pathlib.Path:
        d = self.run_dir / "plots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def ckpts_dir(self) -> pathlib.Path:
        d = self.run_dir / "ckpts"
        d.mkdir(parents=True, exist_ok=True)
        return d


def get_random_noun(n_letters: int = 5) -> str:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt", quiet=True)

    common_words = top_n_list("en", 5000)
    words = [w for w in common_words if len(w) == n_letters]
    tagged = nltk.pos_tag(words)
    nouns = [w for w, tag in tagged if tag.startswith("NN")]
    return random.choice(nouns)
