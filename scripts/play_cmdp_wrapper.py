import ipdb
import numpy as np

from rraa_rl.cmdp_wrapper import CMDPEnvWrapper
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs


def main():
    env_name = "herdos"
    env, _, _ = get_env_and_cbs(env_name, agent_name="vd")
    env_cmdp = CMDPEnvWrapper(env)

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()