import ipdb
import numpy as np

from rraa_rl.control.cmdp_wrapper import CMDPEnvWrapper
from rraa_rl.env.general_task.get_env import get_env_and_cbs


def main():
    env_name = "herding"
    env, _, _ = get_env_and_cbs(env_name, agent_name="vdppo")
    cfg = CMDPEnvWrapper.Cfg()
    env_cmdp = CMDPEnvWrapper(cfg, env)

    for op in env_cmdp.cmdp_info.operations:
        print(f"Operation: {op}")

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
