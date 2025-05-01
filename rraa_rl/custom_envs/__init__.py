from dm_control import suite

# Custom environments
# from custom_envs import safePendulum
from rraa_rl.custom_envs.safeCartpole import safeCartpole

# Setup environments
# suite._DOMAINS["safePendulum"] = safePendulum
suite._DOMAINS["safeCartpole"] = safeCartpole