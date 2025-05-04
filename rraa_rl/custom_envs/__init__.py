from dm_control import suite

# Custom environments
# from custom_envs import safePendulum
from rraa_rl.custom_envs.safeCartpole import safeCartpole

# Setup environments
# suite._DOMAINS["safePendulum"] = safePendulum
suite._DOMAINS["safeCartpole"] = safeCartpole

########### New changes ############
from rraa_rl.custom_envs.safeCartpole_rraa import safeCartpole_rraa
suite._DOMAINS["safeCartpole_rraa"] = safeCartpole_rraa