from valtr.valtr import to_dag, to_dag_notransform

from rraa_rl.src.env.general_task.env import StaticTemporalNodeMixin, Env


class CMDPEnvWrapper(Env):
    def __init__(self, env: StaticTemporalNodeMixin | Env):
        super().__init__(env.cfg, env.specification)
        self._env = env

        dag_builder, dag_root = to_dag_notransform(env.specification, ir_filename="dags/herd_os_ir", dag_filename="dags/cmdp")
        self.dag_nodes_notrans = dag_builder.nodes
        self.dag_root_notrans = dag_root

    @property
    def specification(self):
        return self._env.specification