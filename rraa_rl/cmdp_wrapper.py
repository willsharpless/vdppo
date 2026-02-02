from attrs import define, field, frozen
from valtr.reachability import DAGId
from valtr.valtr import to_dag, to_dag_notransform

from rraa_rl.src.env.general_task.env import Env, StaticTemporalNodeMixin


@frozen
class CMDPOperation:
    pass


@frozen
class CMDPAvoid(CMDPOperation):
    avoid: DAGId  # A(arg). Arg should be purely propositional.


@frozen
class CMDPReachChain(CMDPOperation):
    """Represents F(r_1 & F(r_2 & F(r_3 ...)))"""

    reach: DAGId  # Should be purely propositional.
    condition: list[DAGId] | None  # Conditions to be met along the way.


@frozen
class CMDPReachAvoid(CMDPOperation):
    """Represents q U r"""

    reach: DAGId  # Should be purely propositional.
    avoid: DAGId  # Should be purely propositional.


@frozen
class CMDPFG(CMDPOperation):
    """Represents F G r. Requires an epsilon move."""

    stay: DAGId  # Should be purely propositional.


@frozen
class CMDPGF(CMDPOperation):
    """Represents G F r."""

    reach: DAGId  # Should be purely propositional.


class CMDPEnvWrapper(Env):
    def __init__(self, env: StaticTemporalNodeMixin | Env):
        super().__init__(env.cfg, env.specification)
        self._env = env

        dag_builder, dag_root = to_dag_notransform(
            env.specification, ir_filename="dags/herd_os_ir", dag_filename="dags/cmdp"
        )
        self.dag_nodes_notrans = dag_builder.nodes
        self.dag_root_notrans = dag_root

    @property
    def specification(self):
        return self._env.specification

    @property
    def n_conjunctions(self) -> int:
        raise NotImplementedError("")
