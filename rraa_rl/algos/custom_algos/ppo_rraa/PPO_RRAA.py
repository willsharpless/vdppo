from stable_baselines3.ppo import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
import torch
import numpy as np

class PPO_RRAA(PPO):
    def __init__(self, *args, problem_type='RAA', decomposed_model_1_path=None, decomposed_model_2_path=None, **kwargs):
        self.problem_type = problem_type
        self.decomposed_model_1 = BaseAlgorithm.load(decomposed_model_1_path) \
            if problem_type in ['RAA', 'RR', 'RRAA'] else None
        self.decomposed_model_2 = BaseAlgorithm.load(decomposed_model_2_path) \
            if problem_type in ['RR', 'RRAA'] else None
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = RolloutBufferRRAA(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            problem_type=self.problem_type,
            decomposed_model=self.decomposed_model_1,
            decomposed_model=self.decomposed_model_2,
        )

class RolloutBufferRRAA(RolloutBuffer):
    def __init__(self, *args, problem_type='RAA', decomposed_model_1=None, decomposed_model_2=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the saved model
        self.problem_type = problem_type
        self.decomposed_model_1 = decomposed_model_1
        self.decomposed_model_2 = decomposed_model_2

    def compute_returns_and_advantage(self, last_values, dones):
        """
        A minimal alteration of RolloutBuffer.compute_returns_and_advantges that 
        incorporates the RRAA Bellman updates.

        This involves the use of pre-solved models corresponding to the decomposed
        values. 
        """

        # Compute Decomposed Values Vd(s)
        if self.problem_type != 'default':
            with torch.no_grad():
                obs_tensor = self.observations.to(self.device)

                ## TODO compute l(x), g(x) ie rewards, penalties

                if self.problem_type in ['RAA', 'RR', 'RRAA']:
                    decomposed_value_1 = self.decomposed_model_1.policy.predict_values(obs_tensor).clone().cpu().numpy().flatten()
                if self.problem_type in ['RR', 'RRAA']:
                    decomposed_value_2 = self.decomposed_model_2.policy.predict_values(obs_tensor).clone().cpu().numpy().flatten()
            
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        # Compute GAE
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            # Compute GA for Various Bellman Updates
            if self.problem_type == 'default':
                delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            elif self.problem_type == 'RA':
                # TODO
                pass

            # TODO other cases as well
            
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


    ## Default SB3 compute_returns_and_advantage function 

    # def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
    #     """
    #     Post-processing step: compute the lambda-return (TD(lambda) estimate)
    #     and GAE(lambda) advantage.

    #     Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    #     to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
    #     where R is the sum of discounted reward with value bootstrap
    #     (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

    #     The TD(lambda) estimator has also two special cases:
    #     - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
    #     - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

    #     For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

    #     :param last_values: state value estimation for the last step (one for each env)
    #     :param dones: if the last step was a terminal step (one bool for each env).
    #     """
    #     # Convert to numpy
    #     last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

    #     last_gae_lam = 0
    #     for step in reversed(range(self.buffer_size)):
    #         if step == self.buffer_size - 1:
    #             next_non_terminal = 1.0 - dones.astype(np.float32)
    #             next_values = last_values
    #         else:
    #             next_non_terminal = 1.0 - self.episode_starts[step + 1]
    #             next_values = self.values[step + 1]
    #         delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
    #         last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    #         self.advantages[step] = last_gae_lam
    #     # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
    #     # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
    #     self.returns = self.advantages + self.values