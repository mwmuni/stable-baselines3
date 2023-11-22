from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import (
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    SACPolicy,
)

SACSelf = TypeVar("SACSelf", bound="SAC")


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code
    from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup),
    from the softlearning repo (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used
        for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect
        transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps.
        Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after
        each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as
        steps done in the environment during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the
        different action noise type.
    :param replay_buffer_class: Replay buffer class
        to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass
        to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)
        Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically
        (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network
        every ``target_network_update_freq`` gradient steps.
    :param target_entropy: target entropy when
        learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically.
        (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages
        (such as device or wrappers used), 2 for debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network
        at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        betti_mod: bool = False,
        use_flagser: bool = False,
        use_ripser: bool = False,
        topology_config: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        late_dropout: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.betti_mod = betti_mod
        self.use_flagser = use_flagser
        self.use_ripser = use_ripser
        self.topology_config = topology_config
        self.dropout = dropout
        self.late_dropout = late_dropout

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        #! ------------------- Matt's code ------------------- !#
        if self.betti_mod:
            assert self.env is not None and self.actor is not None
            # self.delta_loss_history = th.ones((5), dtype=th.float32, device='cpu')
            # from .betti_boop_validation.betti_utils import get_flag_adj
            from models import RLAgent
            from stable_baselines3.common.policies import ContinuousCritic
            from typing import cast

            self.model_wrapped = RLAgent(self.env.action_space)
            self.model_wrapped.model = self.actor.latent_pi  # type: ignore
            if self.dropout > 0.0 and self.late_dropout is True:
                self.model_wrapped.model.append(th.nn.Dropout(self.dropout))
            self.critic = cast(ContinuousCritic, self.critic)
            self.critic1 = RLAgent(self.env.action_space)
            self.critic1.model = self.critic.qf0
            self.critic2 = RLAgent(self.env.action_space)
            self.critic2.model = self.critic.qf1

            self.adj_counter = 0
            if self.use_flagser:
                self.betti_avg = []
                self.betti_history = None
            if self.use_ripser:
                self.cocycle_tracker = dict()
                self.cocycle_lengths = []
                self.grad_step_cocycle_neurons = dict()
                self.cocycle_span_history = []
            self.total_gradient_steps = 0
            # Count how many neurons in self.model_wrapped.model
            layers = []
            for layer in self.model_wrapped.model:
                if type(layer) is th.nn.Linear:
                    if len(layers) == 0:
                        layers.append(layer.in_features)
                        layers.append(layer.out_features)
                    else:
                        layers.append(layer.out_features)
            self.num_neurons = sum(layers)
            # self.cocycle_lengths = th.zeros((0, self.num_neurons),
            # dtype=th.float32, device='cpu')
            # self.

        self.prev_rew = float("-inf")
        #! ----------------- End Matt's code ----------------- !#
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(
                np.float32
            )
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which
            #   is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        #! ------------------- Matt's code ------------------- !#
        # # if self.logger.total_timesteps_last <= 200_000:
        # if self.logger.ep_rew_mean_last != self.prev_rew:
        #     self.reward_nondecreasing = self.logger.ep_rew_mean_last > self.prev_rew
        #     self.prev_rew = self.logger.ep_rew_mean_last
        # # else:
        # #     self.reward_nondecreasing = False

        self.reward_nondecreasing = True
        #! ----------------- End Matt's code ----------------- !#

        for gradient_step in range(gradient_steps):
            # print(f"Gradient step {gradient_step}")
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # We need to sample because `log_std` may have
            #   changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.detach())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.detach())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            #! ------------------- Matt's code ------------------- !#
            experimental = True
            track_actor = False
            betti_loss = th.tensor(0.0, device=self.device, dtype=th.float32)
            if (
                not track_actor
                and self.betti_mod
                and self.reward_nondecreasing
                and not experimental
                and self.use_flagser
            ):
                from betti_utils import get_flag_adj
                from typing import cast
                from stable_baselines3.common.policies import ContinuousCritic

                self.critic = cast(ContinuousCritic, self.critic)
                obs_feat = self.critic.extract_features(replay_data.observations)
                _, flag_results1 = get_flag_adj(
                    self.critic1,
                    data=th.cat([obs_feat, replay_data.actions], dim=1).unsqueeze(1),
                    threshold=1.0,
                    max_dissimilarity=0.001,
                    max_dimension=0,
                    use_grad=False,
                    remove_output=False,
                    weighted_mode=False,
                )
                _, flag_results2 = get_flag_adj(
                    self.critic2,
                    data=th.cat([obs_feat, replay_data.actions], dim=1).unsqueeze(1),
                    threshold=1.0,
                    max_dissimilarity=0.001,
                    max_dimension=0,
                    use_grad=False,
                    remove_output=False,
                    weighted_mode=False,
                )
                betti_arrs1 = [f["betti"] for f in flag_results1]
                max_len = max([len(b) for b in betti_arrs1])
                for n in range(len(betti_arrs1)):
                    while len(betti_arrs1[n]) < max_len:
                        betti_arrs1[n].append(0)
                betti_arrs2 = [f["betti"] for f in flag_results2]
                max_len = max([len(b) for b in betti_arrs2])
                for n in range(len(betti_arrs2)):
                    while len(betti_arrs2[n]) < max_len:
                        betti_arrs2[n].append(0)
                betti_loss = th.tensor(betti_arrs1, dtype=th.float32).to(
                    self.device
                ) + th.tensor(betti_arrs2, dtype=th.float32).to(self.device)

            #! ----------------- End Matt's code ----------------- !#

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )  #! Original
            # critic_loss = 0.5 * sum((F.mse_loss(current_q, target_q_values)
            #                          for current_q in current_q_values)) + \
            #                              th.sum(0.5 * betti_loss) #! Matt mod
            critic_losses.append(critic_loss.detach())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ? START EXPERIMENTAL CODE

            # if self.betti_mod and experimental:
            #     import torch_tda
            #     import bats
            #     from typing import cast
            #     from stable_baselines3.common.policies import ContinuousCritic
            #     self.critic = cast(ContinuousCritic, self.critic)
            #     obs_feat = self.critic.extract_features(replay_data.observations)
            # _, adj1 = self.critic1.forward_flag_features(
            #     th.cat([obs_feat, replay_data.actions], dim=1).unsqueeze(1),
            #     threshold=1., max_dissimilarity=0.001, weighted_mode=False,
            #     keep_full_adj=True, keep_grad=True)
            #     # adj1 = adj1 + adj1.T
            # _, adj2 = self.critic2.forward_flag_features(
            #     th.cat([obs_feat, replay_data.actions], dim=1).unsqueeze(1),
            #     threshold=1., max_dissimilarity=0.001, weighted_mode=False,
            #     keep_full_adj=True, keep_grad=True)
            #     # adj2 = adj2 + adj2.T
            #     flags = (bats.standard_reduction_flag(),bats.compression_flag())
            #     layer = torch_tda.nn.RipsLayer(maxdim = 1, reduction_flags=flags)
            #     f1 = torch_tda.nn.BarcodePolyFeature(1,2,0, remove_zero=True)
            #     optimizer = th.optim.Adam([adj1, adj2], lr=1e-2)
            #     # for i in tqdm(range(100)):
            #     for _ in range(1):
            #         loss1 = 0
            #         loss2 = 0
            #         optimizer.zero_grad()
            #         for a1 in adj1:
            #             dgms1 = layer(a1)
            #             loss1 += -f1(dgms1)
            #         for a2 in adj2:
            #             dgms2 = layer(a2)
            #             loss2 += -f1(dgms2)
            #         loss = loss1 + loss2
            #         loss.backward()
            #         optimizer.step()

            # ? END EXPERIMENTAL CODE

            #! ------------------- Matt's code ------------------- !#
            # Compute the flag complex loss
            if self.betti_mod and track_actor and self.use_flagser:
                # if self.betti_mod and self.reward_nondecreasing and \
                #         track_actor and self.use_flagser:
                # betti_loss = th.tensor(0., dtype=th.float32, device=self.device)

                #!
                # self.logger.name_to_value['rollout/ep_rew_mean']

                from betti_utils import get_flag_adj
                from typing import cast
                from stable_baselines3.common.policies import ContinuousCritic

                _, flag_results = get_flag_adj(
                    self.model_wrapped,
                    data=replay_data.observations.unsqueeze(1),
                    threshold=1.0,
                    max_dissimilarity=0.001,
                    max_dimension=3,
                    use_grad=False,
                    remove_output=False,
                    weighted_mode=False,
                )
                # _, flag_results = get_flag_adj(
                #     self.model_wrapped, data=replay_data.observations.unsqueeze(1),
                #     threshold=1., max_dissimilarity=0.001, max_dimension=0,
                #     use_grad=False, remove_output=False, weighted_mode=False)
                # _, flag_results = get_flag_adj(
                #     self.model_wrapped, data=replay_data.observations.unsqueeze(1),
                #     threshold=1., max_dissimilarity=0.001, max_dimension=1,
                #     use_grad=False, remove_output=False, weighted_mode=False)
                betti_arrs = [f["betti"] for f in flag_results]
                max_len = max([len(b) for b in betti_arrs])
                for n in range(len(betti_arrs)):
                    while len(betti_arrs[n]) < max_len:
                        betti_arrs[n].append(0)

                # if self.betti_history is None:
                #     self.betti_history = np.array(betti_arrs).copy()
                # else:
                # self.betti_history = np.append(self.betti_history,
                #                                np.array(betti_arrs), axis=0)
                betti_arrs = th.tensor(betti_arrs, dtype=th.float32)
                # betti_0_1_sum = th.clamp(betti_arrs[:,0] + betti_arrs[:,1], min=1)
                # betti_loss = th.log(betti_arrs)
                # betti_loss = th.log(betti_0_1_sum)
                # betti_loss = betti_0_1_sum
                betti_loss = betti_arrs
                # betti_loss = betti_arrs[:,1]
                # betti_2_3_4_sum = th.clamp(th.sum(betti_arrs[:,2:], dim=1), min=1)
                # betti_loss = betti_0_1_sum / betti_2_3_4_sum

            if (
                experimental
                and self.betti_mod
                and (self.use_ripser or self.use_flagser)
            ):
                import ripser
                from betti_utils import get_flag_adj

                with th.no_grad():
                    # adj2, _ = get_adj(self.model_wrapped,
                    #   replay_data.observations.unsqueeze(1),
                    #   replace_nan=0, remove_layers=[0], remove_output=True)
                    # results = []
                    # self.cocycle_lengths = np.vstack([self.cocycle_lengths,
                    #                           np.zeros((1, self.num_neurons))])
                    # print(self.total_gradient_steps)
                    FLAGSER_MOD_VAL = self.topology_config["flagser frequency"]
                    RIPSER_MOD_VAL = self.topology_config["ripser frequency"]
                    THRESHOLD = self.topology_config["threshold"]
                    MAX_DISSIMILARITY = self.topology_config["max distance"]
                    if (
                        self.use_flagser
                        and self.total_gradient_steps % FLAGSER_MOD_VAL
                        == FLAGSER_MOD_VAL - 1
                    ):
                        print("Running Flagser...", end="")
                        FLAGSER_TARGET_DIM = 3
                        FLAGSER_SAMPLE_LIMIT = 10
                        _, flag_results = get_flag_adj(
                            self.model_wrapped,
                            data=replay_data.observations[
                                :FLAGSER_SAMPLE_LIMIT
                            ].unsqueeze(1),
                            threshold=THRESHOLD,
                            max_dissimilarity=MAX_DISSIMILARITY,
                            max_dimension=FLAGSER_TARGET_DIM,
                            use_grad=False,
                            remove_output=False,
                            weighted_mode=False,
                        )
                        # print('starting hell')
                        # result = ripser.ripser(adj2, maxdim=3, coeff=2,
                        #   do_cocycles=True, distance_matrix=True, n_perm=50)
                        # dgms, cocycles = result['dgms'], result['cocycles']
                        # if self.betti_history is None:
                        #     self.betti_history = []
                        # self.betti_history.append([len(cocycle)
                        #   for cocycle in cocycles])
                        # ==============================================
                        # ? Flagser method
                        betti_arrs = [f["betti"] for f in flag_results]
                        # max_len = max([len(b) for b in betti_arrs])
                        max_len = FLAGSER_TARGET_DIM + 1
                        for n in range(len(betti_arrs)):
                            while len(betti_arrs[n]) < max_len:
                                betti_arrs[n].append(0)

                        if self.betti_history is None:
                            self.betti_history = betti_arrs
                        elif len(betti_arrs) > 0:
                            if type(betti_arrs[0]) is list:
                                self.betti_history.extend(betti_arrs)
                            else:
                                self.betti_history.append([betti_arrs])
                            # self.betti_history = self.betti_history.extend(
                            # betti_arrs
                            # ) if type(betti_arrs[0]) is list else \
                            # self.betti_history.append([betti_arrs])
                        # print(cocycles)
                        print("done.")
                    if (
                        self.use_ripser
                        and self.total_gradient_steps % RIPSER_MOD_VAL
                        == RIPSER_MOD_VAL - 1
                    ):
                        print("Running Ripser...", end="")
                        RIPSER_TARGET_DIM = 1
                        RIPSER_SAMPLE_LIMIT = 1
                        # _, adj1 = self.critic1.forward_flag_features(
                        # th.cat([obs_feat, replay_data.actions],
                        # dim=1).unsqueeze(1), threshold=1.,
                        # max_dissimilarity=0.001, weighted_mode=False,
                        # keep_full_adj=True, keep_grad=True)
                        # _, adj1 = self.model_wrapped.forward_flag_features(
                        # replay_data.observations.unsqueeze(1), threshold=1.,
                        # max_dissimilarity=0.001, weighted_mode=False,
                        # keep_full_adj=True)
                        _, adj1 = self.model_wrapped.forward_flag_features(
                            replay_data.observations[:RIPSER_SAMPLE_LIMIT].unsqueeze(1),
                            threshold=THRESHOLD,
                            max_dissimilarity=MAX_DISSIMILARITY,
                            weighted_mode=False,
                            keep_full_adj=True,
                        )
                        self.adj_counter += 1
                        temp_cocycle_arr = np.zeros(self.num_neurons)
                        temp_cocycle_spans = {}
                        for a in adj1:
                            # Convert boolean torch adjacency matrix to numpy matrix
                            a = a.detach().cpu().numpy().astype(int)
                            # print('starting hell')
                            result = ripser.ripser(
                                a,
                                maxdim=RIPSER_TARGET_DIM,
                                coeff=2,
                                do_cocycles=True,
                                distance_matrix=True,
                            )  # , n_perm=100)
                            # print('finished hell')
                            _, cocycles = result["dgms"], result["cocycles"]
                            # unique_cocycle_indices = set()
                            # if self.betti_history is None:
                            #     self.betti_history = []
                            # self.betti_history.append([len(cocycle) for
                            #                            cocycle in cocycles])
                            len_cocycles = len(cocycles[RIPSER_TARGET_DIM])
                            if len_cocycles > 0:
                                unique_cocycles = [
                                    np.unique(c[:, :2].flatten())
                                    for c in cocycles[RIPSER_TARGET_DIM]
                                ]
                                # Check the min-max entries in each cocycle
                                cocycle_minmax = np.array(
                                    [
                                        [np.min(c)-1, np.max(c)-1]
                                        for c in unique_cocycles
                                    ]
                                )
                                # A layer has 128 neurons, so we need to
                                # take the floor of mixmax/128 to get the layer
                                # and then print the difference between the
                                # min and max to get the number of layers spanned
                                cocycle_span = np.abs(np.diff(
                                    np.floor(cocycle_minmax / 128)
                                    .astype(int),
                                    axis=1,
                                ))
                                # Print the count of each span size
                                print(f"There were {len_cocycles} cocycles found.")
                                unique_spans, span_counts = np.unique(cocycle_span, return_counts=True)
                                print(
                                    f"Cocycle span sizes: {', '.join([f'{s}: {c}' for s, c in zip(unique_spans, span_counts)])}"
                                )
                                if np.any((unique_spans < 0) | (unique_spans > 4)):
                                    print("ERROR: Span size out of range!")
                                # Add the cocycle spans to the dictionary
                                for pair in zip(unique_spans, span_counts):
                                    temp_cocycle_spans[pair[0]] = temp_cocycle_spans.get(pair[0], 0) + pair[1]
                                cocycle_lengths = np.array(
                                    [c.shape[0] for c in unique_cocycles]
                                )
                                neurons_used = np.unique(np.hstack(unique_cocycles))
                                self.grad_step_cocycle_neurons[
                                    self.total_gradient_steps
                                ] = neurons_used
                                for i, uc in enumerate(unique_cocycles):
                                    # Treat each element as a list of indices,
                                    # where the length of the indices is used as the
                                    # value of the indexed neuron in temp_cocycle_arr
                                    temp_cocycle_arr[uc] = np.maximum(
                                        temp_cocycle_arr[uc], cocycle_lengths[i]
                                    )
                            for i in range(len(cocycles[1])):
                                # unique_cocycle_indices = set()
                                for i1, i2, val in cocycles[1][i]:
                                    # unique_cocycle_indices.add(i1)
                                    # unique_cocycle_indices.add(i2)
                                    # self.grad_step_cocycle_neurons[
                                    #     self.total_gradient_steps] = \
                                    #         self.grad_step_cocycle_neurons.get(
                                    #             self.total_gradient_steps, set())
                                    # self.grad_step_cocycle_neurons[self.total_gradient_steps].add(i1)
                                    # temp_cocycle_arr[i1] = \
                                    #     max(temp_cocycle_arr[i1], len_cocycles)
                                    # self.cocycle_lengths[
                                    #     self.total_gradient_steps-1, i1] = \
                                    #         max(self.cocycle_lengths[
                                    #             self.total_gradient_steps-1, i1],
                                    #             len_cocycles)
                                    # self.grad_step_cocycle_neurons[
                                    #     self.total_gradient_steps] = \
                                    #         self.grad_step_cocycle_neurons.get(
                                    #             self.total_gradient_steps, set())
                                    # self.grad_step_cocycle_neurons[self.total_gradient_steps].add(i2)
                                    temp_cocycle_arr[i2] = max(
                                        temp_cocycle_arr[i2], len_cocycles
                                    )
                                    # self.cocycle_lengths[
                                    #     self.total_gradient_steps-1, i2] = \
                                    #         max(self.cocycle_lengths[
                                    #             self.total_gradient_steps-1, i2],
                                    #             len_cocycles)
                                # Iterate over set of unique indices
                                # if len(unique_cocycle_indices) > 0:
                                # str_indices = str(
                                #     sorted(list(unique_cocycle_indices)))
                                # self.cocycle_tracker[str_indices] = \
                                #     self.cocycle_tracker.get(str_indices, 0) + 1
                            # break
                        self.cocycle_lengths.append(temp_cocycle_arr)
                        self.cocycle_span_history.append(temp_cocycle_spans)
                        print("done.")

            if self.betti_mod and not (self.use_flagser or self.use_flagser) and \
                self.total_gradient_steps % self.topology_config["save frequency"] == self.topology_config["save frequency"] - 1:
                # Save the model's current state
                th.jit.save(th.jit.trace(self.model_wrapped.model, (replay_data.observations[:2].unsqueeze(1))), 
                            f"models/model_{self.topology_config['env name']}_{self.total_gradient_steps}.pt")

            # self.prev_rew = self.logger.ep_rew_mean_last
            # print(self.logger.ep_rew_mean_last)

            #! ----------------- End Matt's code ----------------- !#

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            # actor_loss = (ent_coef * log_prob - min_qf_pi + betti_loss*.01).mean()
            actor_loss = (ent_coef * log_prob - min_qf_pi - betti_loss).mean()
            actor_losses.append(actor_loss.detach())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            if self.betti_mod:
                self.total_gradient_steps += 1

        self._n_updates += gradient_steps

        # self.total_gradient_steps = 0

        if self.betti_mod and self.betti_history is not None:
            self.betti_avg.append(np.stack(self.betti_history).mean(axis=0))
            self.betti_history = None

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", sum(ent_coefs) / len(ent_coefs))
        self.logger.record("train/actor_loss", sum(actor_losses) / len(actor_losses))
        self.logger.record("train/critic_loss", sum(critic_losses) / len(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record(
                "train/ent_coef_loss", sum(ent_coef_losses) / len(ent_coef_losses)
            )

    def learn(
        self: SACSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> SACSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
