import torch.nn.functional as F
import numpy as np
import os
from typing import List, Set, Dict, Tuple, Optional, Callable, Any
from gym.spaces import Box

from core.agents.sac import SAC
from core.utils.mpi_utils import mpi_avg_grads, MPI, sync_params, create_comm_correspondences
from core.utils.ensemble_utils import EnsembleModel
from core.utils.GP_utils import NeuralNetGaussianProcess
import torch

device = "cpu"

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


class sacModelBased(SAC):
    def __init__(self, obs_space: Box, action_space: Box, config: dict):

        super(sacModelBased, self).__init__(obs_space, action_space, config)

        self.obs_len = obs_space.shape[0]

        # parameters for the model
        self.model = None
        self.model_losses = []

        # parameters for the model's copy, useful to refresh the custom rewards
        self.model_copy = None

    def initiate_model(
        self, inputs: np.ndarray, outputs: np.ndarray, config: dict, comm: Optional[Any] = None
    ):
        """
        The Gaussian Process needs to be created along some data
        Initiate model is called along with the first training step and creates + fit the model
        For consistency, ensemble and GP are initiated with initiate_model.

        :param inputs: X to be trained on, corresponds to (state, action) couples
        :param outputs: y labels for training, corresponds to next_state for each couple (state, action)
        :param config: dict from the config files
        :param comm: ID of the current thread
        """

        if config["model_type"] == "GP":
            # Gaussian process needs to be initialized knowing the max amount of data it will encounter
            total_num_data = (
                (config["max_timesteps"] - config["batch_size"]) // config["train_freq"]
            ) * config["batch_size"]
            self.model = NeuralNetGaussianProcess(
                inputs,
                outputs,
                config["model_config"],
                total_num_data,
            )
            # creates a copy useful to compute the new reward in train_on_batch()
            model_statedict = self.model.state_dict()
            self.model_copy = NeuralNetGaussianProcess(
                inputs,
                outputs,
                config["model_config"],
                total_num_data,
            )
            self.model_copy.load_state_dict(model_statedict)
            sync_params(self.model)

        elif config["model_type"] == "ensemble":
            self.model = EnsembleModel(inputs, outputs, config["model_config"])
            self.model_copy = EnsembleModel(inputs, outputs, config["model_config"])
            for model in self.model.models:
                sync_params(model)

    def select_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if observation.ndim == 1:
            # single prediction useful during rollout phases
            observation = torch.FloatTensor(observation.reshape(1, -1)).to(device)
            return (
                self.actor(observation, deterministic=deterministic)[0].cpu().data.numpy().flatten()
            )
        else:
            # batch size predictions
            observation = torch.FloatTensor(observation).to(device)
            return self.actor(observation, deterministic=deterministic)[0].cpu().data.numpy()

    def save_model(self, path: str, steps: int):
        """
        saves the model's weights + value and policy networks from self.save() parent function
        :param path: where to store it
        :param steps: when was it saved
        """
        # saves
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        torch.save(self.model.state_dict(), path + "model_steps_{}.pth".format(steps))
        self.save(f"steps_{steps}", path)

    def update_model_copy_weights(self, model_type: str):
        """
        actualizes the model's copy weights
        :param model_type: "GP" or "ensemble"
        """

        if model_type == "GP":
            model_statedict = self.model.state_dict()
            self.model_copy.load_state_dict(model_statedict)
        elif model_type == "ensemble":
            # ensemble case
            for model, model_copy in zip(self.model.models, self.model_copy.models):
                model_statedict = model.state_dict()
                model_copy.load_state_dict(model_statedict)

    def compute_rewards(
        self,
        observations: torch.tensor,
        actions: torch.tensor,
        reward_function: Optional[Callable[[np.ndarray], List[np.ndarray]]] = None,
        imagination_horizon: int = 1,
        dimensions_of_interest: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute each sample's reward depending on the training phase we're in.
        If unsupervised, reward_function is None. Therefore the rewards are the model's uncertainties
        If zero/few shots, the reward_function can be the one provided by the environment.
        It corresponds to the objective task

        :param reward_function: function to be used for the rewards modification R
        :param dimensions_of_interest: in unsupervised phase,
                dimensions where no noise is inputed to get the model's uncertainty.
                if dimensions_of_interest is None, no noise but the entire sample s_t serves as input
        """

        # data is couple (state, action) which is the input for the model
        data = torch.cat((torch.Tensor(observations), torch.Tensor(actions)), dim=1)
        if not reward_function:
            _, rewards = self.model_copy.get_predictions(
                data,
                self,
                imagination_horizon=imagination_horizon,
                dimensions_of_interest=dimensions_of_interest,
            )
            rewards = torch.FloatTensor(rewards).view(-1, 1).to(device)
        else:
            rewards = []
            # Few shot using the provided reward_function
            for state, action in zip(observations, actions):
                reward = reward_function(np.array(action), test_state=np.array(state))
                rewards.append(reward)
            rewards = torch.FloatTensor(rewards).view(-1, 1).to(device)

        return rewards, data

    def train_on_batch(
        self,
        batch: Dict[str, np.ndarray],
        imagination_horizon: int = 1,
        reward_function: Optional[Callable[[np.ndarray], List[np.ndarray]]] = None,
        comm: Optional[Any] = None,
    ):
        """
        Trains the SAC networks + the model.
        Each model type, GP or ensemble has its own train() built-in function / method
        Recomputes the rewards before each update using the uncertainty of the model as reward signal
        """

        # increment iteration counter
        self._n_train_steps_total += 1

        # read batch
        rewards = torch.FloatTensor(batch["rewards"]).view(-1, 1).to(device)
        done = torch.FloatTensor(1.0 - batch["terminals"]).view(-1, 1).to(device)
        observations = torch.FloatTensor(batch["observations"]).to(device)
        actions = torch.FloatTensor(batch["actions"]).to(device)
        new_observations = torch.FloatTensor(batch["new_observations"]).to(device)

        # recompute the batch rewards
        dims = self.model.config.get("dimensions_of_interest", None)
        rewards, data = self.compute_rewards(
            observations,
            actions,
            reward_function=reward_function,
            imagination_horizon=imagination_horizon,
            dimensions_of_interest=dims,
        )
        # train the model
        self.model_losses.append(self.model.train_model(data, target=new_observations))

        # compute actor and alpha losses
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.actor(
            observations, return_log_prob=True
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            # sync_tensor_grads(self.log_alpha)
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1

        Q1_new_actions, Q2_new_actions = self.critic_target(observations, new_obs_actions)
        Q_new_actions = torch.min(Q1_new_actions, Q2_new_actions)
        actor_loss = (alpha * log_pi - Q_new_actions).mean()

        # compute critic losses
        current_Q1, current_Q2 = self.critic(observations, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.actor(new_observations, return_log_prob=True)
        target_Q1, target_Q2 = self.critic_target(new_observations, new_next_actions)
        target_Q_values = torch.min(target_Q1, target_Q2) - alpha * new_log_pi

        target_Q = self.reward_scale * rewards + (done * self.discount * target_Q_values)
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(
            current_Q2, target_Q.detach()
        )

        # optimization
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        mpi_avg_grads(self.actor)
        self.actor_optimizer.step()

        # soft updates
        if self._n_train_steps_total % self.target_update_period == 0:
            self.soft_update_from_to(self.critic, self.critic_target, self.soft_target_tau)
