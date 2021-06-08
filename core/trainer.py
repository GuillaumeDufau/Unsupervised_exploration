import os
import time
import gym
import numpy as np
import torch
from mpi4py import MPI
from typing import Optional

from core.utils.point_maze import PointMaze

# from core.utils.ant_maze import initiate_antMaze
from core.utils.mpi_utils import (
    sync_params,
    create_comm_correspondences,
    setup_pytorch_for_mpi,
)
from core.utils.replay_buffer import ReplayBuffer
from core.agents.sac_modelbased import sacModelBased
from core.utils.visualization_utils import (
    get_video,
    plot_traces,
    plot_loss_evolution,
    make_video_from_images,
    get_visuals,
    update_percent_explored,
)

# from core.environments.continuous_environments.point_maze.point_maze import PointMazeEnv
# from core.environments.continuous_environments.point_maze.point_maze_inertia import PointMazeWithInertiaEnv
# from core.environments.continuous_environments.point_maze.maze_utils import MAZE, THREE_CORRIDORS_MAZE

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


# initiate_antMaze()


class Trainer:
    """Models and agent trainer.

    This is the component where the models and agent are trained according
    to their training frequencies. Tests are also ran in this class.
    It also handles the environments steps and visual and weights saves.
    """

    def __init__(self, config: dict):
        """Initializes the learner.
        Main args (in config):
          num_workers: number of processes for parallel runs.
          env_config: all settings specific to chosen environment.
          agent_name: agent's algorithm, containing its model (e.g. SAC agent + specific model)
          agent_config: contains information for the agent as well as the model
            (e.g. ensemble of models or Gaussian process based)

          config also contains standard ML training parameters (learning rates etc)
        """

        # mpi parralelization setup
        setup_pytorch_for_mpi()

        # different save dir every run
        self.saves_dir = os.path.join(
            "run_data", time.strftime("%d-%b-%Y_%H.%M.%S", time.localtime())
        )

        self.config = config
        # number of parallel processes
        self.config["num_workers"] = num_workers

        # Init the env according to config
        if self.config["env_name"] == "PointMaze-v0":
            self.env = PointMaze(**config["env_config"])
        elif self.config["env_name"] == "PointMazeEnv":
            self.env = PointMazeEnv(
                maze_spec=THREE_CORRIDORS_MAZE,
                max_timesteps=config["env_config"]["max_steps"],
            )
        elif self.config["env_name"] == "PointMaze_inertia":
            self.env = PointMazeWithInertiaEnv(
                maze_spec=THREE_CORRIDORS_MAZE,
                max_timesteps=config["env_config"]["max_steps"],
            )
        else:
            self.env = gym.make(self.config["env_name"])

        self.replay_buffer = ReplayBuffer(max_size=self.config["buffer_max_size"])

        # Get observation space and action space
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        self.num_actions, self.obs_len = action_space.shape[0], obs_space.shape[0]

        # Set seeds
        self.seed = self.config["seed"] + rank + 2
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # visuals settings
        self.global_points_location = []
        # create the grid of booleans to compute the percent explored
        self.states_explored = np.full((40, 40), False, dtype=bool)
        self.list_percent_explored = []
        self.list_of_rewards = []
        self.env_config = config["env_config"]

        # Initialize policy
        if self.config["agent_name"] == "sacMB":
            self.agent = sacModelBased(obs_space, action_space, config=self.config["agent_config"])
        else:
            raise NotImplementedError('agent "{}" is not defined'.format(self.config["agent_name"]))

        # for population based, splitted_comm is each thread agent's ID
        # for each thread, splitted_com is the ID of the corresponding agent
        # one agent be trained on multiple threads
        self.splitted_com = create_comm_correspondences(
            rank, self.config["num_workers"], num_splits=self.config["num_sync_splits"]
        )

        # Sync the actor and critic between all threads
        for network in self.agent.networks_to_sync:
            sync_params(network)

        # Metrics
        global_eval_score = self.evaluate(self.env)
        if rank == 0:
            self.metrics = {
                "global_ep_rewards": [],
                "global_evalutations": [global_eval_score],
            }

    def evaluate(
        self,
        env,
        num_eval_episodes: int = 1,
        visual: bool = False,
        traces: bool = False,
        timestep: Optional[int] = None,
    ) -> float:
        """Evaluate the state of learning"""

        avg_reward = 0.0
        for i in range(num_eval_episodes):
            actions_taken = []
            observation = self.env.reset()
            done = False
            while not done:
                action = self.agent.select_action(np.array(observation))
                # action = self.env.action_space.sample()
                actions_taken.append(action)
                observation, reward, done, _ = self.env.step(action)
                avg_reward += reward

        avg_reward /= num_eval_episodes
        print("rank {} evaluation reward {}".format(rank, avg_reward))
        global_avg_reward = np.mean(comm.allgather(avg_reward))

        # Additional visual effects specific to testing phases
        if visual and rank == 0:
            get_video(env, actions_taken, avg_reward, self.config["max_timesteps"])
        if traces and rank == 0:
            if self.config["env_name"] == "PointMaze-v0":
                plot_traces(env, actions_taken, num_workers=i, maze_env=True, timestep=timestep)
        if rank == 0:
            print(
                "Evaluation with {} workers over {} total episodes: {}".format(
                    num_workers, num_eval_episodes * num_workers, global_avg_reward
                )
            )
        return global_avg_reward

    def train(self):
        """Trains the model(s) as well as the agent. Saves visuals, weights on external files.
        Adds data to the replay buffer and runs steps in the environment

        One env step possible operations (when corresponds to config frequency):
          - reset env
          - update model and agent (initiate model if needed)
          - run evaluation steps on main process, saves the results in visuals etc
          - select action
          - run one env step
          - if final step: saves specific final visuals and information
        """

        torch.set_num_threads(1)
        if rank == 0:
            file_name = "%s_%s_%s" % (
                self.config["agent_name"],
                self.config["env_name"],
                str(self.seed),
            )
            print("---------------------------------------")
            print("Settings: %s" % (file_name))
            print("---------------------------------------")

        last_noise_fit = 0
        global_timesteps = 0
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < self.config["max_timesteps"]:
            if done:
                # Reset environment
                observation = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Update
            if (
                total_timesteps % self.config["train_freq"] == 0
                and total_timesteps > self.config["batch_size"]
            ):
                # create the model or synchronize the real model and its copy for rewards renewal
                if not self.agent.model:
                    self.agent.initiate_model(
                        self.obs_len + self.num_actions, self.obs_len, self.config
                    )
                else:
                    # update model_copy's weights
                    self.agent.update_model_copy_weights(self.config["model_type"])

                # update model's noise generative model
                # (to encourage exploration on specific dimensions of interest)
                if self.config["model_config"].get("dimensions_of_interest") is not None:
                    new_obs = np.vstack(
                        np.array(self.replay_buffer.storage[last_noise_fit:total_timesteps])[:, 0]
                    )
                    self.agent.model.fit_noise(new_obs[:, 2:])
                    self.agent.model_copy.fit_noise(new_obs[:, 2:])
                    last_noise_fit = total_timesteps

                # run the training epochs
                for _ in range(self.config["train_freq"]):
                    batch = self.replay_buffer.sample(batch_size=self.config["batch_size"])

                    # changes the training type according to the phase we're in
                    # (novelty search vs few/zero shot)
                    if (
                        total_timesteps
                        <= self.config["max_timesteps"] - self.config["nb_supervised_steps"]
                    ):
                        # unsupervised phase, the reward is still predicted
                        # from the model uncertainty
                        self.agent.train_on_batch(
                            batch,
                            imagination_horizon=self.config["imagination_horizon"],
                            comm=self.splitted_com,
                        )
                    else:
                        # supervised phase, zero or few shots, the reward_function gives
                        # the estimated reward
                        self.agent.train_on_batch(
                            batch,
                            imagination_horizon=self.config["imagination_horizon"],
                            reward_function=self.env.step,
                            comm=self.splitted_com,
                        )

            # Evaluation
            if total_timesteps % self.config["eval_freq"] == 0 and total_timesteps > 0:
                print(
                    "timesteps",
                    round(total_timesteps / self.config["max_timesteps"], 2),
                )
                global_eval_score = self.evaluate(
                    self.env,
                    num_eval_episodes=2,
                    traces=False,
                    timestep=total_timesteps,
                )
                global_timesteps = sum(comm.allgather(total_timesteps))
                # make sure the env is in reset state to avoid bugs, better solution is an
                # env copy for experiments
                observation = self.env.reset()

                if rank == 0:
                    self.metrics["global_evalutations"].append(global_eval_score)
                    if self.config["save_models"]:
                        self.agent.save("checkpoint_" + str(global_timesteps), self.saves_dir)
                    self.list_of_rewards.append(global_eval_score)

                # create visuals from the experiment
                # gather the points from every worker
                all_points_location = None
                if rank == 0:
                    all_points_location = np.empty(
                        [num_workers, len(self.global_points_location), 2],
                        dtype="float",
                    )
                comm.Gather(np.array(self.global_points_location), all_points_location, root=0)
                if self.config["visuals"] and rank == 0 and self.agent.model:
                    all_points_location = all_points_location.reshape(
                        (num_workers * len(self.global_points_location), 2)
                    )
                    get_visuals(
                        list_of_points=np.array(all_points_location),
                        model=self.agent.model,
                        agent=self.agent,
                        imagination_horizon=self.config["imagination_horizon"],
                        list_of_rewards=self.list_of_rewards,
                        list_of_percents=self.list_percent_explored,
                        save_path="visuals/",
                        timestep=global_timesteps,
                        config=self.config,
                        plot_limits=[
                            self.env_config["x_min"],
                            self.env_config["x_max"],
                            self.env_config["y_min"],
                            self.env_config["y_max"],
                        ],
                    )

                # saves model weights
                if self.config["save_model"] and rank == 0 and self.agent.model is not None:
                    self.agent.save_model(
                        path="saves/seed_{}/".format(self.seed),
                        steps=num_workers * len(self.global_points_location),
                    )

            # Select action randomly or according to policy
            if total_timesteps < self.config["start_timesteps"]:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(np.array(observation), deterministic=False)

            # Perform action
            new_observation, reward, done, _ = self.env.step(action)

            # update information for visuals
            if self.config["visuals"]:
                # record the x/y coordinates from new states
                self.global_points_location.append(new_observation[:2])
                # gather the visited states
                new_visited = None
                if rank == 0:
                    new_visited = np.empty([num_workers, self.obs_len], dtype="float")
                comm.Gather(np.array(new_observation), new_visited, root=0)
                # update the visuals lists for maze coverage in percent
                if rank == 0:
                    (self.states_explored, self.list_percent_explored,) = update_percent_explored(
                        new_visited,
                        self.env_config,
                        self.states_explored,
                        self.list_percent_explored,
                        self.config["model_config"].get(
                            "dimensions_of_interest", list(range(self.obs_len))
                        ),
                    )

            # update episode stats
            episode_reward += reward
            done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            # Store data in replay buffer
            self.replay_buffer.add((observation, new_observation, action, reward, done_bool))
            observation = new_observation

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        global_eval_score = self.evaluate(self.env, num_eval_episodes=2, visual=False, traces=False)
        all_points_location = None
        if rank == 0:
            all_points_location = np.empty(
                [num_workers, len(self.global_points_location), self.obs_len],
                dtype="float",
            )
        comm.Gather(np.array(self.global_points_location), all_points_location, root=0)

        # final visual and checkpoints saves (done on the main process)
        if rank == 0:
            self.metrics["global_evalutations"].append(global_eval_score)
            if self.config["save_models"]:
                self.agent.save("checkpoint_" + str(global_timesteps), self.saves_dir)

            plot_loss_evolution(self.agent.model_losses)
            make_video_from_images(
                save_path="visuals/agentpaths_model_evolution",
                images_path="visuals/",
                fps=1,
            )
