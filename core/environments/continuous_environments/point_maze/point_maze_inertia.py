from d4rl.pointmaze.maze_model import GOAL, WALL
from gym.envs.mujoco import mujoco_env
from gym import utils
from core.environments.continuous_environments.point_maze.maze_utils import MAZE, THREE_CORRIDORS_MAZE, START
from core.environments.continuous_environments.point_maze.maze_utils import parse_maze, point_maze

import numpy as np


class PointMazeWithInertiaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Mujoco simulation of a deceptive maze."""

    def __init__(
        self,
        maze_spec: MAZE = THREE_CORRIDORS_MAZE,
        reward_type: str = "dense",
        reset_target: bool = False,
        max_timesteps: int = 250,
    ):

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == START)))
        self.reset_locations.sort()
        self._target = np.array([0.0, 0.0])
        self.current_step = 0
        self._max_episode_steps = max_timesteps

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        self.wall_locations = list(zip(*np.where(self.maze_arr == WALL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))

    def step(self, action: np.ndarray):
        """Gym step function. The action corresponds to (dx, dy) controls."""
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == "dense":
            reward = -np.linalg.norm(self.point_position - self._target)
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        done = False
        self.current_step += 1
        if self.current_step == self._max_episode_steps:
            done = True
            self.current_step = 0
        return ob, reward, done, {}

    def _get_obs(self):
        """Returns current observation. Observation = [point_position, point_velocity]."""
        return np.concatenate([self.point_position, self.point_velocity]).ravel()

    @property
    def point_position(self) -> np.ndarray:
        """Return the (x,y) position of the point."""
        return self.sim.data.qpos

    @property
    def point_velocity(self) -> np.ndarray:
        """Return the (v_x, v_y) velocity of the point."""
        return self.sim.data.qvel

    def get_target(self) -> np.ndarray:
        """Returns the target destination."""
        return self._target

    def set_target(self, target_location: np.ndarray = None):
        """Set target position."""
        if target_location is None:
            idx = self.np_random.choice(len(self.goal_locations))
            reset_location = np.array(self.goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location  # + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        """Add visual sign at target position in the simulation."""
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def clip_velocity(self):
        """Clip the point velocity."""
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        """Reset the simulation."""
        idx = self.np_random.choice(len(self.reset_locations))
        reset_location = np.array(self.reset_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location: np.ndarray):
        """Reset the simulation and initialize the point position at the given position."""
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass


# pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
if __name__ == "__main__":
    env = PointMazeWithInertiaEnv(maze_spec=THREE_CORRIDORS_MAZE)

    done = False
    steps = 0
    rewards = 0.0
    max_steps = 250

    obs = env.reset()
    env.render()
    print("observation space", env.observation_space, "action space", env.action_space)

    while not done and steps < max_steps:
        _, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        steps += 1
        rewards += reward

    print("End of episode, steps = {}, rewards = {}".format(steps, rewards))
