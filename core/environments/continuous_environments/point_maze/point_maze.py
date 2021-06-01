from gym.spaces import Box
import numpy as np
import cv2
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt


from d4rl.pointmaze.maze_model import WALL
from core.environments.continuous_environments.point_maze.maze_utils import (
    MAZE,
    THREE_CORRIDORS_MAZE,
)
from core.environments.continuous_environments.point_maze.point_maze_inertia import (
    PointMazeWithInertiaEnv,
)


class PointMazeEnv(PointMazeWithInertiaEnv):
    """Mujoco simulation of a deceptive maze. No inertia, the point is "teleported" with increment dx, dy."""

    def __init__(
        self,
        maze_spec: MAZE = THREE_CORRIDORS_MAZE,
        reward_type: str = "dense",
        reset_target: bool = False,
        max_timesteps: int = 200,
    ):

        self.wall_locations = None
        PointMazeWithInertiaEnv.__init__(self, maze_spec, reward_type, reset_target)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.wall_locations = list(zip(*np.where(self.maze_arr == WALL)))
        self.current_step = 0
        self._max_episode_steps = max_timesteps

        # only check for collision in the neighboring points for each possible point location (time optimization)
        self.walls_to_check = {}
        for i in range(np.max(np.array(self.wall_locations)[:, 0])):
            for j in range(np.max(np.array(self.wall_locations)[:, 1])):
                self.walls_to_check[(i, j)] = [
                    wall
                    for wall in self.wall_locations
                    if np.abs(wall[0] - i) < 2 and np.abs(wall[1] - j) < 2
                ]

    def step(self, action: np.ndarray):
        """Gym step function. The action corresponds to (dx, dy) controls."""
        action = np.clip(action, -1.0, 1.0)
        new_position = self.point_position + 0.1 * action
        new_position = self._check_collision(new_position, last_action_taken=0.1 * action)
        self.set_state(new_position, np.array([0.0, 0.0]))
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

    def _lines_intersect(self, p1, p2, p3, p4):
        line1 = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        line2 = LineString([(p3[0], p3[1]), (p4[0], p4[1])])

        try:
            int_pt = line1.intersection(line2)
            point_of_intersection = [int_pt.x, int_pt.y]
            return point_of_intersection
        except:
            return False

    def _check_collision(self, position, last_action_taken):
        x, y = position
        old_position = np.array(position) - last_action_taken
        ball_radius = 0.1  ##### HARDCODED
        if self.wall_locations:
            x_old, y_old = old_position

            collisions = []
            # for each candidate piece of wall, check if there is collision
            for wall_location in self.walls_to_check[(int(x_old), int(y_old))]:

                x_wall, y_wall = wall_location
                x_wall, y_wall = float(x_wall), float(y_wall)

                # -0.75 / +0.25 to correspond to the visuals and to the maze with inertia
                x_wall_min, x_wall_max = x_wall - 0.75 - ball_radius, x_wall + 0.25 + ball_radius
                y_wall_min, y_wall_max = y_wall - 0.75 - ball_radius, y_wall + 0.25 + ball_radius

                tl, tr = [x_wall_min, y_wall_max], [x_wall_max, y_wall_max]
                bl, br = [x_wall_min, y_wall_min], [x_wall_max, y_wall_min]

                # is there a collision between the wall and the vector of movement
                collision_points = [
                    self._lines_intersect(np.array(position), old_position, tl, tr),
                    self._lines_intersect(np.array(position), old_position, bl, br),
                    self._lines_intersect(np.array(position), old_position, tl, bl),
                    self._lines_intersect(np.array(position), old_position, br, tr),
                ]

                for elem in collision_points:
                    if elem is not False:
                        collisions.append(elem)

            if len(collisions) > 0:
                # multiple collisions, find the closest one to the old point (first to hit)
                closest = min(collisions, key=lambda p: np.linalg.norm(p - old_position))
                return 0.999 * np.array(closest) + 0.001 * np.array(old_position)

        # no collision
        return np.array([x, y])

    def _get_obs(self):
        """Returns current observation. Observation = [point_position]."""
        return np.array(self.point_position.ravel())


# pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
if __name__ == "__main__":
    env = PointMazeEnv(maze_spec=THREE_CORRIDORS_MAZE)

    done = False
    steps = 0
    rewards = 0.0
    max_steps = 2000

    obs = env.reset()
    env.render()
    width, height = 400, 400
    print("test intersection", env._lines_intersect([1.0, 0.0], [2.0, 0.0], [1.0, -1], [2.0, 2.0]))
    list_observations = []
    while not done and steps < max_steps:
        ob, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        steps += 1
        rewards += reward
        list_observations.append(ob)

    list_observations = np.array(list_observations)
    plt.scatter(list_observations[:, 0], list_observations[:, 1], marker="o", c="b")

    plt.show()

    print("End of episode, steps = {}, rewards = {}".format(steps, rewards))
