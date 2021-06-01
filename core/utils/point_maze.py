import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class PointMaze(gym.Env):
    def __init__(
        self,
        scale_action_space=10,
        decay_reward=False,
        dense_reward=True,
        x_min=-1,
        x_max=1,
        y_min=-1,
        y_max=1,
        zone_width=0.1,
        max_steps=100,
        zone_width_offset_from_x_min=0.5,
        zone_height_offset_from_y_max=-0.2,
        wall_width_ratio=0.75,
        upper_wall_height_offset=1,
        lower_wall_height_offset=-1,
        start_x=0.0,
        start_y=0.0,
    ):

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.scale_action_space = scale_action_space
        self.start_x = start_x
        self.start_y = start_y

        self.low = np.array([self.x_min, self.y_min], dtype=np.float32)
        self.high = np.array([self.x_max, self.y_max], dtype=np.float32)

        # Action: dx, dy

        self.action_space = spaces.Box(self.low / scale_action_space, self.high / scale_action_space, dtype=np.float32)
        # Observation: x, y
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

        self.n_zones = 1

        self.zone_width = zone_width

        self.zone_width_offset = self.x_min + zone_width_offset_from_x_min
        self.zone_height_offset = self.y_max + zone_height_offset_from_y_max

        self.viewer = None

        self._max_episode_steps = max_steps

        self.dense_reward = dense_reward

        # step count for an episode useful for decaying reward
        self.decay_reward = decay_reward
        self.step_count = 0

        # Walls
        self.wallheight = 0.01
        self.wallwidth = (self.x_max - self.x_min) * wall_width_ratio

        self.upper_wall_width_offset = self.x_min + self.wallwidth / 2
        self.upper_wall_height_offset = upper_wall_height_offset

        self.lower_wall_width_offset = self.x_max - self.wallwidth / 2
        self.lower_wall_height_offset = lower_wall_height_offset

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _in_zone(self, x_pos, y_pos):

        zone_center_width, zone_center_height = self.zone_width_offset, self.zone_height_offset

        if zone_center_width - self.zone_width / 2 <= x_pos <= zone_center_width + self.zone_width / 2:
            if zone_center_height - self.zone_width / 2 <= y_pos <= zone_center_height + self.zone_width / 2:
                return True

        return False

    def _sparse_reward(self, in_zone):
        if in_zone:

            if self.decay_reward:
                reward = 1 - 0.9 * (self.step_count / self._max_episode_steps)
            else:
                reward = 1
        else:
            reward = 0

        return reward

    def _collision_lower_wall(self, y_pos, y_pos_old, x_pos, x_pos_old):
        # From down
        if y_pos_old <= self.lower_wall_height_offset < y_pos:
            x_hitting_wall = (self.lower_wall_height_offset - y_pos_old) / (y_pos - y_pos_old) * (
                x_pos - x_pos_old
            ) + x_pos_old
            if x_hitting_wall >= self.x_max - self.wallwidth:
                y_pos = self.lower_wall_height_offset
                # x_pos = x_hitting_wall

        # From up
        if y_pos < self.lower_wall_height_offset + self.wallheight <= y_pos_old < self.upper_wall_height_offset:
            x_hitting_wall = (self.lower_wall_height_offset - y_pos_old) / (y_pos - y_pos_old) * (
                x_pos - x_pos_old
            ) + x_pos_old
            if x_hitting_wall >= self.x_max - self.wallwidth:
                y_pos = self.lower_wall_height_offset + self.wallheight
                # x_pos = x_hitting_wall

        return y_pos

    def _collision_upper_wall(self, y_pos, y_pos_old, x_pos, x_pos_old):
        # From up
        if y_pos_old >= self.upper_wall_height_offset + self.wallheight > y_pos:
            x_hitting_wall = (self.upper_wall_height_offset - y_pos_old) / (y_pos - y_pos_old) * (
                x_pos - x_pos_old
            ) + x_pos_old
            if x_hitting_wall <= self.x_min + self.wallwidth:
                y_pos = self.upper_wall_height_offset + self.wallheight
                # x_pos = x_hitting_wall

        # From down
        if y_pos > self.upper_wall_height_offset >= y_pos_old > self.lower_wall_height_offset:
            x_hitting_wall = (self.upper_wall_height_offset - y_pos_old) / (y_pos - y_pos_old) * (
                x_pos - x_pos_old
            ) + x_pos_old
            if x_hitting_wall <= self.x_min + self.wallwidth:
                y_pos = self.upper_wall_height_offset

        return y_pos

    def step(self, action, test_state=None):

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if test_state is not None:
            x_pos_old, y_pos_old = test_state
        else:
            self.step_count += 1
            x_pos_old, y_pos_old = self.state

        x_pos = x_pos_old + action[0]
        y_pos = y_pos_old + action[1]

        y_pos = self._collision_lower_wall(y_pos, y_pos_old, x_pos, x_pos_old)
        y_pos = self._collision_upper_wall(y_pos, y_pos_old, x_pos, x_pos_old)

        x_pos = np.clip(x_pos, self.x_min, self.x_max)
        y_pos = np.clip(y_pos, self.y_min, self.y_max)

        # If on the zone
        in_zone = self._in_zone(x_pos, y_pos)
        done = in_zone
        if self.dense_reward:
            reward = -np.linalg.norm(np.array([x_pos - self.zone_width_offset, y_pos - self.zone_height_offset]))
        else:
            reward = self._sparse_reward(in_zone)

        if self.step_count >= self._max_episode_steps and test_state is None:
            done = True

        if test_state is None:
            self.state = x_pos, y_pos

            return np.array(self.state), reward, done, {"in_zone": in_zone, "ind_zone": [0], "pos": [x_pos, y_pos]}
        else:
            return reward

    def reset(self):
        # self.state = np.array([-0.6, self.np_random.uniform(low=self.y_min, high=self.y_max)])
        x_start = self.start_x + self.np_random.uniform(low=self.x_min, high=self.x_max) / self.scale_action_space
        y_start = self.start_y + self.np_random.uniform(low=self.y_min, high=self.y_max) / self.scale_action_space
        self.state = np.array([x_start, y_start])

        self.step_count = 0
        return np.array(self.state)

    def render(self, mode="human", width=2, height=3):

        screen_width = 600
        screen_height = 600

        scale_width = screen_width / (self.x_max - self.x_min)
        scale_height = screen_height / (self.y_max - self.y_min)

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 2 walls
            l, r, t, b = -self.wallwidth / 2, self.wallwidth / 2, self.wallheight / 2, -self.wallheight / 2

            upper_wall = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            upper_trans_width = (self.upper_wall_width_offset - self.x_min) / (self.x_max - self.x_min) * screen_width
            upper_trans_height = (
                (self.upper_wall_height_offset - self.y_min) / (self.y_max - self.y_min) * screen_height
            )

            self.upper_walltrans = rendering.Transform(
                translation=(upper_trans_width, upper_trans_height), scale=(scale_width, scale_height)
            )

            upper_wall.add_attr(self.upper_walltrans)
            self.viewer.add_geom(upper_wall)

            lower_wall = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            lower_trans_width = (self.lower_wall_width_offset - self.x_min) / (self.x_max - self.x_min) * screen_width
            lower_trans_height = (
                (self.lower_wall_height_offset - self.y_min) / (self.y_max - self.y_min) * screen_height
            )

            self.lower_walltrans = rendering.Transform(
                translation=(lower_trans_width, lower_trans_height), scale=(scale_width, scale_height)
            )

            lower_wall.add_attr(self.lower_walltrans)
            self.viewer.add_geom(lower_wall)

            # Zone to reach
            l, r, t, b = -self.zone_width / 2, self.zone_width / 2, self.zone_width / 2, -self.zone_width / 2

            zone = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)], filled=False)

            translation_width = (self.zone_width_offset - self.x_min) / (self.x_max - self.x_min) * screen_width
            translation_height = (self.zone_height_offset - self.y_min) / (self.y_max - self.y_min) * screen_height

            zone_trans = rendering.Transform(
                translation=(translation_width, translation_height), scale=(scale_width, scale_height)
            )
            zone.add_attr(zone_trans)
            self.viewer.add_geom(zone)

            self.circle = rendering.make_circle(5)
            self.circletrans = rendering.Transform()
            self.circle.add_attr(self.circletrans)
            self.viewer.add_geom(self.circle)

        if self.state is None:
            return None

        x, y = self.state
        x_render = x * scale_width + screen_width / 2.0
        y_render = y * scale_height + screen_height / 2.0
        self.circletrans.set_translation(x_render, y_render)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


gym.envs.registration.register(id="PointMaze-v0", entry_point="envs.point_maze:PointMaze")
