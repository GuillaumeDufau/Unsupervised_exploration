import d4rl
import gym

from gym.envs.registration import register
from d4rl.locomotion import ant
from d4rl.locomotion import maze_env

THREE_CORRIDORS_MAZE = (
    "##########\\"
    + "#OOOOO#OG##\\"
    + "#OOOOO#OO##\\"
    + "#OO#OO#OO##\\"
    + "#OO#OO#OO##\\"
    + "#OO#OOOOO##\\"
    + "#SO#OOOOO##\\"
    + "##########"
)


def parse_maze(maze_str):
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = [[None] * height for i in range(width)]
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                maze_arr[w][h] = 1
            elif tile == "G":
                maze_arr[w][h] = "g"
            elif tile == "S":
                maze_arr[w][h] = "r"
            elif tile == " " or tile == "O" or tile == "0":
                maze_arr[w][h] = 0
            else:
                raise ValueError("Unknown tile type: %s" % tile)
            # we transpose before return
    return list(map(list, zip(*maze_arr)))


def initiate_antMaze():
    maze_env.THREE_CORRIDORS_MAZE = parse_maze(THREE_CORRIDORS_MAZE)
    register(
        id="antmaze-zmaze-v0",
        entry_point="d4rl.locomotion.ant:make_ant_maze_env",
        max_episode_steps=700,
        kwargs={
            "maze_map": maze_env.THREE_CORRIDORS_MAZE,
            "reward_type": "sparse",
            "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5",
            "non_zero_reset": False,
            "eval": True,
            "maze_size_scaling": 4.0,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0,
        },
    )
