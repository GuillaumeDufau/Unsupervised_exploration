import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
import torch


def get_video(env, actions_taken, reward, timesteps, width=400, height=400):
    """
    Record and save one episode of the agent
    :param actions_taken: list of action taken during episode
    :param reward: reward obtained
    :param timesteps: training timestep when get_video is called
    :param width: video width
    :param height: video height
    """
    images = []
    print("we're in the get video")
    _ = env.reset()
    for idx, action in enumerate(actions_taken):
        if idx % 1 == 0 and type(action) != int:
            img = env.render(mode="rgb_array", width=width, height=height)
            img = np.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0]], axis=-1)
            images.append(img)
            env.step(action)

    video_name = "timesteps_{}_len_{}_r{}.mp4".format(timesteps, len(actions_taken), reward)
    height, width, layers = images[0].shape
    # change format
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 20, (width, height))  # add fourcc
    for image in images:
        video.write(image)

    video.release()


def plot_traces(env, actions_taken, num_workers, maze_env=False, timestep=None):
    """
    plot agent traces for one episode.
    TODO: function has been modified, doesn't work for different skills anymore
    """
    if timestep:
        filename = "{}_workers_{}_timesteps_traces.png".format(num_workers, timestep)
    else:
        filename = "{}_workers_traces.png".format(num_workers)

    plt.figure(figsize=(6, 6))
    palette = sns.color_palette("hls", 1)
    for skill in range(1):
        for rank_id in range(1):
            _ = env.reset()
            if maze_env == True:
                temp_x, temp_y = env.state
            else:
                temp_x, temp_y, _ = env.env.get_body_com("torso")
            x = [temp_x]
            y = [temp_y]
            # 1 trace for the selected skill
            for idx, action in enumerate(actions_taken):
                if type(action) != int:
                    (obs, _, _, _) = env.step(action)
                    if maze_env == True:
                        temp_x, temp_y = env.state
                    else:
                        temp_x, temp_y, _ = env.env.get_body_com("torso")

                    x.append(temp_x)
                    y.append(temp_y)
            if rank_id == 0:
                plt.plot(np.array(x), np.array(y), c=palette[skill], marker="o")
            else:
                plt.plot(np.array(x), np.array(y), c=palette[skill])

    plt.legend(framealpha=1, frameon=True)
    plt.savefig(filename)
    plt.close()


def get_heat_map(list_of_points, save_path="", timestep=None, nb_bins=50, plot_limits=None):
    """
    get the 2D heatmap from the list_of_points
    :param timestep: indication for save/title
    :param nb_bins: quality of the heatmap -> to high isn't readable
    :param plot_limits: plot's range
    :return: saves the figure
    """
    list_of_points = np.array(list_of_points)
    xs = list_of_points[:, 0]
    ys = list_of_points[:, 1]

    if plot_limits:
        xmin, xmax, ymin, ymax = plot_limits
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, bins=nb_bins, range=[[xmin, xmax], [ymin, ymax]]
        )
    else:
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=nb_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin="lower")
    plt.title("time steps {}".format(timestep))
    plt.savefig(save_path, dpi=400)
    plt.close()


def get_uncertainty_map(
    model, agent, imagination_horizon, save_path="", timestep=None, plot_limits=None, nb_bins=50
):
    """
    TODO: does not work properly, bug from the zs (uncertainty values)
    """
    xs = np.linspace(-10, 10, nb_bins)
    ys = np.linspace(-10, 10, nb_bins)
    xs, ys = np.meshgrid(xs, ys)
    input_samples = np.stack((xs, ys), axis=-1).reshape(nb_bins ** 2, 2)
    actions = (
        agent.actor(torch.from_numpy(input_samples).float(), deterministic=False)[0]
        .cpu()
        .data.numpy()
    )

    input_samples = torch.cat((torch.Tensor(input_samples), torch.Tensor(actions)), dim=1)
    _, zs = model.get_predictions(input_samples, agent, imagination_horizon=imagination_horizon)
    zs = zs.reshape(nb_bins, nb_bins)

    if plot_limits:
        xmin, xmax, ymin, ymax = plot_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(xs, ys, np.array(zs), cmap="RdBu", vmin=np.min(zs), vmax=np.max(zs))
    plt.title("time steps {}".format(timestep))
    # set the limits of the plot to the limits of the data
    fig.colorbar(c, ax=ax)
    plt.savefig(save_path, dpi=400)
    plt.close()


def get_scatter_map(list_of_points, save_path="", timestep=None, nb_bins=50, plot_limits=None):
    """
    get the scatter plot from list_of_points
    :param timestep: indication for save/title
    :param nb_bins: quality of the heatmap -> to high isn't readable
    :param plot_limits: plot's range
    :return: saves the figure
    """
    list_of_points = np.array(list_of_points)
    xs = list_of_points[:, 0]
    ys = list_of_points[:, 1]

    plt.scatter(xs, ys, s=1)

    if plot_limits:
        xmin, xmax, ymin, ymax = plot_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    plt.title("time steps {}".format(timestep))
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_loss_evolution(list_of_losses, save_path=""):
    """
    get the 2D heatmap from the list_of_points
    :param timestep: indication for save/title
    :param nb_bins: quality of the heatmap -> to high isn't readable
    :param plot_limits: plot's range
    :return: saves the figure
    """
    list_of_losses = np.array(list_of_losses)

    fig, ax = plt.subplots()
    ax.plot(np.array([x for x in range(len(list_of_losses))]), list_of_losses)

    ax.set(xlabel="nb of training steps", ylabel="Model loss", title="Loss evolution of the model")
    ax.grid()

    fig.savefig("model_losses.png")


def make_video_from_images(save_path="video", images_path="", fps=1):
    """
    from the saved figures during training create a video showing the evolution. format MF4
    :param save_path: do not need to precise .mp4
    :param fps: number of frame per seconds
    :return: saves it directly in save_path
    """
    image_folder = images_path
    video_name_path = save_path + ".mp4"
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video = cv2.VideoWriter(video_name_path, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()


def update_percent_explored(new_obs, env_config, states_explored, list_percent_explored, x_y_dims):
    x_axis, y_axis = tuple(x_y_dims)
    for i in range(new_obs.shape[0]):
        x_ = int(
            (new_obs[i, x_axis] - env_config["x_min"] - 0.001)
            * 40
            / (env_config["x_max"] - env_config["x_min"])
        )
        y_ = int(
            (new_obs[i, y_axis] - env_config["y_min"] - 0.001)
            * 40
            / (env_config["y_max"] - env_config["y_min"])
        )
    if not states_explored[x_, y_]:
        states_explored[x_, y_] = True
        nb_discretized_states = states_explored.sum()
        nb_total_disccrete_states = len(states_explored) ** 2
        new_percent = (nb_discretized_states / nb_total_disccrete_states) * 100.0
        list_percent_explored.append(new_percent)
    else:
        list_percent_explored.append(list_percent_explored[-1])
    return states_explored, list_percent_explored


def get_visuals(
    list_of_points,
    model,
    agent,
    imagination_horizon,
    config,
    list_of_rewards,
    list_of_percents,
    save_path="",
    timestep=None,
    plot_limits=None,
    nb_bins=50,
):
    """
    Creates and saves 4 figures.
    1) top-right: get the heatmap of model's uncertainty (constructed from a grid of points)
    2) top-left: creates scatter map of all x/y states coordinates visited by the agent
    3) bottom-left: plot the obtained env-based returns obtained during the evaluate() calls
    4) bottom-right: percentage of the maze explored, computed from the x/y coordinates of the visited states
    nb_bins: parameter for heatmap granularity
    """
    # create folder
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass

    # settings
    # trick to name the images in the creation order for the video creation
    steps = str(timestep).zfill(len(str(int(config["num_workers"] * config["max_timesteps"]))))
    save_path += "{}_step.png".format(steps)

    # settings
    env_config = config["env_config"]
    x_axis, y_axis = tuple(config["model_config"].get("dimensions_of_interest", list(range(2))))
    nb_few_shots, nb_steps_total = config["nb_supervised_steps"], config["max_timesteps"]
    unsupervised_steps = nb_steps_total - nb_few_shots
    nb_threads = int(config["num_workers"])

    # set the figure format
    fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    fig.suptitle("time steps {}".format(timestep))

    ## 1) Plot the heatmap of the model's uncertainty
    # create grid
    xs = np.linspace(env_config["x_min"], env_config["x_max"], nb_bins)
    ys = np.linspace(env_config["y_min"], env_config["y_max"], nb_bins)
    xs, ys = np.meshgrid(xs, ys)

    # get uncertainty values on the grid
    obs_len = agent.obs_len
    input_states = np.array(torch.empty((nb_bins, nb_bins, obs_len)).normal_(mean=0.0, std=1.0))
    input_states[:, :, x_axis] = xs
    input_states[:, :, y_axis] = ys
    input_states = input_states.reshape(nb_bins ** 2, obs_len)

    actions = (
        agent.actor(torch.from_numpy(input_states).float(), deterministic=False)[0]
        .cpu()
        .data.numpy()
    )

    input_samples = torch.cat((torch.Tensor(input_states), torch.Tensor(actions)), dim=1)
    _, zs = model.get_predictions(input_samples, agent, imagination_horizon=imagination_horizon)
    zs = zs.reshape(nb_bins, nb_bins)
    if plot_limits:
        xmin, xmax, ymin, ymax = plot_limits

    # plot the heatmap
    c = ax1.pcolormesh(
        xs, ys, np.array(zs), cmap="RdBu", vmin=np.min(zs), vmax=np.max(zs)
    )  # min(np.max(zs), 10.))
    ax1.set_title("Model uncertainty")
    fig.colorbar(c, ax=ax1)

    ## 2) Plot the visited points in scatter format
    list_of_points = np.array(list_of_points)
    xs = list_of_points[:, x_axis]
    ys = list_of_points[:, y_axis]
    # add maze walls
    x_0A, y_0A = env_config["x_min"], env_config["upper_wall_height_offset"]
    if config["env_name"] == "PointMaze_inertia" or config["env_name"] == "PointMazeEnv":
        x_1A, y_1A = 4.00, env_config["upper_wall_height_offset"]
        x_1B, y_1B = 2.50, env_config["lower_wall_height_offset"]
    else:
        x_1A, y_1A = (env_config["x_max"] - env_config["x_min"]) * 0.75 + env_config[
            "x_min"
        ], env_config["upper_wall_height_offset"]
        x_1B, y_1B = (
            -(env_config["x_max"] - env_config["x_min"]) * 0.75 + env_config["x_max"],
            env_config["lower_wall_height_offset"],
        )
    x_0B, y_0B = env_config["x_max"], env_config["lower_wall_height_offset"]

    # plot
    ax2.scatter(xs, ys, s=1)
    ax2.plot([x_0A, x_1A], [y_0A, y_1A], c="k", linewidth=20.0)
    ax2.plot([x_0B, x_1B], [y_0B, y_1B], c="k", linewidth=20.0)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_title("Points visited by the agent")

    ## 3) Plot the returns evolution
    list_of_rewards = np.array(list_of_rewards)
    # change axis to get global timesteps
    x_axis = np.array(
        list(
            range(
                0,
                len(list_of_rewards) * nb_threads * int(config["eval_freq"]),
                nb_threads * int(config["eval_freq"]),
            )
        )
    )
    ax3.plot(x_axis, list_of_rewards)
    # ax4.set_xticklabels(list(range(0, int(nb_steps_total* nb_threads), int(config["eval_freq"]))))
    ax3.set(xlabel="Environment steps", ylabel="Average reward", title="Evaluation rewards")
    ax3.set_xlim(0, nb_steps_total * nb_threads)
    ax3.axvspan(
        unsupervised_steps * nb_threads, nb_steps_total * nb_threads, facecolor="g", alpha=0.1
    )
    ax3.grid()

    ## 4) Plot the exploration percentage evolution from the agent
    list_of_percents = np.array(list_of_percents)
    # change axis to get global timesteps
    x_axis = np.array(list(range(0, len(list_of_percents) * nb_threads, nb_threads)))
    ax4.plot(x_axis, list_of_percents)
    ax4.set(xlabel="Environment steps", ylabel="Maze coverage (%)", title="Maze coverage evolution")
    ax4.grid()
    ax4.set_ylim(0.0, 100.0)
    ax4.set_xlim(0, nb_steps_total * nb_threads)
    ax4.axvspan(
        unsupervised_steps * nb_threads, nb_steps_total * nb_threads, facecolor="g", alpha=0.1
    )

    # Save the 4 plots
    fig.savefig(save_path, dpi=300)
    plt.close()
