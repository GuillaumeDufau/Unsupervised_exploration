import numpy as np

import torch
import torch.nn as nn
from core.utils.mpi_utils import mpi_avg_grads

from sklearn import mixture

# source
# https://github.com/mpritzkoleit/deep-ensembles/blob/master/Deep%20Ensembles.ipynb


class MLP(nn.Module):
    """Multilayer perceptron (MLP) with tanh/sigmoid activation
    functions implemented in PyTorch for regression tasks.

    Attributes:
        inputs (int): inputs of the network
        outputs (int): outputs of the network
        hidden_layers (list): layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
        activation (string): activation function used ('relu', 'tanh' or 'sigmoid')

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation="relu"):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.nLayers = len(hidden_layers)
        self.net_structure = [inputs, *hidden_layers, outputs]

        if activation == "relu":
            self.act = torch.relu
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        else:
            assert 'Use "relu","tanh" or "sigmoid" as activation.'
        # create linear layers y = Wx + b

        for i in range(self.nLayers + 1):
            setattr(
                self, "layer_" + str(i), nn.Linear(self.net_structure[i], self.net_structure[i + 1])
            )

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, "layer_" + str(i))
            x = self.act(layer(x))
        layer = getattr(self, "layer_" + str(self.nLayers))
        x = layer(x)
        return x


class GaussianMLP(MLP):
    """Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation="relu"):
        # when needs to output the variance of the each model, set outputs to 2*outputs
        super(GaussianMLP, self).__init__(
            inputs=inputs, outputs=outputs, hidden_layers=hidden_layers, activation=activation
        )
        self.inputs = inputs
        self.outputs = outputs
        self.batch_norm1 = nn.BatchNorm1d(128)

    def forward(self, x):
        # connect layers

        for i in range(self.nLayers):
            if i == 0:
                layer = getattr(self, "layer_" + str(i))
                x = self.act(self.batch_norm1(layer(x)))
            else:
                layer = getattr(self, "layer_" + str(i))
                x = self.act(layer(x))
        layer = getattr(self, "layer_" + str(self.nLayers))
        mean = layer(x)
        # mean, variance = torch.split(x, self.outputs, dim=1)
        # variance = F.softplus(variance) + 1e-6
        return mean  # , variance


def NLLloss(y, mean, var):
    """Negative log-likelihood loss function."""
    return (torch.log(var) + ((y - mean).pow(2)) / var).sum()


def MeanSquareError(y, mean):
    """Negative log-likelihood loss function."""
    return ((y - mean).pow(2)).sum()


class EnsembleModel(nn.Module):
    """Gaussian mixture MLP which outputs is the mean.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, input_dim, output_dim, config, hidden_layers=[128, 128], activation="relu"):
        super(EnsembleModel, self).__init__()
        self.config = config
        self.num_models = config["num_models"]
        self.inputs = input_dim
        self.outputs = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.models = []
        for i in range(self.num_models):
            model = GaussianMLP(
                inputs=self.inputs,
                outputs=self.outputs,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
            )
            setattr(self, "model_" + str(i), model)
            self.models.append(model)

        self.models_optimizers = []
        for i in range(self.num_models):
            model = getattr(self, "model_" + str(i))
            self.models_optimizers.append(
                torch.optim.Adam(
                    params=model.parameters(), lr=config["model_lr"], weight_decay=4e-5
                )
            )

        # mixture of Gaussians for noise sampling in case of behavior descriptor approach
        self.state_noise_density = mixture.BayesianGaussianMixture(
            n_components=self.outputs, tol=1e-1, max_iter=10
        )

    def fit_noise(self, X_train):
        """
        update the mixture of Gaussians for the states space approximation from the new visited states.
        Uses the EM algorithm to update.
        :param X_train: new states in np.array
        """
        self.state_noise_density.fit(X_train)

    def forward(self, x):
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, "model_" + str(i))
            mean = model(torch.tensor(x).float())
            means.append(mean)
            # variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        # variances = torch.stack(variances)
        # variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
        return mean, means  # mean, variance, means

    def train_one_model(self, model, optimizer, x, y):
        """Training an individual gaussian MLP of the deep ensemble."""
        optimizer.zero_grad()
        # mean, var = model(x)
        mean = model(x)
        loss = MeanSquareError(y, mean)
        loss.backward()
        mpi_avg_grads(model)
        optimizer.step()
        return loss.item()

    def train_model(self, data, target):
        """
        train all the models (the train_model stands for the entiere ensemble as the RL model).

        :param data: X to be trained on, corresponds to (state, action) couples
        :param target: y labels for training, corresponds to next_state for each couple (state, action)
        """
        losses = []
        for i in range(self.num_models):
            model = getattr(self, "model_" + str(i))
            loss = self.train_one_model(model, self.models_optimizers[i], data, target)
            losses.append(loss)

        return float(loss)

    def get_predictions(
        self, input_sample, agent, imagination_horizon=1, dimensions_of_interest=None
    ):
        """
        Output mean (equivalent to standard prediction) and the custom reward given
        to the agent during self-exploration phase.
        According to the paper 'Planning to explore via Self-supervised world models' (Sekar, 2020) we use the latent
        disagreement as the reward i.e. the variance over the predicted means from each model.
        If noise inputs are provided to replace dimensions of states (dimensions_of_interest is not None), the uncertainty
        is estimated multiple time and averaged over nb_noise_computation time.

        :param input_sample: couples of (state, action) in np.array format
        """
        # HARDCODED
        nb_noise_computations = 5 if dimensions_of_interest is not None else 1
        rewards_computed = []

        for h in range(nb_noise_computations):

            # replace states dimensions with noise sampled from noise generative model
            if dimensions_of_interest:
                obs_len = agent.obs_len

                sampled_noise, _ = self.state_noise_density.sample(input_sample.shape[0])
                input_sample = torch.cat(
                    (
                        input_sample[:, :2],
                        torch.FloatTensor(sampled_noise),
                        input_sample[:, obs_len:],
                    ),
                    dim=-1,
                )

            # step t
            predicted_states, all_models_means = self.forward(input_sample)
            for i in range(imagination_horizon - 1):
                # action t
                actions = agent.select_action(torch.Tensor(predicted_states))
                # (st, at)
                predicted_states_actions = torch.cat(
                    (torch.Tensor(predicted_states), torch.Tensor(actions)), dim=-1
                )
                # get st+1
                predicted_states, all_models_means = self.forward(predicted_states_actions)

            # get the variance among the K predicted states from the ensemble models
            disagreements = torch.var(all_models_means, dim=0)
            reward = disagreements.sum(-1).detach().numpy().astype(dtype="float32")

            # case where there is one single input, used for training rollouts
            if input_sample.shape[0] == 1:
                reward = reward[0]

            rewards_computed.append(reward)

        # # in the plan to explore the output is the a normal(mean, 1)

        return predicted_states, np.mean(np.array(rewards_computed), axis=0)
