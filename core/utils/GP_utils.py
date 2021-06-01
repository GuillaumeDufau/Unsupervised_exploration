import numpy as np
from torch.optim import SGD

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from core.utils.mpi_utils import mpi_avg_grads
from core.utils.sac_utils import initialize_last_layer, initialize_hidden_layer


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a MultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(4), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class NeuralNet(nn.Module):
    def __init__(self, input_dim, features_dim, hidden_layers=[64, 64]):
        super(NeuralNet, self).__init__()

        # network architecture
        self.l1 = nn.Linear(input_dim, 64)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.l2 = nn.Linear(64, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.l3 = nn.Linear(64, features_dim)
        self.batch_norm3 = nn.BatchNorm1d(features_dim)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3)

    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class DKLModel(gpytorch.Module):
    """
    from GPytorch library examples 'https://gpytorch.ai/'

    :param grid_bounds: where to set the inducing points from SVDKL, i.e. range on which the function dynamics are estimated.
    :param feature_extractor: Neural Network used as feature extractor, trained alongside the GP
    """

    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10.0, 10.0)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        # trick to avoid error in the cholesky computation later on
        if features.shape[0] == 1 and features[0, 0] == 0.0:
            features[0, 0] += np.random.normal(0.0, 0.0001, 1)
        elif features.shape[0] == 1 and features[0, 1] == 0.0:
            features[0, 1] += np.random.normal(0.0, 0.0001, 1)
        features_scaled = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features_scaled.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res


class NeuralNetGaussianProcess(gpytorch.Module):
    """
    Neural net feature extractor into multitask variational GP . Does not contain the full
    SVDKL's GP addition in the last layer. reference paper:  	arXiv:1611.00336

    :param input_dim: size of the input space -> couple (action, state)
    :param output_dim: size of the output space -> state space size
    :param total_num_data: In total, throughout the entire training phase, how many data points will be used
    """

    def __init__(self, input_dim, output_dim, config, total_num_data):

        super(NeuralNetGaussianProcess, self).__init__()

        self.config = config

        self.core = DKLModel(
            NeuralNet(input_dim=input_dim, features_dim=config["nn_num_features"]), num_dim=config["nn_num_features"]
        )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)

        lr = config["model_lr"]
        self.optimizer = SGD(
            [
                {"params": self.core.feature_extractor.parameters(), "weight_decay": 1e-4},
                {"params": self.core.gp_layer.hyperparameters(), "lr": lr * 0.01},
                {"params": self.core.gp_layer.variational_parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0,
        )

        # compute the number of example the model will be trained on, pass this parameter in the mll
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.core.gp_layer, num_data=int(total_num_data))

    def __call__(self, x):
        return self.likelihood(self.core(x))

    def train_model(self, data, target):
        """
        For one query, the prediction is the average over the predicted likelihoods.
        :param data: X to be trained on, corresponds to (state, action) couples
        :param target: y labels for training, corresponds to next_state for each couple (state, action)
        :return loss: for visual figures
        """
        with gpytorch.settings.use_toeplitz(False):
            self.core.train()
            self.likelihood.train()

            with gpytorch.settings.num_likelihood_samples(self.config["train_num_likelihood_samples"]):
                self.optimizer.zero_grad()
                output = self.core(data)
                loss = -self.mll(output, target)
                loss.backward()
                mpi_avg_grads(self)
                self.optimizer.step()

        return float(loss)

    def test_model(self, data, target):
        """
        For one query, the uncertainty prediction corresponds to the standard deviation over the sampled likelihoods.
        :param data: X to be trained on, corresponds to (state, action) couples
        :param target: y labels for training, corresponds to next_state for each couple (state, action)
        """
        self.core.eval()
        self.likelihood.eval()

        correct = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.config["test_num_likelihood_samples"]):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            predictions = self.likelihood(self.core(data))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

    def get_predictions(self, input_sample, agent, imagination_horizon, dimensions_of_interest=None):
        """
        For one query (state, action), the uncertainty prediction corresponds to the
        standard deviation over the sampled likelihoods.

        Outputs mean (equivalent to standard prediction) and the custom reward given
        to the agent during self-exploration phase. Uses rollout of imagined states, outputs the uncertainty
        of the S(t+tau) state prediction as the reward.
        The noise input are not sampled from the fitted noise generative model. Rather hardcoded Gaussians 0, 1.

        :param input_sample: couples (state, action) passed as input to the model
        """
        # sample noise and replace states dimensions with noise
        if dimensions_of_interest:
            obs_len = agent.obs_len
            action_len = input_sample.shape[-1] - obs_len

            for i in range(obs_len):
                if i not in dimensions_of_interest:
                    input_sample[:, i] = torch.empty(input_sample.shape[0]).normal_(mean=0.0, std=1.0)

        self.core.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.config["test_num_likelihood_samples"]):
            if torch.cuda.is_available():
                input_sample = input_sample.cuda()
            output = self.likelihood(self.core(input_sample))

        # rollout of imagined states, outputs the uncertainty of the S(t+tau) state prediction
        # state t
        predicted_states = output.mean
        # get confidence region extremes, the uncertainty
        lower, upper = output.confidence_region()
        # sums the variance along each axis
        if input_sample.shape[0] == 1:
            reward = torch.abs(lower[0] - upper[0]).sum().detach().numpy()
        else:
            reward = torch.abs(lower - upper).sum(-1).detach().numpy()
        cumulative_horizon_reward = reward

        for i in range(imagination_horizon - 1):
            # action t
            actions = agent.select_action(torch.Tensor(predicted_states))
            # (st, at)
            predicted_states_actions = torch.cat((torch.Tensor(predicted_states), torch.Tensor(actions)), dim=-1)
            # get st+1
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.config["test_num_likelihood_samples"]):
                if torch.cuda.is_available():
                    input_sample = input_sample.cuda()
                output = self.likelihood(self.core(predicted_states_actions))
            predicted_states = output.mean
            # get the uncertainty
            lower, upper = output.confidence_region()
            # sums the variance along each axis
            if input_sample.shape[0] == 1:
                reward = torch.abs(lower[0] - upper[0]).sum().detach().numpy()
            else:
                reward = torch.abs(lower - upper).sum(-1).detach().numpy()
            # HARDCODED cumulative uncertainty obtained is weighted and the sum serves as total reward over imagination horizon
            cumulative_horizon_reward += 0.9 ** (i + 1) * reward

        return predicted_states, cumulative_horizon_reward
