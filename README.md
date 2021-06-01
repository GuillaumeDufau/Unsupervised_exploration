# Novelty search using model-based RL

The parallelization is handled with MPI.

Main idea from [Planning to Explore via Self-Supervised World Models (Sekar R. et al.)](https://arxiv.org/abs/2005.05960) with added features for experimentation:
i.e. exploration on chosen dimensions + Gaussian Processes in the model design for better uncertainty measurement.

## Code Abstract

Model-based reinforcement learning seeks to capture the dynamics of the environment. It potentially allows agents adaptation to different tasks and behaviors via planning. However, it often
falls short due to insufficient exploration and problematic sample efficiency. It is usually tackled
using retrospective reward tuning for the agent but such approaches often lack of control. To that
end, this code aims to analyse self-supervised reward shaping technique enabling adaptation to
new tasks unseen during exploration phase. Furthermore, the exploration can be
focused on specific dimensions of the states. The agent gets the incentive to act towards the
states with high uncertainty on the model predictions. Therefore, during exploration the agent
seeks states it has never seen before. Exploration rate of different techniques on challenging control tasks can be compared, ensemble of models vs Gaussian processes.
Focusing on specific states dimensions when possible
is faster than the whole in these cases. In particular, for the point maze and ant maze environment, focusing
the novelty search over the x/y positional information of the state is more efficient than the entire
state.

## Environments

PLease refer to the `.json` config files in `config/` folder to get default arguments for each tested env.

## Usage

Example of simple usage:

Will run 4 parallel workers of the same agent
```sh
mpiexec -n 4 python run.py --config unsup_explo_maze.json
```

## Configurations

Two types of models are used for the experiments. First, the bootstrap ensemble shown in arXiv:2005.05960 (Plan2explore).
Second, a gaussian process based approached mixing a Neural Network feature extractor + Gaussian process like output, arXiv:1611.00336 (SVDKL).

Explanation of `configs` files. Some Gaussian process (GP) parameters are different from ensemble.

### 1. model_config

| Parameters   |      Details      |  Example | 
|----------|:-------------:|------:|
| nn_num_features |  Number of output from Neural network in GP | [1; inf) |
| train_num_likelihood_samples |   sampled likelihoods from the GP to estimate the standard deviation |  [1; inf)  |
| test_num_likelihood_samples |   sampled likelihoods from the GP to estimate the standard deviation |  [1; inf)  |
| num_models |   number of Neural networks in the bootstrap ensemble method |  [2; inf)  |
| dimensions_of_interest |  Used for behavior descriptor approach where we want to focus the novelty search on specific dimensions of the state|  null or list of dimensions  |


### 2. General config parameters

| Parameters   |      Details      |  Example | 
|----------|:-------------:|------:|
| model_type | type of model, the custom reward corresponds to this model uncertainty | "ensemble" or "GP" |
| imagination_horizon | from plan2explore paper, on which horizon do we capture model's uncertainty |  [1; inf)  |
| num_sync_splits | how many times do we split the threads for population based: num_threads / num_sync_splits = num_agents |  \[1; num_workers\] |



