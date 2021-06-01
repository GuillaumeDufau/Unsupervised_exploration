from abc import ABC, abstractmethod
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OffPolicyAlgorithm(ABC):
    def __init__(self, actor, critic, name='actor-critic'):
        self.actor = actor
        self.critic = critic
        self.name = name

    def soft_update_from_to(self, source, target, tau):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @abstractmethod
    def train_on_batch(self, batch):
        pass

    @abstractmethod
    def select_action(self, observation, deterministic=True):
        pass

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=device))



