
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units

        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]


    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class GaussianPolicyNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body):
        super(GaussianPolicyNet, self).__init__()

        self.actor_body = actor_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs):
        '''
        TODO:
            In the tasks with continuous action space, we model the policy at each state with a Gaussian Distribution.
            Your task is to fill out the forward pass of this policy network.
            (You may still change the variables in the __init__ if you want to!)
            Hint:
                1. You might want to use self.actor_body, self.fc_action and tanh activation function to predict mean
                2. You might want to use self.std and softplus activation to predict std
                3. You may find torch.distributions.Normal useful!
                4. You may want to take a sample form the predicted distribution as a predicted action
                5. You may want to compute log_probability of taking that action with the predicted policy
                6. You may want to compute entropy of your distribution to regularize your policy update
        '''
        ##############################################################
        ############### YOUR CODE HERE - 6-8 lines ###################
        import torch
        from torch.distributions import Normal
        features = self.actor_body(obs)
        mean = torch.tanh(self.fc_action(features))
        std = F.softplus(self.std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        ##############################################################
        ######################## END YOUR CODE #######################
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean}

class CategoricalPolicyNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body):
        super(CategoricalPolicyNet, self).__init__()
        self.actor_body = actor_body
        self.fc_action = nn.Linear(actor_body.feature_dim, action_dim)
        self.to(Config.DEVICE)

    def forward(self, obs):
        '''
        TODO:
            In the tasks with discrete action space, we model the policy at each state with a Categorical Distribution.
            Your task is to fill out the forward pass of this policy network.
            (You may still change the variables in the __init__ if you want to!)
            Hint:
                1. You might want to use self.actor_body and self.fc_action to predict parameters of your distribution
                3. You may find torch.distributions.Categorical useful!
                4. You may want to take a sample form the predicted distribution as a predicted action
                5. You may want to compute log_probability of taking that action with the predicted policy
                6. You may want to compute entropy of your distribution to regularize your policy update

        '''
        ##############################################################
        ############### YOUR CODE HERE - 6-8 lines ###################
        import torch
        from torch.distributions import Categorical
        features = self.actor_body(obs)
        logits = self.fc_action(features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        ##############################################################
        ######################## END YOUR CODE #######################
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy}

class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body,
                 critic_body):
        super(GaussianActorCriticNet, self).__init__()

        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())

        self.to(Config.DEVICE)

    def forward(self, obs):
        '''
            TODO:
                In the tasks with continuous action space, we model the policy at each state with a Gaussian Distribution.
                Your task is to fill out the forward pass of this policy network and value function network.
                (You may still change the variables in the __init__ if you want to!)
                Hint:
                    1. You might find hints from GaussianPolicyNet.forward useful!
                    2. You might want to use self.critic_body and self.fc_critic to predict the value of the input state
        '''
        ##############################################################
        ############### YOUR CODE HERE - 8-10 lines ##################
        import torch
        from torch.distributions import Normal
        actor_features = self.actor_body(obs)
        mean = torch.tanh(self.fc_action(actor_features))
        std = F.softplus(self.std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        critic_features = self.critic_body(obs)
        v = self.fc_critic(critic_features)
        ##############################################################
        ######################## END YOUR CODE #######################
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean,
                'v': v}

class CategoricalActorCriticNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body,
                 critic_body):
        super(CategoricalActorCriticNet, self).__init__()

        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs):
        '''
        TODO:
            In the tasks with discrete action space, we model the policy at each state with a Categorical Distribution.
            Your task is to fill out the forward pass of this policy network and value function network.
            (You may still change the variables in the __init__ if you want to!)
            Hint:
                1. You might find hints from CategoricalPolicyNet.forward useful!
                2. You might want to use self.critic_body and self.fc_critic to predict the value of the input state
        '''
        ##############################################################
        ############### YOUR CODE HERE - 8-10 lines ##################
        import torch
        from torch.distributions import Categorical
        actor_features = self.actor_body(obs)
        logits = self.fc_action(actor_features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        critic_features = self.critic_body(obs)
        v = self.fc_critic(critic_features)
        ##############################################################
        ######################## END YOUR CODE #######################
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'v': v}

