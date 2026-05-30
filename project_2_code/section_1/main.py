from agent import *
from component import *
from utils import *
from plot import *

def run_steps(agent):
    '''
    the main training loop
    :param agent: the agent to be trained (A2CAgent/ REINFOCEAgent) both of them are inherited from the BaseAgent
    '''

    config = agent.config

    # Continue training until the total number of steps reaches the config.max_steps
    while True:

        if config.max_steps and agent.total_steps >= config.max_steps:

            # Evaluate the agent at the end of training
            agent.eval_episodes()

            agent.close()
            break

        agent.step()


def REINFORCE(**kwargs):
    '''
        Set the parameters (task, optimizer, network, ...) and start training of REINFORCE agent
        You do not need to implement anything in this function. However,
        you will need to use parameters defined here in other functions
    '''
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    # Define the task function in a wrapper
    config.task_fn = lambda: Task(config.game, config.seed)

    # Compute properties of state and action space: config.state_dim, config.action_dim, config.discrete (bool)
    config.eval_env = Task(config.game, config.seed)

    # Define the optimizer function and set its learning rate
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001)

    # Define the network function based on action space of the task (discrete vs continuous action space)
    if config.discrete:
        config.network_fn = lambda: CategoricalPolicyNet(
            config.action_dim, actor_body=FCBody(config.state_dim,hidden_units=(64,64)))
    else:
        config.network_fn = lambda: GaussianPolicyNet(
             config.action_dim, actor_body=FCBody(config.state_dim, hidden_units=(64,64)))

    config.discount = 0.99             # discount factor
    config.episode_length = 200        # max length of an episode for both Pendulum and CartPole-v0
    config.eval_episodes = 2           # number of episodes to evaluate agent
    if config.game == "Pendulum-v1":
        config.max_steps = int(1e6)    # number of total steps for training
    elif config.game == "CartPole-v1":
        config.max_steps = int(2e5)


    # Start training of the REINFORCE agent with the defined parameters stored in config
    run_steps(REINFORCEAgent(config))


def a2c(**kwargs):
    '''
        Set the parameters (task, optimizer, network, ...) and start training of A2C agent
        You do not need to implement anything in this function. However,
        you will need to use parameters defined here in other functions
    '''

    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)


    config.task_fn = lambda: Task(config.game,config.seed)
    config.eval_env = Task(config.game,config.seed)

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001)

    if config.discrete:
        config.network_fn = lambda: CategoricalActorCriticNet(
            config.action_dim,
            actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    else:
        config.network_fn = lambda: GaussianActorCriticNet(
            config.action_dim,
            actor_body=FCBody(config.state_dim,hidden_units=(64,64)), critic_body=FCBody(config.state_dim,hidden_units=(64,64)))

    config.discount = 0.99
    config.entropy_weight = 0.01
    config.value_loss_weight = 1
    config.gradient_clip = 5
    config.eval_episodes = 2
    if config.game == "Pendulum-v1":
        config.rollout_length = 64
        config.max_steps = int(1e6)
    elif config.game == "CartPole-v1":
        config.rollout_length = 100
        config.max_steps = int(2e5)


    run_steps(A2CAgent(config))


def set_seed(seed):
    '''
    set random seed to have reproducible results
    :param seed: integer
    '''
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')

    # game = 'CartPole-v0'
    game = 'Pendulum-v1'

    t = get_time_str()

    for seed in range(1):
        set_seed(seed=seed)
        REINFORCE(game=game,seed=seed, time=t)
        a2c(game=game, seed=seed, time=t)


    # The training return, policy_loss and value_loss data are logged in ./data/
    # You can also use the logged data in ./tf_log/ to track the training return using TensorBoard
    # The output of console is also logged in ./log/

    # You can use the function plot provided for you to draw the desired figures. (look at the end of plot.py for more details)
    REINFORCE_path = "./data/REINFORCEAgent_"+game+"_"+t+ "/"
    A2C_path = "./data/A2CAgent_" + game + "_" + t + "/"

    # Note that the plot will be saved in the last path if you want to plot multiple data in one figure. (A2C_path in this ecample)
    plot([
        (REINFORCE_path, "REINFORCE"),
        (A2C_path, "A2C"),
    ],"return", game)


    plot([(REINFORCE_path, "REINFORCE"),],"policy_loss", game)

    plot([(A2C_path, "A2C"),],"policy_loss", game)
    plot([(A2C_path, "A2C"),],"value_loss", game)

