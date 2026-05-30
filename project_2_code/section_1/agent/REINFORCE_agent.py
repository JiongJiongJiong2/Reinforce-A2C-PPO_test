
from component import *
from .BaseAgent import *


class REINFORCEAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.state = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.episode_length)
        state = self.state

        for _ in range(config.episode_length):
            prediction = self.network(state)
            next_state, reward, terminal, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            storage.feed(prediction)
            storage.feed({'reward': tensor(reward).unsqueeze(-1),
                         'mask': tensor(1 - terminal).unsqueeze(-1)})
            state = next_state
            self.total_steps += 1
            if terminal:
                break

        '''
        TODO:
            We need to approximate the Q function for each state in the stored trajectory using the stored rewards.
            That is we want
                Q(x_t,a_t)= \ sum_{t'=t}^{T-1} \gamma^{t'-t} r(x_{t'},a_{t'})
            based on the trajectory stored in the storage in lines 20-31.
            Store these values properly in storage.ret
        '''
        #############################################################
        ############### YOUR CODE HERE - 3-5 lines ##################
        rewards = storage.reward
        masks = storage.mask
        returns = []
        G = 0
        for r, mask in zip(reversed(rewards), reversed(masks)):
            G = r + config.discount * G * mask
            returns.insert(0, G)
        storage.ret = returns
        ##############################################################
        ######################## END YOUR CODE #######################

        entries = storage.extract(['log_pi_a', 'ret'])

        '''
        TODO:
            Use entries.ret that you just computed to calculate the policy loss
            Hint:
                    1. policy_loss = - 1/N \sum_{t=1}^{N} log\pi_{theta}(a_t|x_t)Q(x_t, a_t)
                    2. You might also find "entries.log_pi_a" useful!
        '''
        #############################################################
        ############### YOUR CODE HERE - 1-2 lines ##################
        policy_loss = -(entries.log_pi_a * entries.ret).mean()
        ##############################################################
        ######################## END YOUR CODE #######################

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


        with open(self.log_dir + 'policy_loss'+str(config.seed)+'.txt', 'a') as file:
            file.write(str(policy_loss.item()) + '\n')