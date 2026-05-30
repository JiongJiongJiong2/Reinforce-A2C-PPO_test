
from component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
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
        storage = Storage(config.rollout_length)
        state = self.state
        for _ in range(config.rollout_length):
            prediction = self.network(state)
            next_state, reward, terminal, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            storage.feed(prediction)
            storage.feed({'reward': tensor(reward).unsqueeze(-1),
                         'mask': tensor(1 - terminal).unsqueeze(-1)})

            state = next_state
            self.total_steps += 1

        self.state = state
        prediction = self.network(state)
        storage.feed(prediction)
        storage.placeholder()
        returns = prediction['v'].detach()

        '''
        TODO:
            Using the values we already stored in the storage (eg. storage.reward, storage.v , ...)
                1. Compute the n-step return for each state visisted in the stored rollout and store it properly in storage.ret

                    n-step return: G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n V_{t+n-1}(x_{t+n})

                    In order to get the most out of the data,
                        for the first state visited in the stored rollout calculate n-setp return
                        for the second state visited in the stored rollout calculate (n-1)-step return
                        ...
                        for the last state visited in the rollout calculate one-step return

                    (Note that we sample a fixed number of transitions (config.rollout_length) in the previous loop. There is a chance 
                    that the episode terminates before sampling all these transitions. In that case instead of breaking the sampling loop, we 
                    continue sampling in a new episode [it is handled in envs.OriginalReturnWrapper.step()].
                    So now to make sure the computed returns are correct (as the data in storage can be from more than one episode), we need
                    to keep track of terminal states in the memory. We have a variable called "mask" for each transition to identify terminal
                    states and should be used to compute the correct return.)


                2. Compute the advantage function for each (state,action) visited in the stored rollout and store it
                   properly in storage.advantage

            Hint: You might find "detach()" and reversed() functions useful in this block!

        '''
        #############################################################
        ############### YOUR CODE HERE - 5-7 lines ##################
        rewards = storage.reward
        masks = storage.mask
        values = storage.v
        ret_list = []
        adv_list = []
        G = returns
        for i in reversed(range(len(rewards))):
            G = rewards[i] + config.discount * G * masks[i]
            ret_list.insert(0, G)
            adv_list.insert(0, G - values[i].detach())
        storage.ret = ret_list
        storage.advantage = adv_list
        ##############################################################
        ######################## END YOUR CODE #######################


        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        '''
        TODO:
            Use entries.advantage and entries.ret that you just computed to calculate the policy loss and value loss
            Hint:
                    1. policy_loss = - 1/N \sum_{t=1}^{N} log\pi_{theta}(a_t|x_t)A(x_t, a_t)
                    2. value_loss = 1/2 ||return - V||_2^2
                    3. You might also find "entries.log_pi_a" and "entries.v" useful!
        '''
        #############################################################
        ############### YOUR CODE HERE - 2-4 lines ##################
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        ##############################################################
        ######################## END YOUR CODE #######################

        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        with open(self.log_dir + 'value_loss'+str(config.seed)+'.txt', 'a') as file:
            file.write(str(value_loss.item()) + '\n')


        with open(self.log_dir + 'policy_loss'+str(config.seed)+'.txt', 'a') as file:
            file.write(str(policy_loss.item()) + '\n')