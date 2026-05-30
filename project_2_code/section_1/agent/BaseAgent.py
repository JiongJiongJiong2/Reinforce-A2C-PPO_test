
from utils import *
import pickle

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        agent_name = self.__class__.__name__
        self.log_dir = "data/"+agent_name+"_"+config.game+"_"+config.time+"/"
        mkdir(self.log_dir)

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))


    def eval_episode(self):
        '''
        evaluate the policy for one episode
        :return: sum of rewards in an episode
        '''
        env = self.config.eval_env
        state = env.reset()
        '''
        TODO:
            Evaluate the policy by calculating the sum of rewards when following agent's current policy (in self.network) in an episode
            Hint: you might find self.network and env.step useful
        '''
        #############################################################
        ############### YOUR CODE HERE - 6-8 lines ##################
        ret = 0
        done = False
        while not done:
            prediction = self.network(state)
            action = to_np(prediction['action'])
            state, reward, done, info = env.step(action)
            ret += reward
        ##############################################################
        ######################## END YOUR CODE #######################
        return ret


    def eval_episodes(self):
        '''
        evaluate the policy for self.config.eval_episodes times
        :return: mean total reward of self.config.eval_episodes episodes
        '''
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(total_rewards)
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        '''
        logging the total reward of each episode during training
        '''
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                with open(self.log_dir+'return'+str(self.config.seed)+'.txt', 'a') as file:
                    file.write(str(ret)+'\n')
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError



