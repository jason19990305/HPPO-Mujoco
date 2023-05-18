from PPO2.continuous import Agent
from PPO2.normalization import Normalization
import gym
import numpy as np
import argparse

class main():
    def __init__(self,args,env_name):
        env = gym.make(env_name)    # The wrapper encapsulates the gym env
        num_achive = env.observation_space['achieved_goal'].shape[0]
        num_desire = env.observation_space['desired_goal'].shape[0]
        num_obs =  env.observation_space['observation'].shape[0]
        num_actions = env.action_space.shape[0]
        num_states = num_obs + num_desire + num_achive
        # args
        args.num_actions = num_actions
        args.num_states = num_states
        args.num_goal = num_desire
        # env
        env = EnvPanda(env_name)

        # print args
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent        
        hidden_layer_num_list = [256,256,256]
        agent = Agent(args,hidden_layer_num_list) 

        #agent.load_actor_model("PPO_Actor_"+env_name+"_pre.pt")

        print(agent.actor)
        print(agent.critic)
        print("---------------")

        # training
        agent.train(args,env,env_name) 


        path_actor = "model/HPPO_Actor_"+env_name+".pt"
        path_critic = "model/HPPO_Critic_"+env_name+".pt"
        print("Saving actor model at:",path_actor)
        print("Saving critic model at:",path_critic)
        agent.save_actor_model(path_actor)
        agent.save_critic_model(path_critic)

        # save the normalization paramter to yaml 
        agent.state_norm.save_yaml(env_name+"_state.yaml")
        agent.goal_norm.save_yaml(env_name+"_goal.yaml")

        # evaluate 
        env_evaluate = EnvPanda(env_name,render_mode='human')
        
        for i in range(10000):
            evaluate_reward = agent.evaluate_policy(args, env_evaluate,render=True)
            print("Evaluate reward:",evaluate_reward)

   
# overwrite env
class EnvPanda(gym.Env):
    def __init__(self,env_name,render_mode=None):
        if render_mode == None:
            self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
            self.render_mode = render_mode
        else:
            self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
            self.render_mode = render_mode
        self.threshold = 0.05

    def distance(self,p1,p2):
        d = (p1-p2)**2
        d = np.sum(d)
        d = np.sqrt(d)
        return d
    def compute_reward(self,ach,des):
        costs = self.distance(ach,des)
        reward = 50.0 if costs < self.threshold else 0

        return reward
        
    def step(self, action):
        if self.render_mode == 'human':
            self.env.render()

        state, reward,truncted, info = self.env.step(action)   # calls the gym env methods
        ach = state['achieved_goal']
        des = state['desired_goal']
        obs = state['observation']

        reward = self.compute_reward(ach,des)
        
        bdw = True if reward > 0 else False
        
        if reward == 0:
            if self.distance(ach,self.pervious_ach) > 0.001:
                reward = 0.01

        self.pervious_ach = ach
        
        state = np.hstack([des,ach,obs])
        return state, reward, bdw , truncted, info

    def reset(self):

        while True:
            obs = self.env.reset()   # same for reset
            
            #ach = self.object
            ach = obs['achieved_goal']
            des = obs['desired_goal']
            obs = obs['observation']
            if self.compute_reward(ach,des) > 0:
                continue
            else:
                break

        self.init_ach = ach
        self.count = 1
        self.pervious_ach = ach

        state = np.hstack([des,ach,obs])
        return [state]#[np.hstack([obs,des,ach])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for HPPO-continuous")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epochs", type=int, default=10, help="HPPO training iteration parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="HPPO clip parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.00, help="Trick 5: policy entropy")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--max_rollout_step", type=int, default=3200, help=" Maximum number of rollout steps")
    parser.add_argument("--use_hindsight_goal", type=bool, default=True, help="Flag for using hindsight goal")
    parser.add_argument("--evaluate_freq", type=int, default=1, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_model_freq_training_epoch", type=int, default=10, help="Save model frequance")
    parser.add_argument("--mini_batch_size_ratio", type=int, default=512, help="mini_batch_size_ratio")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Flag for using state normalization")
    parser.add_argument("--use_goal_norm", type=bool, default=True, help="Flag for using state normalization")
    parser.add_argument("--use_HGF", type=bool, default=True, help="Flag for using hindsight goal filter")
    parser.add_argument("--actor_std_min", type=float, default=1.1, help="Flag for using hindsight goal filter")

    parser.add_argument("--env_name", type=str, default="FetchReach-v1", help=" Maximum number of rollout steps")


    args = parser.parse_args()
    env_name = args.env_name

    main(args,env_name=env_name)


