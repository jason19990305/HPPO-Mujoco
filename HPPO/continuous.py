from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from .replaybuffer import ReplayBuffer
from .normalization import Normalization
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import random
import torch
import copy
import time
import os

import matplotlib.pyplot as plt


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
class Actor(nn.Module):
    def __init__(self,args,hidden_layers=[64,64]):
        super(Actor, self).__init__()
        self.actor_std_min = args.actor_std_min
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list make the input is num.states and output is num_actions
        hidden_layers.insert(0,self.num_states)
        hidden_layers.append(self.num_actions)
        print(hidden_layers)
        

        # create layers
        fc_list = []
        # 0 ~ length-1
        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            fc_list.append(layer)
        
        # weight and bias initialization
        for i in range(len(fc_list)-1):
            orthogonal_init(fc_list[i])
        orthogonal_init(fc_list[-1], gain=0.01)


        # put in ModuleList
        self.layers = nn.ModuleList(fc_list)

        #self.sigma = nn.Parameter(torch.ones(1,self.num_actions))          
        self.sigma = nn.Parameter(torch.log(1 * torch.ones(1,self.num_actions)+1e-8)) 


        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # when actor(s) will activate the function 
    def forward(self,s):
        for i in range(len(self.layers)):
            s = self.activation(self.layers[i](s))
        return s

    def get_dist(self,state):

        mean = self.forward(state)

        #std = self.sigma.expand_as(mean)
        
        #std = torch.sigmoid(std) + self.actor_std_min

        log_std = self.sigma.expand_as(mean)
        std = torch.exp(log_std) + 0.1

        try:
            dist = Normal(mean,std)
        except Exception as e:
            for param in self.parameters():
                print("actor parameter:",param.data)
            
        return dist

class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        # add in list
        hidden_layers.insert(0,self.num_states)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        fc_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            fc_list.append(layer)

        # weight and bias initialization
        for i in range(len(fc_list)-1):
            orthogonal_init(fc_list[i])

        # put in ModuleList
        self.layers = nn.ModuleList(fc_list)
        # activation function
        self.activation = nn.Tanh()

    def forward(self,s):
        for i in range(len(self.layers)-1):
            s = self.activation(self.layers[i](s))
        v_s = self.layers[-1](s)
        return v_s

class Agent():
    def __init__(self,args,hidden_layer_num_list=[64,64]):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:",self.device)

        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.actor_target = Actor(args,hidden_layer_num_list.copy())
        self.critic = Critic(args,hidden_layer_num_list.copy())

        self.actor_target.to(self.device)
        self.critic.to(self.device)

        self.actor_copy = Actor(args,hidden_layer_num_list.copy())
        self.epochs = args.epochs
        self.max_rollout_step = args.max_rollout_step
        self.mini_batch_size_ratio = args.mini_batch_size_ratio
        self.evaluate_freq = args.evaluate_freq

        self.num_actions = args.num_actions
        self.num_states = args.num_states
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_hindsight_goal = args.use_hindsight_goal
        self.use_state_norm = args.use_state_norm
        self.use_HGF = args.use_HGF
        self.use_goal_norm = args.use_goal_norm
        
        
        self.epsilon = args.epsilon
        self.entropy_coef = args.entropy_coef
        self.num_goal = args.num_goal
        self.lr = args.lr
        self.max_train_steps = args.max_train_steps
        self.save_model_freq_training_epoch = args.save_model_freq_training_epoch

        self.optimizer_actor = torch.optim.Adam(self.actor_target.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

        self.plot_count = 0

        self.x_max = args.x_max 
        self.x_min = args.x_min 
        self.y_max = args.y_max
        self.y_min = args.y_min

        self.actor_std_min = args.actor_std_min

        self.state_norm = Normalization(shape = self.num_states - self.num_goal * 2)  # Trick 2:state normalization
        self.goal_norm = Normalization(shape = self.num_goal)

        self.state_norm_target = Normalization(shape = self.num_states - self.num_goal * 2)  # Trick 2:state normalization
        self.goal_norm_target = Normalization(shape = self.num_goal)

        #print(device)
    def evaluate(self,des,ach,state):
        # numpy convert to tensor.[num_state]->[1,num_state]
        des = torch.tensor(des, dtype=torch.float)
        ach = torch.tensor(ach, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)

        s = torch.cat((des,ach,state))

        # if don't use detach() the requires_grad = True . Can't call numpy()
        # or able use torch.no_grad
        with torch.no_grad():
            a = self.actor(s)
            a = torch.clamp(a ,-1 , 1)
        #return self.choose_action(des,ach,state)
        return a.numpy().flatten()

    def choose_action(self,des,ach,state):

        des = torch.tensor(des, dtype=torch.float)
        ach = torch.tensor(ach, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(torch.cat((des,ach,state)) , 0)
        
        with torch.no_grad():
            # input state and forward. predict mean and std then get the Normal distribution
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a ,-1 , 1)  # clip
        
        return a.numpy().flatten()
    
    def evaluate_policy(self,args,env,render=False,first=False):

        times = 200
        evaluate_reward = 0
        sucess_list = []

        for i in range(times):
            s = env.reset()[0]
            done = False
            episode_reward = 0
            episode_steps = 0
            while True:
                if render:
                    time.sleep(0.08)

                tmp_s = copy.deepcopy(s[self.num_goal*2:])
                tmp_des = copy.deepcopy(s[:self.num_goal])
                tmp_ach = copy.deepcopy(s[self.num_goal:self.num_goal*2])

                if self.use_state_norm :
                    if first:
                        self.state_norm_target(tmp_s,update=True)
                    # normalizer same as training
                    tmp_s = self.state_norm(tmp_s,update=False)

                if self.use_goal_norm :
                    if first:
                        self.goal_norm_target(tmp_des,update=True)
                        self.goal_norm_target(tmp_ach,update=True)

                    tmp_des = self.goal_norm(tmp_des,update=False)
                    tmp_ach = self.goal_norm(tmp_ach,update=False)
                
                a = self.evaluate(tmp_des,tmp_ach,tmp_s)
            
                s_, r, done, truncted,_ = env.step(a)
                episode_reward += r
                s = s_
                
                if truncted or done:
                    sucess_list.append(done)
                    break
                episode_steps += 1
            evaluate_reward += episode_reward
        sucess_count = np.sum(sucess_list)
        return sucess_count / len(sucess_list)*100
    
    def generate_subgoal(self,new_goals,des_goals):
        num_origin_goals = len(new_goals)

        # draw plot
        ach_goals = np.array(new_goals) 
        des_goals = np.array(des_goals)
        
        
        # remove useless data
        sampled_goal_num = 100
        ach_goals = np.round(ach_goals, decimals = 2)
        des_goals = np.unique(des_goals,axis=0)

        dg_max = np.max(des_goals, axis=0)
        dg_min = np.min(des_goals, axis=0)

        #print("max,min:",dg_max,dg_min)
        g_ind = (dg_min != dg_max)
        subgoals_ind = (np.sum((ach_goals[:, g_ind] > dg_max[g_ind]) |
                                      (ach_goals[:, g_ind] < dg_min[g_ind]), axis = -1) == 0)
        subgoals = ach_goals[subgoals_ind]
        rest = ach_goals[1-subgoals_ind]
        if subgoals.shape[0] < sampled_goal_num and rest.shape[0] > 0:
            dist_to_dg_center = np.linalg.norm(rest - np.mean(des_goals, axis = 0), axis=1)
            ind_subgoals = np.argsort(dist_to_dg_center)
            rest = rest[ind_subgoals[:(sampled_goal_num - subgoals.shape[0])]]
            subgoals = np.concatenate([subgoals, rest], axis = 0)
        ach_goals = subgoals
        size = min(sampled_goal_num, ach_goals.shape[0])
        #  initialization
        init_ind = np.random.randint(ach_goals.shape[0])
        selected_subgoals = ach_goals[init_ind:init_ind + 1]
        ach_goals = np.delete(ach_goals, init_ind, axis=0)
        dists = np.linalg.norm(
                    np.expand_dims(selected_subgoals, axis=0) - np.expand_dims(ach_goals, axis=1),
                    axis=-1)
        for g in range(size-1):
            selected_ind = np.argmax(np.min(dists, axis=1))
            selected_subgoal = ach_goals[selected_ind:selected_ind+1]
            selected_subgoals = np.concatenate((selected_subgoals, selected_subgoal), axis = 0)

            ach_goals = np.delete(ach_goals, selected_ind, axis = 0)
            dists = np.delete(dists, selected_ind, axis = 0)

            new_dist = np.linalg.norm(
                np.expand_dims(selected_subgoal, axis=0) - np.expand_dims(ach_goals, axis=1),axis=-1)

            dists = np.concatenate((dists, new_dist), axis=1)

        result_subgoals = selected_subgoals
        result_subgoals = torch.tensor(result_subgoals).to(self.device)
        print("subgoals:",result_subgoals.shape)
        self.plot_count += 1
       

        return result_subgoals

    def compute_fake_data(self,episode_batch,new_goals,des_goals):


        # generate subgoal
        new_goals = np.array(new_goals)
        num_origin_goals = len(new_goals)

        index1 = np.logical_and(new_goals[:,0] < self.x_max , new_goals[:,0] > self.x_min)
        index2 = np.logical_and(new_goals[:,1] < self.y_max , new_goals[:,1] > self.y_min)
        index  = np.logical_and(index1 , index2)

        #print(index)
        new_goals = new_goals[index] 

        # remove same data
        ach_goals = np.round(new_goals, decimals = 2)
        ach_goals = np.unique(ach_goals,axis=0)

        print("%d -> %d -> %d"%(self.max_rollout_step,len(new_goals),len(ach_goals)))

        if self.use_HGF :
            new_goals = self.generate_subgoal(new_goals,des_goals)
        else:
            k=100
            
            if len(ach_goals) < k:
                ratio = k // len(ach_goals) + 1
                ach_goals = np.repeat(ach_goals,ratio,axis=0)

            index = np.random.choice(len(ach_goals), k, replace=False)
            new_goals = ach_goals[index]
            new_goals = torch.tensor(new_goals).to(self.device)
            
        print("Total number of goals : %d -> %d "%(num_origin_goals , len(new_goals)))

        num_subgoals = len(new_goals)
        
        num_origin_count = self.replay_buffer.count

        # origin data
        mb_s = torch.tensor(np.array(self.replay_buffer.s), dtype=torch.float).to(self.device)
        mb_a =  torch.tensor(np.array(self.replay_buffer.a), dtype=torch.long).to(self.device)
        mb_s_ = torch.tensor(np.array(self.replay_buffer.s_), dtype=torch.float).to(self.device)
        mb_r = torch.tensor(np.array(self.replay_buffer.r), dtype=torch.float).to(self.device)
        mb_done = torch.tensor(np.array(self.replay_buffer.done), dtype=torch.float).to(self.device)
        mb_discount = torch.tensor(np.array(self.replay_buffer.discount), dtype=torch.float).to(self.device)
        mb_weight = torch.ones((num_origin_count,1)).to(self.device)

        # state normalization
        if self.use_state_norm :
            # state normalization
            mean = torch.Tensor(self.state_norm.running_ms.mean).to(self.device)
            std = torch.Tensor(self.state_norm.running_ms.std).to(self.device)
            std = torch.clamp(std,10e-4)
            #std += 10e-8
            #std = torch.sqrt(std)

            mb_s[:,self.num_goal*2:] = (mb_s[:,self.num_goal*2:]  - mean)/(std)
            mb_s_[:,self.num_goal*2:] =  (mb_s_[:,self.num_goal*2:]  - mean)/(std )

                
        if self.use_goal_norm :
            # goal normalization
            mean = torch.Tensor(self.goal_norm.running_ms.mean).to(self.device)
            std = torch.Tensor(self.goal_norm.running_ms.std).to(self.device)
            std = torch.clamp(std,10e-4)
            #std += 10e-8
            #std = torch.sqrt(std)
            
            # des goal
            mb_s[:,:self.num_goal] = (mb_s[:,:self.num_goal]  - mean)/(std)
            mb_s_[:,:self.num_goal] = (mb_s_[:,:self.num_goal]  - mean)/(std)

            # ach goal
            mb_s[:,self.num_goal:self.num_goal*2] = (mb_s[:,self.num_goal:self.num_goal*2]  - mean)/(std)
            mb_s_[:,self.num_goal:self.num_goal*2] = (mb_s_[:,self.num_goal:self.num_goal*2]  - mean)/(std)

        # clip
        mb_s = torch.clamp(mb_s , -5 , 5)
        mb_s_ = torch.clamp(mb_s_ , -5 , 5)

        # standard data convert to tensor
        self.replay_buffer.s = mb_s
        self.replay_buffer.a = mb_a
        self.replay_buffer.s_ = mb_s_
        self.replay_buffer.r = mb_r
        self.replay_buffer.done = mb_done
        self.replay_buffer.discount = mb_discount

        # if not hindsight data then weight is 1.
        self.replay_buffer.weight = mb_weight

        #print("The count of object moved : ",(self.replay_buffer.r > 0).sum())

        if not self.use_hindsight_goal:
            return
        
        threshold = 0.05
        # generate hindsight data
        fake_data_count = 0
        for mb_batch in episode_batch:
            mb_state,mb_action,mb_next_state = copy.deepcopy(mb_batch) # unpack episode data

            #init ach goal
            init_ach = mb_state[0,self.num_goal:self.num_goal*2]
            index = (new_goals - init_ach).pow(2).sum(axis=1).sqrt() >= threshold
            sub_goal = new_goals[index]
            #print("subgoal %d -> %d"%(len(new_goals),len(sub_goal)))

            num_subgoals = len(sub_goal)


            num_episode_step = len(mb_state)
            
            fake_data_count += num_episode_step * num_subgoals


            fake_mb_state = torch.tile(mb_state,(num_subgoals,1)) # expand data
            fake_mb_next_state = torch.tile(mb_next_state,(num_subgoals,1))# expand data

            # expand dim of action
            oringin_mb_action = torch.tile(mb_action,(num_subgoals,1)) # expand action dim 

            # replace des goal
            expand_new_goals = sub_goal.repeat_interleave(num_episode_step,dim=0)
            origin_mb_state = copy.deepcopy(fake_mb_state)

            fake_mb_state[:,:self.num_goal] = expand_new_goals
            fake_mb_next_state[:,:self.num_goal] = expand_new_goals


            # get the achieve goal from fake data
            fake_mb_ach_goal = fake_mb_next_state[:,self.num_goal:self.num_goal*2]
            fake_mb_des_goal = fake_mb_next_state[:,:self.num_goal]
            fake_mb_pervious_ach_goal = fake_mb_state[:,self.num_goal:self.num_goal*2]


            fake_cost = (fake_mb_ach_goal - fake_mb_des_goal).pow(2).sum(axis=1).sqrt() < threshold
            fake_cost = fake_cost.view(-1,1)
            
            #print(fake_mb_ach_goal.shape,expand_init_ach.shape)
            fake_bonus = (fake_mb_ach_goal - fake_mb_pervious_ach_goal).pow(2).sum(axis=1).sqrt() > 0.001
            fake_bonus = fake_bonus.view(-1,1)
            fake_mb_r = (fake_cost * 50.0) + fake_bonus * 1
            fake_mb_r = fake_mb_r.view(-1,num_episode_step)

            done = copy.deepcopy(fake_cost)
            # make index for filtering
            
            fake_cost = fake_cost.view(-1,num_episode_step)
            index = torch.zeros_like(fake_mb_r,dtype=torch.bool)
            for ep in range(len(fake_mb_r)):
                tmp = (fake_cost[ep])
                ind = tmp.nonzero()
                if ind.size()[0]==0:
                    dw_length = num_episode_step
                else :
                    dw_length = ind[0][0] + 1
                index[ep][:dw_length] = True   
            #print("-------")
            fake_mb_r = fake_mb_r.view(-1,1)
            index = index.view(-1)

            # the reward suggest same with episode length
            discount = torch.pow(self.gamma,torch.arange(num_episode_step) + 1).to(self.device)
            discount = torch.tile(discount,(num_subgoals,1)).view([-1,1])


            # state preprocess
            #origin_mb_state = torch.clamp(origin_mb_state , -200 , 200)
            #fake_mb_state = torch.clamp(fake_mb_state , -200 , 200)
            #fake_mb_next_state = torch.clamp(fake_mb_next_state , -200 , 200)

            if self.use_state_norm :
                # state normalization
                mean = torch.Tensor(self.state_norm.running_ms.mean).to(self.device)
                std = torch.Tensor(self.state_norm.running_ms.std).to(self.device)
                std = torch.clamp(std,10e-4)
                #std += 10e-8
                #std = torch.sqrt(std)

                origin_mb_state[:,self.num_goal*2:] = (origin_mb_state[:,self.num_goal*2:]  - mean)/(std)
                fake_mb_state[:,self.num_goal*2:] =  (fake_mb_state[:,self.num_goal*2:]  - mean)/(std )
                fake_mb_next_state[:,self.num_goal*2:] = (fake_mb_next_state[:,self.num_goal*2:]  - mean)/(std) 

                
            if self.use_goal_norm :
                # goal normalization
                mean = torch.Tensor(self.goal_norm.running_ms.mean).to(self.device)
                std = torch.Tensor(self.goal_norm.running_ms.std).to(self.device)
                std = torch.clamp(std,10e-4)
                #std += 10e-8
                #std = torch.sqrt(std)
                
                # des goal
                origin_mb_state[:,:self.num_goal] = (origin_mb_state[:,:self.num_goal]  - mean)/(std )
                fake_mb_state[:,:self.num_goal] = (fake_mb_state[:,:self.num_goal]  - mean)/(std)
                fake_mb_next_state[:,:self.num_goal] = (fake_mb_next_state[:,:self.num_goal]  - mean)/(std)

                # ach goal
                origin_mb_state[:,self.num_goal:self.num_goal*2] = (origin_mb_state[:,self.num_goal:self.num_goal*2]  - mean)/(std)
                fake_mb_state[:,self.num_goal:self.num_goal*2] = (fake_mb_state[:,self.num_goal:self.num_goal*2]  - mean)/(std)
                fake_mb_next_state[:,self.num_goal:self.num_goal*2] = (fake_mb_next_state[:,self.num_goal:self.num_goal*2]  - mean)/(std)

            # clip
            origin_mb_state = torch.clamp(origin_mb_state , -5 , 5)
            fake_mb_state = torch.clamp(fake_mb_state , -5 , 5)
            fake_mb_next_state = torch.clamp(fake_mb_next_state , -5 , 5)

            with torch.no_grad():
                old_dist = self.actor_target.get_dist(origin_mb_state)
                new_dist = self.actor_target.get_dist(fake_mb_state)

                old_log_probs = old_dist.log_prob(oringin_mb_action)
                new_log_probs = new_dist.log_prob(oringin_mb_action)

            
            d_log = new_log_probs.sum(axis = 1, keepdim=True) - old_log_probs.sum(axis = 1, keepdim=True)
            d_log = d_log.view(-1,num_episode_step)
            
            h_ratio = torch.exp(d_log.cumsum(dim=1))#.view([-1,1])
            h_ratios_sum = torch.sum(h_ratio, dim=1, keepdim = True)
            h_ratio = (h_ratio / (h_ratios_sum)).view([-1,1])
            

            # user index filtering data
            fake_mb_state = fake_mb_state[index]
            fake_mb_next_state = fake_mb_next_state[index]
            oringin_mb_action = oringin_mb_action[index]
            fake_mb_r = fake_mb_r[index]
            done = done[index]
            discount = discount[index]
            h_ratio = h_ratio[index]

            #print("------------")

            
            self.replay_buffer.s = torch.cat((self.replay_buffer.s,fake_mb_state))
            self.replay_buffer.s_ = torch.cat((self.replay_buffer.s_,fake_mb_next_state))
            self.replay_buffer.a = torch.cat((self.replay_buffer.a,oringin_mb_action))
            self.replay_buffer.r = torch.cat((self.replay_buffer.r,fake_mb_r))
            self.replay_buffer.done = torch.cat((self.replay_buffer.done,done))
            self.replay_buffer.discount = torch.cat((self.replay_buffer.discount,discount))
            self.replay_buffer.weight = torch.cat((self.replay_buffer.weight,h_ratio))

        total = fake_data_count+num_origin_count
        
            
        print("norm -> new state max :",self.replay_buffer.s.max())
        print("norm -> new state min :",self.replay_buffer.s.min())
       

        #self.replay_buffer.s = torch.clamp(self.replay_buffer.s,-5,5)
        
        # hindsight data filter
        print("weight [max,min] : [%0.2f,%0.2f]"%(self.replay_buffer.weight.max(),self.replay_buffer.weight.min()))
        if torch.isinf(self.replay_buffer.weight).any():
            print("***** isinf *****")
            mask = ~torch.isnan(self.replay_buffer.weight).view(-1)
            self.replay_buffer.s = self.replay_buffer.s[mask]
            self.replay_buffer.a = self.replay_buffer.a[mask]
            self.replay_buffer.s_ = self.replay_buffer.s_[mask]
            self.replay_buffer.r = self.replay_buffer.r[mask]
            self.replay_buffer.discount = self.replay_buffer.discount[mask]
            self.replay_buffer.weight = self.replay_buffer.weight[mask]

        if torch.isnan(self.replay_buffer.weight).any():
            print("***** isnan *****")
            mask = ~torch.isnan(self.replay_buffer.weight).view(-1)
            self.replay_buffer.s = self.replay_buffer.s[mask]
            self.replay_buffer.a = self.replay_buffer.a[mask]
            self.replay_buffer.s_ = self.replay_buffer.s_[mask]
            self.replay_buffer.r = self.replay_buffer.r[mask]
            self.replay_buffer.discount = self.replay_buffer.discount[mask]
            self.replay_buffer.weight = self.replay_buffer.weight[mask]

        dw_index = self.replay_buffer.r >= 50
        self.replay_buffer.r[dw_index] = 50.0
        print("move :%d , win :%d"%((self.replay_buffer.r > 0).sum(),(self.replay_buffer.r == 50).sum()))
        self.replay_buffer.r[~dw_index] = 0

        after = len(self.replay_buffer.weight)

        print(torch.unique(self.replay_buffer.r))

        print("filter data %d -> %d [%0.2f%%]"%(total,after,after/total*100))

    def train(self,args,env,env_name):
        total_steps = 0  # Record the total steps during the training
        training_count = 0

        self.replay_buffer = ReplayBuffer(args)
         
        home_directory = os.path.expanduser( '~' )
        log_dir=home_directory+'/Log/HPPO_'+env_name+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        writer = SummaryWriter(log_dir=log_dir)

        rollout_step = 0

        # evaluate
        success_rate = self.evaluate_policy(args, env,first=True)
        self.update_normalizer()
        self.env = env
        print("Evaluate Sucess rate:%0.2f%%"%(success_rate))
        print("-----------")

        writer.add_scalar('step_success_rate_{}'.format(env_name), success_rate, global_step=total_steps)
        
        while total_steps < args.max_train_steps:

            new_goals = []
            des_goals = []
            episode_batch = []
            rollout_step = 0            

            while True:

                # Standar rollout
                s = env.reset()[0]
                des_goal = s[:self.num_goal]
                des_goals.append(des_goal)

                episode_steps = 0
                
                mb_action = []
                mb_state = []
                mb_next_state =[]

                discount = self.gamma

                while True:

                    # for actor
                    tmp_s = copy.deepcopy(s[self.num_goal*2:])
                    tmp_des = copy.deepcopy(s[:self.num_goal])
                    tmp_ach = copy.deepcopy(s[self.num_goal:self.num_goal*2])

                    # state normalization
                    if self.use_state_norm :
                        self.state_norm_target(copy.deepcopy(tmp_s),update=True)
                        
                        tmp_s = self.state_norm(tmp_s,update=False)
                        
                    # goal normalization
                    if self.use_goal_norm :
                        self.goal_norm_target(copy.deepcopy(tmp_des),update=True)
                        self.goal_norm_target(copy.deepcopy(tmp_ach),update=True)

                        tmp_des = self.goal_norm(tmp_des,update=False)
                        tmp_ach = self.goal_norm(tmp_ach,update=False)
                        

                    a = self.choose_action(tmp_des,tmp_ach,tmp_s)  # Action and the corresponding log probability
                    s_, r, done, truncated,_ = env.step(a)# Interaction with environment
                    
                    # store achieve goal as subgoal
                    ach_goal = s_[self.num_goal:self.num_goal*2]
                    new_goals.append(ach_goal)

                    # store data
                    mb_action.append(a)
                    mb_state.append(s)
                    mb_next_state.append(s_)

                    # update data
                    s = s_
                    total_steps += 1
                    rollout_step += 1
                    discount *= self.gamma


                    self.replay_buffer.store(s,a,r,s_,done,discount)
                    
                    if truncated or done:
                        break
                    
                    if rollout_step >= self.max_rollout_step:
                        break

                    episode_steps += 1

                # convert to tensor and store in episode_batch
                s = torch.tensor(np.array(mb_state) , dtype=torch.float).to(self.device)
                a = torch.tensor(np.array(mb_action) , dtype=torch.long).to(self.device)
                s_ = torch.tensor(np.array(mb_next_state), dtype=torch.float).to(self.device)
                episode_batch.append((s,a,s_))

                # if get enough data then end loop
                if rollout_step >= self.max_rollout_step:
                    break

            # Have enough data of subgoal and batch then generate fack data and training
            start = time.time()
            print("-----------")
            # generate fake data from subgoals
            m_r = np.array(self.replay_buffer.r)
            print("The count of object moved : ",np.sum(m_r > 0))
            writer.add_scalar('step_move_object{}'.format(env_name), np.sum(m_r> 0), global_step=total_steps)

            # generate fake data for training
            self.compute_fake_data(episode_batch,new_goals,des_goals)
            print("Training epoch:",training_count,"\tStep:",total_steps,"/",args.max_train_steps,"\t")

            # update Actor Critic
            self.update(self.replay_buffer, total_steps)

            if self.actor_std_min > 1:
                self.update_actor_std(self.actor_std_min)
                print("Actor sigma set to :",self.actor_std_min)
                self.actor_std_min *= 0.99

            end = time.time()
            print("Spending time:%02d:%02d"%(int(end-start)//60,int(end-start)%60))
            
            # evalute and write tensorboard
            if training_count % self.evaluate_freq == 0:
                success_rate = self.evaluate_policy(args, env)
                print("Evaluate Sucess rate:%0.2f%%"%(success_rate))
                print("-----------")
                writer.add_scalar('step_success_rate_{}'.format(env_name), success_rate, global_step=total_steps)
            
            # save model
            if training_count % self.save_model_freq_training_epoch == 0:
                path_actor = "model/HPPO_Actor_"+env_name+"_"+str(training_count)+".pt"
                path_critic = "model/HPPO_Critic_"+env_name+"_"+str(training_count)+".pt"
                self.save_actor_model(path_actor)
                self.save_critic_model(path_critic)

            # after update and evaluate. upate the state/goal normalizer
            self.update_normalizer()

            training_count += 1
            # clear buffer
            rollout_step = 0
            new_goals = []
            self.replay_buffer.clear()
        # end of training
        success_rate = self.evaluate_policy(args, env)
        print("Evaluate Sucess rate:%0.2f%%"%(success_rate))
        print("-----------")
        writer.add_scalar('step_success_rate_{}'.format(env_name), success_rate, global_step=total_steps)

    def update_actor_std(self,std_min):
        std_min = torch.tensor(std_min)
        self.actor.actor_std_min = std_min
        self.actor_target.actor_std_min = std_min.to(self.device)

                
    def update_normalizer(self):
        # update normalizer
        self.state_norm.running_ms.mean = self.state_norm_target.running_ms.mean
        self.state_norm.running_ms.std = self.state_norm_target.running_ms.std

        self.goal_norm.running_ms.mean = self.goal_norm_target.running_ms.mean
        self.goal_norm.running_ms.std = self.goal_norm_target.running_ms.std

    def update(self,replay_buffer,total_steps):

        s, a, r, s_, done , weights , discount = replay_buffer.unpack()  # Get training data .type is tensor

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        weights = weights.to(self.device)
        discount = discount.to(self.device)

        dw = (r > 45).float()
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)

            adv = r +  self.gamma  * (1 - dw) * vs_ - vs#(r / 50.0 - 1.0)*self.gamma * vs_ - vs
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 
            #adv = discount * adv
            #adv = weights * adv

            old_log_prob = self.actor_target.get_dist(s).log_prob(a)  

        
        num_batch = len(r)
        #num_batch_ratio = num_batch // self.mini_batch_size
        mini_batch_size = num_batch // self.mini_batch_size_ratio

        print("num training batch :",num_batch)
        print("num training mini_batch_size :",mini_batch_size)

        #print("Actor std:",torch.exp(self.actor.std) + 1.1)

        print("Actor std:",torch.sigmoid(self.actor.sigma) +  self.actor.actor_std_min)
        print("Actor sigma :",torch.exp(self.actor.sigma))

        inds = np.arange(num_batch)# 0 ~ num_batch

        for i in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0,num_batch,mini_batch_size):
                end = start + mini_batch_size
                index = inds[start:end]

                # get the current distribution of actor
                new_dist = self.actor_target.get_dist(s[index])
                # get entropy of actor distribution
                dist_entropy = new_dist.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                # get the new log probability
                new_log_prob = new_dist.log_prob(a[index])
                # shape = [mini_batch_size , num_action]. Summation over the axis=1 -> [1,num_action]
                ratios = torch.exp(new_log_prob.sum(1, keepdim=True) - old_log_prob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                #ratios = torch.clamp(ratios,0,1e3)

                # adv.shape = [mini_batch_size,1]
                p1 = ratios  * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                # clip ratios to 1-epsilon ~ 1+epsilon
                p2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # choice the minimum value of p1 or p2. 

                actor_loss = -torch.min(p1, p2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy


                if torch.isinf(actor_loss).sum() > 0 or torch.isinf(actor_loss.mean()).sum() > 0:
                    print("*****is inf .skip*****")
                    print("weights max:",weights.max())
                    print("weights min:",weights.min())
                    print("actor loss max:",actor_loss.max())
                    print("actor loss min:",actor_loss.min())
                    print("ratios max:",ratios.max())
                    print("ratios min:",ratios.min())
                    print("actor loss mean:",actor_loss.mean())
                    exit()

                if torch.isnan(actor_loss).sum() > 0 or torch.isnan(actor_loss.mean()).sum() > 0:
                    print("*****is nan .skip*****")
                    print("weights max:",weights.max())
                    print("weights min:",weights.min())
                    print("actor loss max:",actor_loss.max())
                    print("actor loss min:",actor_loss.min())
                    print("ratios max:",ratios.max())
                    print("ratios min:",ratios.min())
                    print("actor loss mean:",actor_loss.mean())
                    exit()

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_target.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        sd=self.actor_target.state_dict()
        self.actor.load_state_dict(sd)
        #self.lr_decay(total_steps=total_steps)

    def save_actor_model(self,path):
        print("Save actor model:",path)
        torch.save(self.actor, path)
    def save_critic_model(self,path):
        print("Save critic model:",path)
        torch.save(self.critic, path)
    def load_actor_model(self,path):
        print("Load actor model:",path)
        self.actor = torch.load(path).train()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
    def load_critic_model(self,path):
        print("Load critic model:",path)
        self.critic = torch.load(path).train()
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)


    def lr_decay(self, total_steps):
        lr_a_now = self.lr * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr * (1 - total_steps / self.max_train_steps)

        for opt in self.optimizer_actor.param_groups:
            opt['lr'] = lr_a_now + 10e-6
        for opt in self.optimizer_critic.param_groups:
            opt['lr'] = lr_c_now + 10e-6