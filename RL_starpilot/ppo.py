#code inspiration from https://github.com/joonleesky/train-procgen-pytorch
from torch.types import Storage

#libs.joon is a copied libary
from libs.joon.logger import Logger
from libs.joon.misc_util import adjust_lr

import torch
import torch.optim as optim
import numpy as np
import pickle
import HyperPrams

class PPO_agent():
    def __init__(self,
                
                env,
                policy,
                logger : Logger,
                storage : Storage,
                device,
                
                hyperPrams : HyperPrams,

                n_checkpoints : int,
                n_steps : int = 128,
                n_envs : int = 8,
                epoch : int = 3,
                mini_batch_per_epoch : int = 8,
                mini_batch_size : int = 32*8,
                
                ):

        self.env = env;
        self.policy = policy;
        self.logger = logger;
        self.storage = storage;
        self.device = device;
        self.num_checkpoints = n_checkpoints;

        self.hyperPrams = hyperPrams;
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperPrams.learning_rate, betas=(0.9, 0.999), eps=1e-4);

        self.n_steps = n_steps;
        self.n_envs = n_envs;
        self.epoch = epoch;
        self.mini_batch_per_epoch = mini_batch_per_epoch;

        self.mini_batch_size = mini_batch_size;
        self.t = 0;

    #will return the policy's action 
    def predict(self, observation, hidden_state):
        
        #no grad will simply disable gradient calculation for the following functions
        with torch.no_grad():
            #simply turn the array into a tensor mounted on the device.
            observation = torch.FloatTensor(observation).to(device = self.device);         
            hidden_state = torch.FloatTensor(hidden_state).to(device = self.device);

            #get the policy's action
            #obs is [128, 3, 64, 64]
            dist, value, hidden_state = self.policy(observation, hidden_state, None);
            act = dist.sample(); #sample from the likelyhood distributions.
            log_prob_act = dist.log_prob(act); #log the likelyhood, used later for 

        #make it numpy friendly
        act = act.cpu().numpy();
        log_prob_act = log_prob_act.cpu().numpy();
        value = value.cpu().numpy();
        hidden_state = hidden_state.cpu().numpy();

        return act, log_prob_act, value, hidden_state;

    # define optimize function
    def optimize(self):

        pi_loss_list = [];
        value_loss_list = [];
        entropy_loss_list = [];
        
        #calculate the batch size as a function of steps, number of enviroments and mini_batches
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch;
        
        #calculate how many times we should accumulate data before calling optimizer.step()
        gradient_accumulation_steps = batch_size / self.mini_batch_size;
        gradient_accumulation_cnt = 1

        #do the training
        self.policy.train();

        for _ in range(self.epoch):

            #fetch training data for each epoch, sampled random, so we don't have 2 very similar frames
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size);
            
            #itterate
            for sample in generator:
                #unpack the sample
                obs_batch, _, act_batch, done_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample;

########################################################################################

                #TODO,
                dist_batch, value_batch, _ = self.policy(obs_batch, None, None);

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.hyperPrams.eps_clip, 1.0 + self.hyperPrams.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.hyperPrams.eps_clip, self.hyperPrams.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.hyperPrams.value_coef * value_loss - self.hyperPrams.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if gradient_accumulation_cnt % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperPrams.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                gradient_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

################################################################################


        #return the losses
        return {'Loss/pi': np.mean(pi_loss_list), 'Loss/v': np.mean(value_loss_list), 'Loss/entropy': np.mean(entropy_loss_list)};


    def train(self, num_timesteps, print_stage=False):

        #save so we don't loose all our work when the HPC stops
        save_every = num_timesteps // self.num_checkpoints;
        #how many times have we saved so far
        checkpoint_cnt = 0;

        #init the hiddenstate and the based on the data in 
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size));
        done = np.zeros(self.n_envs);

        #initial observation, the values does not matter 
        obs = self.env.reset();
        
        #train for num_timesteps steps 
        while self.t < num_timesteps:           
            if(print_stage):
                print("running session");

            #evaluate the policy
            self.policy.eval();


            #do a prediction for each state
            for _ in range(self.n_steps):
                
                #get the policies prodiction
                act, log_prob_act, value, new_hidden_state = self.predict(obs, hidden_state);
                #step the enviroment forwards
                new_obs, rew, done, info = self.env.step(act);
                
                #store the "step/frame/prediction" for later use
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value);

                #the new observation becomes the old observation
                obs = new_obs;
                #The new hidden state becomes the old hidden state
                hidden_state = new_hidden_state;

            # Calculate advantage function estimates
            self.storage.compute_estimates(self.hyperPrams.gamma, self.hyperPrams.lmbda);

            #more debugging
            if(print_stage):
                print("optimizing")
            
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()

            #from the data collected, we now optimze the NN
            summary = self.optimize()

            #output data to the logger
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            self.logger.dump()

            #adjust learinging reate
            self.optimizer = adjust_lr(self.optimizer, self.hyperPrams.learning_rate, self.t, num_timesteps)

            #for debugging
            if(print_stage):
                print("stage: ", self.t, "done");
            
            # Save the policy
            if self.t > ((checkpoint_cnt + 1) * save_every):
                # open a file, to save policy
                file = open('policy.pck', 'wb');
                # dump information to that file
                pickle.dump(self.policy, file);
                # close the file
                file.close();
                
                #when we save 
                checkpoint_cnt += 1;

        #close the enviroment for good practice.
        self.env.close()
