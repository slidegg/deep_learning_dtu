from libs.joon.policy import CategoricalPolicy

import matplotlib.pyplot as plt
from procgen import ProcgenEnv
from libs.joon.env.procgen_wrappers import *
from pyglet.window import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import numpy as np
import pickle
import io

#from internet
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def make_me_vids():
    #contents = pickle.load(f) becomes...
    file = open('policy.pck', 'rb');
    policy = CPU_Unpickler(file).load();
    file.close();


    #will return the policy's action 
    def predict(policy, observation, hidden_state):
        
        #no grad will simply disable gradient calculation for the following functions
        with torch.no_grad():
            #simply turn the array into a tensor mounted on the device.
            observation = torch.FloatTensor(observation);
            observation = observation.permute(2,1,0);
            observation = torch.unsqueeze(observation, 0);
            hidden_state = torch.FloatTensor(hidden_state);

            #get the policy's action
            dist, value, hidden_state = policy(observation, hidden_state, None);
            act = dist.sample(); #sample from the likelyhood distributions.
            log_prob_act = dist.log_prob(act); #log the likelyhood, used later for 

        #make it numpy friendly
        act = act.cpu().numpy();
        log_prob_act = log_prob_act.cpu().numpy();
        value = value.cpu().numpy();
        hidden_state = hidden_state.cpu().numpy();

        return act, log_prob_act, value, hidden_state;


    # Make evaluation environment
    env = gym.make("procgen:procgen-starpilot-v0", use_backgrounds=False, render_mode="rgb_array", render=True, distribution_mode="easy", center_agent=True);
    env = gym.wrappers.Monitor(env, "./vids", force=True) # Create wrapper to display environment

    #env = ProcgenEnv(num_envs = 1, env_name="starpilot", distribution_mode="easy", use_backgrounds=False, render_mode = "rgb_array", paint_vel_info = True);
    #env = VecExtractDictObs(env, "rgb");
    #env = VecNormalize(env, False);
    #env = TransposeFrame(env);
    #env = ScaledFloatFrame(env);

    observation = env.reset();
    #viewer = env.get_viewer();
    #myimg = torch.zeros([64, 64, 3], dtype=torch.int32)
    #viewer.imshow(myimg);

    hidden_state = np.zeros((1, policy.embedder.output_dim));

    reward_total = 0;

    while True:

        env.render() # Render environment
        action, _, val, _ = predict(policy, observation, hidden_state);

        observation_, reward, done, info = env.step(action[0]);
        reward_total = reward_total + reward;

        observation_ = observation_.astype(np.float32);
        observation = observation_;

        render_ob = env.render();

        print(reward_total);

        if done:
            observation = env.reset();
            reward_total = 0;


if __name__=='__main__':
    make_me_vids();