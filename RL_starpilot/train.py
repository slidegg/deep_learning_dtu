#code inspiration from https://github.com/joonleesky/train-procgen-pytorch

import os, time, yaml, argparse

from procgen import ProcgenEnv
import random
import torch
import gym
from HyperPrams import HyperPrams

#libs.joon is a copied libary
from libs.joon.logger import Logger
from libs.joon.env.procgen_wrappers import *
from libs.joon.storage import Storage
from libs.joon import set_global_seeds, set_global_log_levels
from libs.joon.policy import CategoricalPolicy
from libs.joon.model import ImpalaModel

#PPO implementation
from ppo import PPO_agent
from model import MyModel


if __name__=='__main__':

    print('parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name',        type=str, default = 'test', help='name of test')
    parser.add_argument('--env_name',         type=str, default = 'starpilot', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--game_mode',        type=str, default = 'easy', help='game mode for environment (easy/hard/whatever)')
    parser.add_argument('--param_name',       type=str, default = 'test1', help='hyper parameter ID')
    parser.add_argument('--num_timesteps',    type=int, default = int(25000000), help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--print_stage',      type=bool,default = False, help='should we print every time step')
    parser.add_argument('--num_checkpoints',  type=int, default = int(100), help='number of checkpoints to store')
    parser.add_argument('--background',       type=bool,default = False, help='Enable or Disable backgound')

    args = parser.parse_args();
    test_name = args.test_name;
    env_name = args.env_name;
    start_level = args.start_level;
    num_levels = args.num_levels;
    game_mode = args.game_mode;
    param_name = args.param_name;
    num_timesteps = args.num_timesteps;
    seed = args.seed;
    log_level = args.log_level;
    print_stage = args.print_stage;
    num_checkpoints = args.num_checkpoints;
    background = args.background;

    set_global_seeds(seed);
    set_global_log_levels(log_level);

    print('loading hyperparameters');
    with open('config.yml', 'r') as f:
        hp = yaml.safe_load(f)[param_name];
    for key, value in hp.items():
        print("\t", key, ':', value);
    n_steps = hp.get('n_steps', 256);
    n_envs = hp.get('n_envs', 0);

    hyperPrams : HyperPrams = HyperPrams(hp["gamma"], hp["lmbda"],  hp["learning_rate"],  hp["grad_clip_norm"],  hp["eps_clip"],  hp["value_coef"],  hp["entropy_coef"]);

    print('setting up device');
    use_cuda = torch.cuda.is_available();
    device = torch.device("cuda" if use_cuda else "cpu");
    print('using device:', device);
    print("cpu threads count:",torch.get_num_threads(), "\n");
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0));
        print('memory Usage:');
        print('allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB');
        print('cached:   ', torch.cuda.memory_reserved(0)/1024**3, 'GB');


    print('initializaing env');
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=game_mode,
                     use_backgrounds=background)
    env = VecExtractDictObs(env, "rgb");
    env = VecNormalize(env, False);
    env = TransposeFrame(env);
    env = ScaledFloatFrame(env);

    print('initialaziang logger');
    logdir = 'procgen/' + env_name + '/' + test_name + '/' + 'seed' + '_' + str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S");
    logdir = os.path.join('logs', logdir);
    if not (os.path.exists(logdir)):
        os.makedirs(logdir);
    logger = Logger(n_envs, logdir);

    print('initializing model');
    observation_shape = env.observation_space.shape;
    in_channels = observation_shape[0];
    action_space = env.action_space;
    action_space_size = action_space.n;
    model = ImpalaModel(in_channels);
    #model = MyModel(in_channels, action_space_size);

    print('initializing policy');
    policy = CategoricalPolicy(model, False, action_space_size);
    policy.to(device);

    print('initializing storage');
    hidden_state_dim = model.output_dim;
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device);

    print('initializing agent');
    agent = PPO_agent(env, policy, logger, storage, device, hyperPrams, num_checkpoints, n_steps, n_envs, hp["epoch"], hp["mini_batch_per_epoch"], hp["mini_batch_size"]);

    print('start traning');
    agent.train(num_timesteps, print_stage);

