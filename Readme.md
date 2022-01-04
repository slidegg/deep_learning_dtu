## Team Jacob, Greg, Angelo

Training Procgen environment with Pytorch

Training logs for `starpilot` can be found on `logs/procgen/starpilot`.

## Requirements

- python>=3.6
- torch 1.3
- procgen
- pyyaml

## Train

Use `train.py`.

After you start training your agent, log and parameters are automatically stored in `logs/procgen/env-name/starpilot/`

## Commands

If your GPU device could handle larger memory than 5GB, increase the mini-batch size to facilitate the trianing.

`bsub < hpc_training.sh`

To run in on DTU HPC. Edit hpc_training.sh to change parameters.

## Usefull links

[1] [PPO: Human-level control through deep reinforcement learning ](https://arxiv.org/abs/1707.06347) <br>
[2] [GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation ](https://arxiv.org/abs/1506.02438) <br>
[3] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) <br>
[4] [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729) <br>
[5] [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588)<br>
[6] [List of gpus on DTU HPC](https://www.hpc.dtu.dk/?page_id=2759)

