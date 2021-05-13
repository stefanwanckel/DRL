# Hindsight Experience Replay (HER)
This is a pytorch implementation of [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) for the Universal Robot 5e

## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)

## Requirements
- python
- openai-gym
- mujoco-py
- pytorch
- mpi4py

## Instruction to run the code
Train the **ur5_reach-v1**:
```bash
mpirun -np 1 python -u train.py --env-name='FetchReach-v1' --n-cycles=10 2>&1 | tee reach.log
```

### Play Demo
```bash
python demo.py --env-name=<environment name>
```

## Results
tbd
### Demo:
ur5_reach-v1
![](https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/figures/ur5_reach-v1_ddpg-her_2021-05-07.gif)
