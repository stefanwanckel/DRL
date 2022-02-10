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
mpirun -np 4 python -u train.py --env-name='ur5_reach-v1' 2>&1 | tee reach.log
```
Train the **ur5_reach-v1**:
```bash
mpirun -np 8 python -u train.py --env-name='ur5_push-v1' 2>&1 | tee push.log
```

### Play Demo
```bash
python demo.py --env-name=<environment name>
```

## Results
Following table contains the success rate during training.
| ur5_reach-v1 | ur5_push-v1 |
| :-------------: | :-------------: |
| insert reach graph| <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/figures/success_rate_ur5push.svg" width=600> |
| **ur5_pick-v1** | **ur5_slide-v1** |
| <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/figures/success_rate_ur5push.svg" width=600> | inser slide graph |
### Videos Simulation
| ur5_reach-v1 | ur5_push-v1 |
| :-------------: | :-------------: |
| <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/videos/ur5_reach-v1_ddpg-her_2021-05-07.gif" width=600> | <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/videos/ur5_push-v1_ddpg-her_2021-05-13.gif" width=600> |
| **ur5_pick-v1** | **ur5_slide-v1** |
|<img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/videos/ur5_pick-v1_ddpg-her_2021-05-20.gif" width=600> | insert gif of slide here |

### Videos UR5e
| | Success | Semi-Success | Failure |
| :-------------: | :-------------: | :-------------: | :-------------: |
| pick-and-place | <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/videos/vid_4.gif" width=600> | <img src="https://raw.githubusercontent.com/stefanwanckel/DRL/master/hindsight-experience-replay-ur5/Results/videos/ur5_push-v1_ddpg-her_2021-05-13.gif" width=600> | |



