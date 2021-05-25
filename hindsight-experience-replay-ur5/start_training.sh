#!/bin/bash
#train the ur5_reach-v1:
#  mpirun -np 1 python -u train.py --env-name='ur5_reach-v1' --n-cycles=10 2>&1 | tee reach.log
#train the ur5_push-v1:
#  mpirun -np 8 python -u train.py --env-name='ur5_push-v1' 2>&1 | tee push.log
#train the ur5_PickAndPlace-v1:
#mpirun -np 8 python -u train.py --env-name='ur5_PickAndPlace-v1' --n-epochs=80 2>&1 | tee pick.log
#train the FetchSlide-v1:
mpirun -np 8 python -u train.py --env-name='ur5_slide-v1' --n-epochs=200 2>&1 | tee slide.log