#!/bin/bash
#train the ur5_reach-v1:
#mpirun -np 4 python -u train.py --env-name='ur5_reach_no_gripper-v1' 2>&1 | tee reach_no_gripper.log
#train the ur5_push-v1:
#mpirun -np 4 python -u train.py --env-name='ur5_push-v1' 2>&1 | tee push.log
#train the ur5_PickAndPlace-v1:
#mpirun -np 8 python -u train.py --env-name='ur5_PickAndPlace-v1' --n-epochs=80 2>&1 | tee pick.log
#train the FetchSlide-v1:
#mpirun -np 8 python -u train.py --env-name='ur5_slide-v1' --n-epochs=200 2>&1 | tee slide.log
now=(date +"%T")
env_name="ur5_push_no_gripper-v1"
log_extension=".log"
continue_training = false
log_name = $env_name$now$log_extension
mpirun -np 4 python -u train.py --env-name=$env_name 2>&1 | tee $log_name
if [$continue_training -eq false]
then
    echo Start training with $env_name environment.
    echo Log saving location:  $log_name
    mpirun -np 4 python -u train.py --env-name=$env_name 2>&1 | tee $log_name
else
    echo Continue training with $env_name environment.
    echo Log saving location: $log_name
    mpirun -np 4 python -u train.py --env-name=$env_name 2>&1 --continue| tee $log_name
fi