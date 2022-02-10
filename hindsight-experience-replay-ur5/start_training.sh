#!/bin/bash

#making sure we dong overwrite log files. Increasing counter in name if log exists.
#change the index of arr_log_file to the position of the counter depending on environment name.
#log_file="ur5_push_no_gripper_1.log"
#log_file="ur5_pick_and_place_1.log"
log_file="ur5_pick_and_place_rg2-v1.log"
#log_file="ur5_reach_no_gripper-v1.log"

#echo $log_index
if test -f "$log_file"; then
    arr_log_file=(${log_file//_/ })
    log_index=${arr_log_file[4]}
    log_index=(${log_index//./ })
    log_index=${log_index[0]}
    ((log_index++))
    log_file="the_log_"$log_index".log"
fi
log_file="./logs/"$log_file
echo $log_file
#train the ur5_push_no_gripper-v1:
#mpirun -np 4 python -u train.py --env-name="ur5_push_no_gripper-v1" 2>&1 | tee $log_file
#mpirun -np 4 python -u train.py --env-name="ur5_push_no_gripper-v1" --continue-training 2>&1 | tee $log_file
#train the ur5_reach-v1:
mpirun -np 4 python -u train.py --env-name='ur5_reach_no_gripper-v1' 2>&1 | tee $log_file
#train the ur5_push-v1:
#mpirun -np 4 python -u train.py --env-name='ur5_push-v1' 2>&1 | tee push.log
#train the ur5_PickAndPlace-v1:
#mpirun -np 4 python -u train.py --env-name='ur5_pick_and_place-v1' --continue-training  2>&1 | tee $log_file
#train the FetchSlide-v1:
#mpirun -np 8 python -u train.py --env-name='ur5_slide-v1' --n-epochs=200 2>&1 | tee slide.log

#mpirun -np 4 python -u train.py --env-name='ur5_pick_and_place_rg2-v1' --continue-training 2>&1 | tee $log_file

#mpirun -np 4 python -u train.py --env-name='ur5_pick_and_place_rg2-v1' 2>&1 | tee $log_file

