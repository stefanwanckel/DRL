#!/bin/bash
echo starting demo...

#start python demo for ur5_reach_no_gripper-v1
#python demo.py --env-name "ur5_reach_no_gripper-v1" --project-dir "ur5_reach_raw"


#start python demo for ur5_push_no_gripper-v1
#ython demo.py --env-name "ur5_push_no_gripper-v1" --project-dir "12-12-2021_2_sharpen"
#python demo.py --env-name "ur5_push_no_gripper-v1" --project-dir "12-12-2021_2_sharpen"
#python demo.py --env-name "ur5_push_no_gripper-v1" --project-dir "06-01-2022_1_raw_cylinder"
#python demo.py --env-name "ur5_push_no_gripper-v1"


#start python demo for ur5_pick_and_place-v1
#python demo.py --env-name "ur5_pick_and_place-v1" --project-dir "13-12-2021_3_sharpen"
#python demo.py --env-name "ur5_pick_and_place-v1" --project-dir "13-12-2021_3_sharpen"
#if model is archived ( meaning it is already in a sub-folder) then use project-dir tag to specify path 


#start python demo for ur5_pick_and_place_rg2-v1
#python demo.py --env-name "ur5_pick_and_place_rg2-v1" 
python demo.py --env-name "ur5_pick_and_place_rg2-v1" --project-dir "10-01-2022_raw_0_"


