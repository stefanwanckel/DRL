#!/bin/bash
echo starting demo...

#start python demo for ur5_push_no_gripper-v1
#python demo.py --env-name "ur5_push_no_gripper-v1" --project-dir "12-12-2021_2_sharpen"
#start python demo for ur5_pick_and_place-v1
#python demo.py --env-name "ur5_pick_and_place-v1"
#if model is archived ( meaning it is already in a sub-folder) then use project-dir tag to specify path 
python demo.py --env-name "ur5_pick_and_place-v1" --project-dir "13-12-2021_3_sharpen"