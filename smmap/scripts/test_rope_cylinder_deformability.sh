#!/bin/bash
function calc { 
    awk "BEGIN { print "$*" }"
}

base_environment="rope_cylinder"
base_experiment="10_step"
roslaunch smmap $base_environment.launch test_id:=$base_experiment"_multi_model" multi_model:=1

for trans in `seq 10 30`;
do
   trans_deform=`calc $trans/2`
    for rot in `seq 10 30`;
    do
        rot_deform=`calc $rot/2`
        test_id="10_step_single_model/trans_"$trans_deform"_rot_"$rot_deform
        roslaunch smmap $base_environment.launch test_id:=$test_id deformability_override:=1 translational_deformability:=$trans_deform rotational_deformability:=$rot_deform multi_model:=0
    done
done
