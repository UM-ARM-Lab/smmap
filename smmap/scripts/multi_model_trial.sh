#!/bin/bash

function calc { 
    awk "BEGIN { print "$*" }"
}

function multi_model_trial
{
    roslaunch smmap $base_environment.launch test_id:=$base_experiment"_multi_model" multi_model:=1 planning_horizion:=$planning_horizion
}

function single_model_trial_multiple_deform_values
{
    min=`calc 2*$1`
    max=`calc 2*$2`
    for trans in `seq $min $max`;
    do
        trans_deform=`calc $trans/2`
        for rot in `seq $min $max`;
        do
            rot_deform=`calc $rot/2`
            test_id=$base_experiment"_single_model/trans_"$trans_deform"_rot_"$rot_deform
            echo $test_id
            roslaunch smmap $base_environment.launch test_id:=$test_id deformability_override:=1 translational_deformability:=$trans_deform rotational_deformability:=$rot_deform multi_model:=0 planning_horizion:=$planning_horizion
        done
    done
}
