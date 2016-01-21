#!/bin/bash

function calc { 
    awk "BEGIN { print "$*" }"
}

for trans in `seq 10 30`;
do
   trans_deform=`calc $trans/2`
    for rot in `seq 10 30`;
    do
        rot_deform=`calc $rot/2`

        test_id="rope_cylinder/trans_"$trans_deform"_rot_"$rot_deform

        roslaunch smmap rope_cylinder.launch test_id:=$test_id deformability_override:=1 translational_deformability:=$trans_deform rotational_deformability:=$rot_deform
    done
done
