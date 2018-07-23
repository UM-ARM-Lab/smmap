#!/usr/bin/env bash

EXPERIMENT=${1}
BASENAME="ijrr_trials/trial_"

STARTING_DIR=${PWD}
cd "/home/dmcconac/Dropbox/catkin_ws/src/smmap/logs/${EXPERIMENT}"

#for i in `seq 0 99`;
#do
#    mkdir -p ${BASENAME}${i}
#done

PARAMS="start_bullet_viewer:=false disable_all_visualizations:=true use_random_seed:=true static_seed_override:=false"

CMD=""
for i in `seq ${2} ${3}`;
do
    CMD="${CMD} mkdir -p ${BASENAME}${i} && roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}${i} --screen &> ${BASENAME}${i}/output.log && "
done
CMD="${CMD} cd ${STARTING_DIR}"
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}0 --screen &>> ${BASENAME}0/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}1 --screen &>> ${BASENAME}1/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}2 --screen &>> ${BASENAME}2/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}3 --screen &>> ${BASENAME}3/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}4 --screen &>> ${BASENAME}4/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}5 --screen &>> ${BASENAME}5/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}6 --screen &>> ${BASENAME}6/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}7 --screen &>> ${BASENAME}7/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}8 --screen &>> ${BASENAME}8/output.log && \
#roslaunch deformable_manipulation_experiment_params generic_experiment_dale.launch task_type:=${EXPERIMENT} ${PARAMS} test_id:=${BASENAME}9 --screen &>> ${BASENAME}9/output.log && \
#cd ${STARTING_DIR}

eval ${CMD}