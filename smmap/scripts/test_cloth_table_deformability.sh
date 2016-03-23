#!/bin/bash

source trial_types.sh

base_environment="cloth_table"
planning_horizion=1
base_experiment=$planning_horizion"_step_kalman_trials"
#single_model_trial_multiple_deform_values 10 18 2
multi_model_trial
