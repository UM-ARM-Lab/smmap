#!/bin/bash

source multi_model_trial.sh

base_environment="cloth_table"
planning_horizion=1
base_experiment=$planning_horizion"_step"
multi_model_trial
single_model_trial_multiple_deform_values 10 20
planning_horizion=10
base_experiment=$planning_horizion"_step"
multi_model_trial
single_model_trial_multiple_deform_values 10 20
