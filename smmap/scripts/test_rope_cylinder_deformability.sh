#!/bin/bash

source multi_model_trial.sh

base_environment="rope_cylinder"
planning_horizion=1
base_experiment=$planning_horizion"_step"
multi_model_trial
single_model_trial_multiple_deform_values 5 15
planning_horizion=10
base_experiment=$planning_horizion"_step"
multi_model_trial
single_model_trial_multiple_deform_values 5 15
