#!/bin/bash

source multi_model_trial.sh

base_environment="colab_folding"
planning_horizion=1
base_experiment=$planning_horizion"_step"
single_model_trial_multiple_deform_values 10 20
#multi_model_trial
#planning_horizion=10
#base_experiment=$planning_horizion"_step"
#single_model_trial_multiple_deform_values 10 20
#multi_model_trial
