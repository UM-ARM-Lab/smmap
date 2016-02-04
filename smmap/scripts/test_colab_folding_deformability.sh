#!/bin/bash

source multi_model_trial.sh

base_environment="colab_folding"
base_experiment="1_step"
multi_model_trial_cloth
base_experiment="10_step"
multi_model_trial_cloth
