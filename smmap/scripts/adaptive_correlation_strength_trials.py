#!/usr/bin/python

from run_trial import *

for noise_factor in [0.001, 0.01, 0.1, 1]:
    run_trial(
            experiment="cloth_table",
            start_bullet_viewer='true',
            screenshots_enabled='true',
            logging_enabled='true',
            test_id='adaptive_correlation_strength_max_0.9/KFMANDB_noise_factor' + str(noise_factor),
            optimization_enabled='true',
            bandit_algorithm='KFMANDB',
            multi_model='true',
            calculate_regret='true',
            use_random_seed='false',
            max_correlation_strength_factor=0.9,
            process_noise_factor=noise_factor,
            observation_noise_factor=noise_factor)


for noise_factor in [0.001, 0.01, 0.1, 1]:
    run_trial(
            experiment="cloth_table",
            start_bullet_viewer='true',
            screenshots_enabled='true',
            logging_enabled='true',
            test_id='adaptive_correlation_strength_max_0.9/KFMANB_noise_factor' + str(noise_factor),
            optimization_enabled='true',
            bandit_algorithm='KFMANB',
            multi_model='true',
            calculate_regret='true',
            use_random_seed='false',
            max_correlation_strength_factor=0.9,
            process_noise_factor=noise_factor,
            observation_noise_factor=noise_factor)