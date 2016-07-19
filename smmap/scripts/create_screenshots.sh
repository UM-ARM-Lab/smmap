#!/bin/bash

# roslaunch smmap generic_experiment.launch task_type:=rope_cylinder multi_model:=true bandit_algoritm:=KFRDB logging_enabled:=true screenshots_enabled:=true test_id:=screenshot_generation start_bullet_viewer:=true --screen

# roslaunch smmap generic_experiment.launch task_type:=cloth_table multi_model:=true bandit_algorithm:=KFRDB logging_enabled:=true screenshots_enabled:=true test_id:=screenshot_generation start_bullet_viewer:=true --screen

roslaunch smmap generic_experiment.launch task_type:=cloth_wafr multi_model:=true bandit_algorithm:=KFRDB logging_enabled:=true screenshots_enabled:=true test_id:=screenshot_generation_random_tests start_bullet_viewer:=true --screen
