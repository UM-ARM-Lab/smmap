#!/bin/bash

# roslaunch smmap generic_experiment.launch task_type:=rope_cylinder test_id:=screenshots_for_dmitry_with_optimization screenshots_enabled:=true start_bullet_viewer:=true multi_model:=true optimization_enabled:=true use_random_seed:=false static_seed_override:=true static_seed:=147466530D6092ED --screen
# roslaunch smmap generic_experiment.launch task_type:=cloth_table test_id:=screenshots_for_dmitry_with_optimization screenshots_enabled:=true start_bullet_viewer:=true multi_model:=true optimization_enabled:=true use_random_seed:=false static_seed_override:=true static_seed:=a8710913d2b5df6c --screen
roslaunch smmap generic_experiment.launch task_type:=cloth_wafr test_id:=screenshots_for_dmitry_with_optimization screenshots_enabled:=true start_bullet_viewer:=true multi_model:=true optimization_enabled:=true use_random_seed:=false static_seed_override:=true static_seed:=1475BCCC38300564 --screen
