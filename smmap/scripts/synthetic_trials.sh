#!/bin/bash
LOG_FOLDER=/home/dmcconachie/Dropbox/catkin_ws/src/smmap/logs
rosrun smmap kalman_filter_synthetic_trials 100 1000 10    3  2 > $LOG_FOLDER/kalman_synthetic_trials_SMALL_result_100_trials_1000_pulls.txt
rosrun smmap kalman_filter_synthetic_trials 100 1000 60  147  6 > $LOG_FOLDER/kalman_synthetic_trials_MEDIUM_result_100_trials_1000_pulls.txt
rosrun smmap kalman_filter_synthetic_trials 100 1000 60 6075 12 > $LOG_FOLDER/kalman_synthetic_trials_LARGE_result_100_trials_1000_pulls.txt
