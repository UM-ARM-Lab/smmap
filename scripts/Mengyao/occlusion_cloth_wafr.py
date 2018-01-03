#!/usr/bin/python

from mengyao_run_trial import *

from numpy import arange as frange


def mengyao_run_trials(experiment, generate_screenshots="false", log_prefix=""):

        # Note that this is 0 to 25 as range does [start, stop), thus we get 0:4:24 in Matlab speak
        # deform_range = range(0, 25, 4)
        # 5:5:25
        # stretching_threshold_range = frange(0.35, 0.7, 0.05)
    #    stretching_threshold_range = [0.4]
    #    trans_dir_deformability_range = [900]
    #    trans_dis_deformability_range = [10, 3]
    #    rotation_deformability_range = [20, 5]
    #    down_scaling_range = [300]
    #    stretching_threshold = 0.4

    mengyao_run_trial(experiment=experiment,
    	start_bullet_viewer = "false",
    	test_id = log_prefix,
    	fully_observable = "true",
    	recollect_all_templates = "true",
    	collect_templates = "true")        

    mengyao_run_trial(experiment=experiment,
    	start_bullet_viewer = "true",
    	test_id = log_prefix,
    	fully_observable = "false",
    	recollect_all_templates = "false",
    	collect_templates = "false")



if __name__ == '__main__':
  mengyao_run_trials("cloth_wafr")        
