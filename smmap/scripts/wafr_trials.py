#!/usr/bin/python

from generic_trials import *

run_trials("cloth_table", run_baseline=True, run_UCB=True, run_KFMANB=True, run_KFRDB=True, generate_screenshots=True, optimization_enabled=True, log_prefix="wafr_final_submission/")
run_trials("rope_cylinder", run_baseline=True, run_UCB=True, run_KFMANB=True, run_KFRDB=True, generate_screenshots=True, optimization_enabled=True, log_prefix="wafr_final_submission/")
run_trials("cloth_wafr", run_baseline=True, run_UCB=True, run_KFMANB=True, run_KFRDB=True, generate_screenshots=True, optimization_enabled=True, log_prefix="wafr_final_submission/")
