#!/usr/bin/python

from generic_wafr_trials import *

run_wafr_trials("cloth_table", run_baseline=False, run_UCB=False, run_KFMANB=True, run_KFRDB=True, optimization_enabled=True, log_prefix="optimization_enabled")
run_wafr_trials("rope_cylinder", run_baseline=False, run_UCB=False, run_KFMANB=True, run_KFRDB=True, optimization_enabled=True, log_prefix="optimization_enabled")
run_wafr_trials("cloth_wafr", run_baseline=False, run_UCB=False, run_KFMANB=True, run_KFRDB=True, optimization_enabled=True, log_prefix="optimization_enabled")
