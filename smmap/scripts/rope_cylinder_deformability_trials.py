#!/usr/bin/python

from run_trial import *

# Note that this is 6 to 15 as range does [start, stop)
deform_range = range(6, 15, 4)
planning_horizion = 1

for translational_deform in deform_range:
    for rotational_deform in deform_range:
        run_trial(experiment = "rope_cylinder",
                  logging_enabled = "true",
                  test_id = str(planning_horizion) + "_step_baseline/"
                        + "trans_" + str(translational_deform)
                        + "_rot_" + str(rotational_deform),
                  planning_horizion = planning_horizion,
                  multi_model = "false",
                  deformability_override = "true",
                  translational_deformability = translational_deform,
                  rotational_deformability = rotational_deform)
