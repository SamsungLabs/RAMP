import numpy as np
import trimesh.transformations as tra

TABLE_LABEL = 0
ROBOT_LABEL = np.iinfo(np.uint8).max
START_SPHERE_LABEL = 101
GOAL_SPHERE_LABEL = 102
DEPTH_CLIP_RANGE = 3.0

ROBOT_Q_INIT = np.array(
    [
        -1.22151887,
        -1.54163973,
        -0.3665906,
        -2.23575787,
        0.5335327,
        1.04913162,
        -0.14688508,
        0.0,
        0.0,
    ]
)

ROBOT_Q_HOME = np.array(
    [
        0.0,            #Joint #1
        -0.785398,      #Joint #2
        0.0,            #Joint #3
        -1.5708,        #Joint #4
        0.0,            #Joint #5
        0.785398,       #Joint #6
        0.785398,       #Joint #7
        0.0,            #Gripper joint #1
        0.0,            #Gripper joint #2
    ]
)
