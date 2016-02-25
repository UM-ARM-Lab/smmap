#ifndef GRIPPER_COLLISION_CHECKER_H
#define GRIPPER_COLLISION_CHECKER_H

#include <ros/ros.h>

#include "smmap/collision_helpers.hpp"
#include "smmap/trajectory.hpp"

namespace smmap
{
    class GripperCollisionChecker
    {
        public:
            GripperCollisionChecker( ros::NodeHandle& nh );

            std::vector< CollisionData > gripperCollisionCheck(
                    const AllGrippersSinglePose& gripper_poses );

        private:
            ros::ServiceClient collision_checker_client_;
    };
}

#endif // GRIPPER_COLLISION_CHECKER_H
