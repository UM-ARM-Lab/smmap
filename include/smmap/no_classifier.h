#ifndef NO_CLASSIFIER_H
#define NO_CLASSIFIER_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <arc_utilities/ros_helpers.hpp>

namespace smmap
{
    class NoClassifier
    {
    public:
        NoClassifier(std::shared_ptr<ros::NodeHandle> /* nh */,
                     std::shared_ptr<ros::NodeHandle> ph)
            : num_features_(ROSHelpers::GetParamRequired<int>(*ph, "classifier/dim", __func__).GetImmutable())
        {}

        std::string static Name() { return "none"; }
        int numFeatures() const { return num_features_; }
        double predict(Eigen::VectorXd const& /* vec */) const { return -1.0; }

    private:
        int const num_features_;
    };
}

#endif // NO_CLASSIFIER_H
