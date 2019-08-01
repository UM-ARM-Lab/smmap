#ifndef NN_CLASSIFIER_H
#define NN_CLASSIFIER_H

#include <ros/ros.h>
#include <Eigen/Core>
#include <flann/flann.hpp>
#include <arc_utilities/ros_helpers.hpp>

namespace smmap
{
    class NNClassifier
    {
    public:
        NNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                     std::shared_ptr<ros::NodeHandle> ph);

        int numFeatures() const { return num_features_; }
        double predict(const Eigen::VectorXd& vec) const;

    private:
        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;

        int const num_features_;
        std::vector<double> nn_raw_data_;
        std::vector<double> labels_;
        flann::KDTreeSingleIndex<flann::L2<double>> nn_index_;
    };
}

#endif // NN_CLASSIFIER_H
