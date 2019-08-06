#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <flann/flann.hpp>

namespace smmap
{
    class kNNClassifier
    {
    public:
        kNNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                     std::shared_ptr<ros::NodeHandle> ph);

        std::string static Name() { return "kNN"; }
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
