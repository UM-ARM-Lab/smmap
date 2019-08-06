#ifndef TORCH_CLASSIFIER_H
#define TORCH_CLASSIFIER_H

#include <torch/script.h>
#include <Eigen/Dense>
#include <ros/ros.h>

namespace smmap
{
    class TorchClassifier
    {
    public:
        TorchClassifier(std::shared_ptr<ros::NodeHandle> nh,
                        std::shared_ptr<ros::NodeHandle> ph);

        std::string static Name() { return "dnn"; }
        int numFeatures() const { return num_features_; }
        double predict(Eigen::VectorXd const& vec);

    private:
        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;
        int const num_features_;
        double const threshold_;

        torch::jit::script::Module model_;
    };
}

#endif // TORCH_CLASSIFIER_H
