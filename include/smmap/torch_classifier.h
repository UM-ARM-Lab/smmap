#ifndef TORCH_CLASSIFIER_H
#define TORCH_CLASSIFIER_H

#include "smmap/classifier.h"
#include <torch/script.h>

namespace smmap
{
    class TorchClassifier : public Classifier
    {
    public:
        TorchClassifier(std::shared_ptr<ros::NodeHandle> nh,
                        std::shared_ptr<ros::NodeHandle> ph);

    private:
        virtual double predict_impl(Eigen::VectorXd const& vec) const override final;

        torch::jit::script::Module model_;
        double const threshold_;
    };
}

#endif // TORCH_CLASSIFIER_H
