#include "smmap/torch_classifier.h"
#include <arc_utilities/ros_helpers.hpp>

using namespace smmap;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
    return ROSHelpers::GetParamRequired<std::string>(nh, "torch/model_file", __func__).GetImmutable();
}

TorchClassifier::TorchClassifier(
        std::shared_ptr<ros::NodeHandle> nh,
        std::shared_ptr<ros::NodeHandle> ph)
    : nh_(nh)
    , ph_(ph)
    , num_features_(ROSHelpers::GetParamRequired<int>(*ph_, "classifier/dim", __func__).GetImmutable())
    , threshold_(ROSHelpers::GetParamRequired<double>(*ph_, "torch/threshold", __func__).GetImmutable())
    , model_(torch::jit::load(getModelFilename(*ph_)))
{}

double TorchClassifier::predict(Eigen::VectorXd const& vec)
{
    auto vec_torch = torch::empty({num_features_});
    for (int idx = 0; idx < num_features_; ++idx)
    {
        vec_torch[idx] = vec[idx];
    }
    std::vector<torch::jit::IValue> const query(1, vec_torch);
    auto const output = model_.forward(query).toTensor().item().toFloat();
    auto const mistake = (output > threshold_) ? 1.0 : -1.0;
    return mistake;
}
