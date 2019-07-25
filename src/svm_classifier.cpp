#include "smmap/svm_classifier.h"
#include <ros/package.h>
#include <arc_utilities/ros_helpers.hpp>
#include <deformable_manipulation_experiment_params/ros_params.hpp>

using namespace smmap;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
    std::string const basepath = ros::package::getPath("smmap");
    std::string const expr = GetTaskTypeString(nh);
    auto constexpr file = "/4feature.model";
//    return "/home/dmcconac/local/src_repos/libsvm/tools/heart_scale.model";
    return ROSHelpers::GetParam<std::string>(nh, "svm/model_file", basepath + "/logs/" + expr + file);
}

SVMClassifier::SVMClassifier(std::shared_ptr<ros::NodeHandle>& nh,
                             std::shared_ptr<ros::NodeHandle>& ph)
    : nh_(nh)
    , ph_(ph)
    , model_(svm_load_model(getModelFilename(*nh_).c_str()))
    , num_features_(ROSHelpers::GetParamRequired<int>(*ph_, "svm/dim", __func__).GetImmutable())
//    , transformer_(nh, ph)
{
    assert(model_->nSV != nullptr);
    assert(model_->SV != nullptr);
    assert(num_features_ == model_->SV[0].dim - 1);
    assert(num_features_ > 0);
    query_.dim = num_features_ + 1;
    query_.values = (double*)(calloc(model_->SV[0].dim, sizeof(double)));
}


SVMClassifier::~SVMClassifier()
{
    svm_free_and_destroy_model(&model_);
    free(query_.values);
}

double SVMClassifier::predict(const Eigen::VectorXd& vec) const
{
    assert(static_cast<int>(vec.rows()) == num_features_);
//    const Eigen::VectorXd scaled = transformer_(vec);
    memcpy(&query_.values[1], vec.data(), num_features_);
    return svm_predict(model_, &query_);
}
