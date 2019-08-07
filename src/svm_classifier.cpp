#include "smmap/svm_classifier.h"
#include <stdlib.h>
#include <arc_utilities/ros_helpers.hpp>

using namespace smmap;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
    return ROSHelpers::GetParamRequired<std::string>(nh, "svm/model_file", __func__).GetImmutable();
}

SVMClassifier::SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                             std::shared_ptr<ros::NodeHandle> ph)
    : Classifier(nh, ph, "svm")
    , model_(svm_load_model(getModelFilename(*ph_).c_str()))
{
    assert(model_->nr_class = 2);
    assert(model_->l >= 1);
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

double SVMClassifier::predict_impl(Eigen::VectorXd const& vec)
{
    assert(static_cast<int>(vec.rows()) == num_features_);
    memcpy(&query_.values[1], vec.data(), num_features_ * sizeof(double));
    auto const mistake = svm_predict(model_, &query_);
    return mistake;
}
