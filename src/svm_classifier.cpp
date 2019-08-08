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
}


SVMClassifier::~SVMClassifier()
{
    svm_free_and_destroy_model(&model_);
}

double SVMClassifier::predict_impl(Eigen::VectorXd const& vec) const
{
    assert(static_cast<int>(vec.rows()) == num_features_);

    svm_node query;
    query.dim = num_features_ + 1;
    query.values = (double*)(malloc(model_->SV[0].dim * sizeof(double)));
    query.values[0] = 0.0;
    memcpy(&query.values[1], vec.data(), num_features_ * sizeof(double));
    auto const mistake = svm_predict(model_, &query);
    free(query.values);
    return mistake;
}
