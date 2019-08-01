#include "smmap/svm_classifier.h"
#include <ros/package.h>
#include <arc_utilities/ros_helpers.hpp>
#include <deformable_manipulation_experiment_params/ros_params.hpp>

using namespace smmap;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
//    std::string const basepath = ros::package::getPath("smmap");
//    std::string const expr = GetTaskTypeString(nh);
//    auto constexpr file = "/4feature.model";
//    return ROSHelpers::GetParam<std::string>(nh, "svm/model_file", basepath + "/logs/" + expr + file);
//    return "/home/dmcconac/local/src_repos/libsvm/tools/heart_scale.model";
    return ROSHelpers::GetParamRequired<std::string>(nh, "svm/model_file", __func__).GetImmutable();
}

SVMClassifier::SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                             std::shared_ptr<ros::NodeHandle> ph)
    : nh_(nh)
    , ph_(ph)
    , model_(svm_load_model(getModelFilename(*ph_).c_str()))
    , num_features_(ROSHelpers::GetParamRequired<int>(*ph_, "classifier/dim", __func__).GetImmutable())
//    , transformer_(nh, ph)
{
    assert(model_->nr_class = 2);
    assert(model_->l >= 1);
    assert(model_->nSV != nullptr);
    assert(model_->SV != nullptr);
    assert(num_features_ == model_->SV[0].dim - 1);
    assert(num_features_ > 0);
    query_.dim = num_features_ + 1;
    query_.values = (double*)(calloc(model_->SV[0].dim, sizeof(double)));

    const auto total_svs = model_->l;
    nn_raw_data_.resize(total_svs * num_features_);
    for (int idx = 0; idx < total_svs; ++idx)
    {
        memcpy(&nn_raw_data_[idx * num_features_], model_->SV[idx].values, num_features_);
    }

    flann::Matrix<double> data(nn_raw_data_.data(), total_svs, num_features_);
    nn_index_.buildIndex(data);
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

std::pair<double, Eigen::VectorXd> SVMClassifier::nearestNeighbour(const Eigen::VectorXd& vec) const
{
    Eigen::VectorXd vec_mut = vec;
    const flann::Matrix<double> query(vec_mut.data(), 1, num_features_);

    const size_t knn = 1;
    std::vector<std::vector<size_t>> indices(query.rows, std::vector<size_t>(knn, -1));
    std::vector<std::vector<double>> dists(query.rows, std::vector<double>(knn, INFINITY));

    const float eps = 0.0;
    flann::SearchParams params(flann::flann_checks_t::FLANN_CHECKS_UNLIMITED, eps);
    nn_index_.knnSearch(query, indices, dists, knn, params);

    const auto sv_idx = indices[0][0];
    const Eigen::VectorXd nn = Eigen::Map<Eigen::VectorXd>(model_->SV[sv_idx].values, num_features_);
    const double label = std::copysign(1.0, model_->sv_coef[0][sv_idx]);
    return {label, nn};
}
