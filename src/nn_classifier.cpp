#include "smmap/nn_classifier.h"
#include "smmap/min_max_transformer.hpp"
#include <algorithm>
#include <fstream>
#include <deformable_manipulation_experiment_params/utility.hpp>

using namespace smmap;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
//    std::string const basepath = ros::package::getPath("smmap");
//    std::string const expr = GetTaskTypeString(nh);
//    auto constexpr file = "/4feature.model";
//    return ROSHelpers::GetParam<std::string>(nh, "svm/model_file", basepath + "/logs/" + expr + file);
    return "/mnt/big_narstie_data/dmcconac/transition_learning_data_generation/smmap_generated_plans/rope_hooks/scaled_data_13features.ssv";
//    return ROSHelpers::GetParamRequired<std::string>(nh, "nn/model_file", __func__).GetImmutable();
}

NNClassifier::NNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                           std::shared_ptr<ros::NodeHandle> ph)
    : nh_(nh)
    , ph_(ph)
    , num_features_(ROSHelpers::GetParamRequired<int>(*ph_, "classifier/dim", __func__).GetImmutable())
{
//    auto const scaler = MinMaxTransformer(nh_, ph_);
    auto const filename = getModelFilename(*ph_);
    auto file = std::ifstream(filename);

//    const auto line_count = std::count(
//                std::istream_iterator<char>(file),
//                std::istream_iterator<char>(), '\n');

//    ROS_INFO_STREAM("Reading " << line_count << " data points from " << filename);

    auto const expected_line_count = 965110;
    nn_raw_data_.reserve(num_features_ * expected_line_count);
    labels_.reserve(expected_line_count);

    std::string line;
    while (std::getline(file, line))
    {
        // Replace commas with spaces to enable us to treat everything as a space seperated file
        std::replace(line.begin(), line.end(), ',', ' ');

        std::stringstream iss(line);
        double label;
        iss >> label;
        labels_.push_back(label);
        for (int i = 0; i < num_features_; ++i)
        {
            double val;
            iss >> val;
            nn_raw_data_.push_back(val);
        }
    }

    flann::Matrix<double> data(nn_raw_data_.data(), labels_.size(), num_features_);
    nn_index_.buildIndex(data);
}

double NNClassifier::predict(const Eigen::VectorXd& vec) const
{
    Eigen::VectorXd vec_mut = vec;
    const flann::Matrix<double> query(vec_mut.data(), 1, num_features_);

    const size_t knn = 1;
    std::vector<std::vector<size_t>> indices(query.rows, std::vector<size_t>(knn, -1));
    std::vector<std::vector<double>> dists(query.rows, std::vector<double>(knn, INFINITY));

    const float eps = 0.0;
    flann::SearchParams params(flann::flann_checks_t::FLANN_CHECKS_UNLIMITED, eps);
    nn_index_.knnSearch(query, indices, dists, knn, params);

//    const auto sv_idx = indices[0][0];
//    const Eigen::VectorXd nn = Eigen::Map<Eigen::VectorXd>(model_->SV[sv_idx].values, num_features_);
//    const double label = std::copysign(1.0, model_->sv_coef[0][sv_idx]);
    return labels_[indices[0][0]];
}
