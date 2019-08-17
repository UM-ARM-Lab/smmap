#include "smmap/knn_classifier.h"
#include <algorithm>
#include <fstream>
#include <arc_utilities/ros_helpers.hpp>

using namespace smmap;

std::once_flag kNNClassifier::init_instance_flag_;
std::vector<double> kNNClassifier::nn_raw_data_;
std::vector<double> kNNClassifier::labels_;
flann::KDTreeSingleIndex<flann::L2<double>> kNNClassifier::nn_index_;

inline std::string getModelFilename(ros::NodeHandle& nh)
{
    return ROSHelpers::GetParamRequired<std::string>(nh, "knn/data_file", __func__).GetImmutable();
}

kNNClassifier::kNNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                             std::shared_ptr<ros::NodeHandle> ph)
    : Classifier(nh, ph, "kNN")
{
    std::call_once(init_instance_flag_, kNNClassifier::Initialize, this);
}

void kNNClassifier::Initialize(kNNClassifier* knn)
{
    auto const filename = getModelFilename(*knn->ph_);
    auto file = std::ifstream(filename);

    auto const expected_line_count = 965110;
    nn_raw_data_.reserve(knn->num_features_ * expected_line_count);
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
        for (int i = 0; i < knn->num_features_; ++i)
        {
            double val;
            iss >> val;
            nn_raw_data_.push_back(val);
        }
    }

    flann::Matrix<double> data(nn_raw_data_.data(), labels_.size(), knn->num_features_);
    nn_index_.buildIndex(data);
}

double kNNClassifier::predict_impl(const Eigen::VectorXd& vec) const
{
    Eigen::VectorXd vec_mut = vec;
    const flann::Matrix<double> query(vec_mut.data(), 1, num_features_);

    const size_t knn = 1;
    std::vector<std::vector<size_t>> indices(query.rows, std::vector<size_t>(knn, -1));
    std::vector<std::vector<double>> dists(query.rows, std::vector<double>(knn, INFINITY));

    const float eps = 0.0;
    flann::SearchParams params(flann::flann_checks_t::FLANN_CHECKS_UNLIMITED, eps);
    nn_index_.knnSearch(query, indices, dists, knn, params);
    const auto mistake = labels_[indices[0][0]];
    return mistake;
}
