#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <ros/ros.h>
#include <svm/svm.h>
#include <Eigen/Core>
#include <flann/flann.hpp>

namespace smmap
{
    class SVMClassifier
    {
    public:
        SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

        ~SVMClassifier();

        int numFeatures() const { return num_features_; }
        double predict(const Eigen::VectorXd& vec) const;
        std::pair<double, Eigen::VectorXd> nearestNeighbour(const Eigen::VectorXd& vec) const;

    private:
        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;

        svm_model* model_;
        int const num_features_;
        svm_node query_;
//        MinMaxTransformer const transformer_;

        std::vector<double> nn_raw_data_;
        flann::KDTreeSingleIndex<flann::L2<double>> nn_index_;
    };
}

#endif // SVM_CLASSIFIER_H
