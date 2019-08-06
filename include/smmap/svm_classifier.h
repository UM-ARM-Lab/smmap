#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <ros/ros.h>
#include <svm/svm.h>
#include <Eigen/Dense>

namespace smmap
{
    class SVMClassifier
    {
    public:
        SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

        ~SVMClassifier();

        std::string static Name() { return "svm"; }
        int numFeatures() const { return num_features_; }
        double predict(const Eigen::VectorXd& vec);

    private:
        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;

        svm_model* model_;
        int const num_features_;
        svm_node query_;
    };
}

#endif // SVM_CLASSIFIER_H
