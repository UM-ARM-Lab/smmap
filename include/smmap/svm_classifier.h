#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include "smmap/classifier.h"
#include <svm/svm.h>

namespace smmap
{
    class SVMClassifier : public Classifier
    {
    public:
        SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

        ~SVMClassifier();

    private:
        virtual double predict_impl(Eigen::VectorXd const& vec) override final;

        svm_model* model_;
        svm_node query_;
    };
}

#endif // SVM_CLASSIFIER_H
