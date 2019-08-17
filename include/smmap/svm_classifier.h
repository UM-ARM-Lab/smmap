#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include "smmap/classifier.h"
#include <svm/svm.h>
#include <mutex>

namespace smmap
{
    class SVMClassifier : public Classifier
    {
    public:
        SVMClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

        ~SVMClassifier();

    private:
        virtual double predict_impl(Eigen::VectorXd const& vec) const override final;

        static void Initialize(SVMClassifier* svm);
        static std::once_flag init_instance_flag_;
        static svm_model* model_;
    };
}

#endif // SVM_CLASSIFIER_H
