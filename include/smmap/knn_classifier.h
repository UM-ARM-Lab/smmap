#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include "smmap/classifier.h"
#include <flann/flann.hpp>

namespace smmap
{
    class kNNClassifier : public Classifier
    {
    public:
        kNNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

    private:
        virtual double predict_impl(const Eigen::VectorXd& vec) const override final;

        std::vector<double> nn_raw_data_;
        std::vector<double> labels_;
        flann::KDTreeSingleIndex<flann::L2<double>> nn_index_;
    };
}

#endif // NN_CLASSIFIER_H
