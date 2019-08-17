#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include "smmap/classifier.h"
#include <flann/flann.hpp>
#include <mutex>

namespace smmap
{
    class kNNClassifier : public Classifier
    {
    public:
        kNNClassifier(std::shared_ptr<ros::NodeHandle> nh,
                      std::shared_ptr<ros::NodeHandle> ph);

    private:
        virtual double predict_impl(const Eigen::VectorXd& vec) const override final;

        // https://www.modernescpp.com/index.php/thread-safe-initialization-of-data
        static void Initialize(kNNClassifier* knn);
        static std::once_flag init_instance_flag_;
        static std::vector<double> nn_raw_data_;
        static std::vector<double> labels_;
        static flann::KDTreeSingleIndex<flann::L2<double>> nn_index_;
    };
}

#endif // NN_CLASSIFIER_H
