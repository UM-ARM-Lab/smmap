#ifndef NO_CLASSIFIER_H
#define NO_CLASSIFIER_H

#include "smmap/classifier.h"

namespace smmap
{
    class NoClassifier : public Classifier
    {
    public:
        NoClassifier(std::shared_ptr<ros::NodeHandle> nh,
                     std::shared_ptr<ros::NodeHandle> ph)
            : Classifier(nh, ph, "none")
        {}

    private:
        virtual double predict_impl(Eigen::VectorXd const& /* vec */) const override final
        {
            return -1.0;
        }
    };
}

#endif // NO_CLASSIFIER_H
