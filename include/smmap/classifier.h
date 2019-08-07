#ifndef SMMAP_CLASSIFIER_H
#define SMMAP_CLASSIFIER_H

#include <Eigen/Dense>
#include <ros/ros.h>
#include <arc_utilities/ros_helpers.hpp>

namespace smmap
{
    class Classifier
    {
    public:
        typedef std::shared_ptr<Classifier> Ptr;

        static Ptr MakeClassifier(std::shared_ptr<ros::NodeHandle> nh,
                                  std::shared_ptr<ros::NodeHandle> ph);

        Classifier(std::shared_ptr<ros::NodeHandle> nh,
                   std::shared_ptr<ros::NodeHandle> ph,
                   const std::string& name)
            : nh_(nh)
            , ph_(ph)
            , name_(name)
            , num_features_(ROSHelpers::GetParamRequired<int>(*ph_, "classifier/dim", __func__).GetImmutable())
        {}

        std::string name()
        {
            return name_;
        }

        int numFeatures() const
        {
            return num_features_;
        }

        double predict(Eigen::VectorXd const& vec)
        {
            return predict_impl(vec);
        }

    protected:
        ~Classifier() {}

        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;

        std::string const name_;
        int const num_features_;

    private:
        virtual double predict_impl(Eigen::VectorXd const& vec) = 0;
    };
}

#endif // SMMAP_CLASSIFIER_H
