#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <ros/ros.h>
#include <svm/svm.h>
#include <Eigen/Core>
#include <arc_utilities/ros_helpers.hpp>

namespace smmap
{
    class MinMaxTransformer
    {
    public:
        MinMaxTransformer(std::shared_ptr<ros::NodeHandle>& nh,
                          std::shared_ptr<ros::NodeHandle>& ph)
        {
            (void)nh;
            const int vec_len = ROSHelpers::GetParamRequired<int>(*ph, "svm/dim", __func__).GetImmutable();
            const auto scale = ROSHelpers::GetVectorRequired<double>(*ph, "svm/transformer/scale", __func__).GetImmutable();
            const auto offset = ROSHelpers::GetVectorRequired<double>(*ph, "svm/transformer/offset", __func__).GetImmutable();
            assert(static_cast<int>(scale.size()) == vec_len);
            assert(static_cast<int>(offset.size()) == vec_len);

            scale_ = Eigen::VectorXd::Map(scale.data(), vec_len);
            offset_ = Eigen::VectorXd::Map(offset.data(), vec_len);

//            for (int idx = 0; idx < vec_len; ++idx)
//            {
//                const auto str = std::to_string(idx);
//                scale_[idx] = ROSHelpers::GetParamRequired<double>(*ph, "svm/transformer/scale_" + str, __func__).GetImmutable();
//                offset_[idx] = ROSHelpers::GetParamRequired<double>(*ph, "svm/transformer/offset_" + str, __func__).GetImmutable();
//            }
        }

        MinMaxTransformer(const Eigen::VectorXd& scale,
                          const Eigen::VectorXd& offset)
            : scale_(scale)
            , offset_(offset)
        {
            assert(scale_.rows() == offset_.rows());
        }

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            return scale_.cwiseProduct(x) + offset_;
        }

    private:
        Eigen::VectorXd scale_;
        Eigen::VectorXd offset_;
    };

    class SVMClassifier
    {
    public:
        SVMClassifier(std::shared_ptr<ros::NodeHandle>& nh,
                      std::shared_ptr<ros::NodeHandle>& ph);

        ~SVMClassifier();

        int numFeatures() const { return num_features_; }
        double predict(const Eigen::VectorXd& vec) const;

    private:
        std::shared_ptr<ros::NodeHandle> const nh_;
        std::shared_ptr<ros::NodeHandle> const ph_;

        svm_model* model_;
        int const num_features_;
        svm_node query_;
//        MinMaxTransformer const transformer_;
    };
}

#endif // SVM_CLASSIFIER_H
