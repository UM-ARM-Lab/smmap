#ifndef MIN_MAX_TRANSFORMER_HPP
#define MIN_MAX_TRANSFORMER_HPP

#include <arc_utilities/ros_helpers.hpp>
#include <Eigen/Dense>

namespace smmap
{
    class MinMaxTransformer
    {
    public:
        MinMaxTransformer(std::shared_ptr<ros::NodeHandle> nh,
                          std::shared_ptr<ros::NodeHandle> ph)
        {
            (void)nh;
            const int vec_len = ROSHelpers::GetParamRequired<int>(*ph, "classifier/dim", __func__).GetImmutable();
            const auto scale = ROSHelpers::GetVectorRequired<double>(*ph, "classifier/transformer/scale", __func__).GetImmutable();
            const auto offset = ROSHelpers::GetVectorRequired<double>(*ph, "classifier/transformer/offset", __func__).GetImmutable();
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
            return x.cwiseProduct(scale_) + offset_;
        }

        Eigen::VectorXd inverse(const Eigen::VectorXd& sx) const
        {
            return (sx - offset_).cwiseQuotient(scale_);
        }

    private:
        Eigen::VectorXd scale_;
        Eigen::VectorXd offset_;
    };
}

#endif
