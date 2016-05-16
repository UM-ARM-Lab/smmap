#ifndef KALMAN_FILTER_MULTIARM_BANDIT_HPP
#define KALMAN_FILTER_MULTIARM_BANDIT_HPP

#include <assert.h>
#include <vector>
#include <random>
#include <utility>

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/eigen_helpers.hpp>

namespace smmap
{
    template< typename Generator = std::mt19937_64 >
    class KalmanFilterMultiarmBandit
    {
        public:
            KalmanFilterMultiarmBandit(
                    const Eigen::VectorXd& prior_mean = Eigen::VectorXd::Zero(1),
                    const Eigen::MatrixXd& prior_covar = Eigen::MatrixXd::Identity(1, 1))
                : arm_mean_(prior_mean)
                , arm_covar_(prior_covar)
            {
                assert(arm_mean_.rows() == arm_covar_.cols());
                assert(arm_covar_.cols() == arm_covar_.rows());
            }

            /**
             * @brief selectArmToPull Perform Thompson sampling on the bandits,
             *                        and select the bandit with the largest sample.
             * @param generator
             * @return
             */
            ssize_t selectArmToPull(Generator& generator)
            {
                // Sample from the current distribuition
                arc_helpers::MultivariteGaussianDistribution distribution(
                            arm_mean_, arm_covar_);
                const Eigen::VectorXd sample = distribution(generator);

                // Find the arm with the highest sample
                ssize_t best_arm = -1;
                sample.maxCoeff(&best_arm);

                return best_arm;
            }

            /**
             * @brief updateArms
             * @param transition_covariance
             * @param arm_pulled
             * @param obs_reward
             * @param obs_var
             */
            void updateArms(
                    const Eigen::MatrixXd& transition_covariance,
                    const Eigen::MatrixXd& observation_matrix,
                    const Eigen::VectorXd& observed_reward,
                    const Eigen::MatrixXd& observation_covariance)
            {
//                std::cout << "Pre Kalman Update:\n"
//                          << "arm mean\n"
//                          << arm_mean_.transpose() << std::endl
//                          << "arm_covariance\n"
//                          << arm_covar_ << std::endl
//                          << std::endl;

//                std::cout << "Kalman params:\n"
//                          << "Transition covariance\n"
//                          << transition_covariance << std::endl
//                          << "Observation_matrix\n"
//                          << observation_matrix << std::endl
//                          << "Observed reward:\n"
//                          << observed_reward.transpose() << std::endl
//                          << "observation_covariance:\n"
//                          << observation_covariance << std::endl
//                          << std::endl;

                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wconversion"
                const Eigen::MatrixXd& C = observation_matrix;

                // Kalman predict
                const Eigen::VectorXd& predicted_mean = arm_mean_;                                  // No change to mean
                const Eigen::MatrixXd& predicted_covariance = arm_covar_ + transition_covariance;   // Add process noise

                // Kalman update - symbols from wikipedia article
                const Eigen::VectorXd innovation = observed_reward - C * predicted_mean;                                            // tilde y_k

//                std::cout << "observed: " << observed_reward.transpose() << std::endl;
//                std::cout << "C*predic: " << (C * predicted_mean).transpose() << std::endl;
//                std::cout << "inovate:  " << innovation.transpose() << std::endl;

                const Eigen::MatrixXd innovation_covariance = C * predicted_covariance * C.transpose() + observation_covariance;    // S_k
                const Eigen::MatrixXd kalman_gain = predicted_covariance * C.transpose() * innovation_covariance.inverse();         // K_k


                arm_mean_ = predicted_mean + kalman_gain * innovation;                                                              // hat x_k|k
                arm_covar_ = predicted_covariance - kalman_gain * C * predicted_covariance;                                         // P_k|k
                #pragma GCC diagnostic pop

//                std::cout << "update:   " << (kalman_gain * innovation).transpose() << std::endl;
//                std::cout << "result:   " << arm_mean_.transpose() << std::endl;
//                std::cout << std::endl;


                assert(!(arm_mean_.unaryExpr([] (const double &val) { return std::isnan(val); })).any() && "NaN Found in arm_mean_ in kalman banidt!");
                assert(!(arm_mean_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "Inf Found in arm_mean_ in kalman banidt!");
                assert(!(arm_covar_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "NaN Found in arm_covar_ in kalman bandit!");
                assert(!(arm_covar_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "Inf Found in arm_covar_ in kalman bandit!");
            }

            const Eigen::VectorXd& getMean() const
            {
                return arm_mean_;
            }

            Eigen::VectorXd getMean()
            {
                return arm_mean_;
            }

            const Eigen::MatrixXd& getCovariance() const
            {
                return arm_covar_;
            }

            Eigen::MatrixXd getCovariance()
            {
                return arm_covar_;
            }

        private:
            Eigen::VectorXd arm_mean_;
            Eigen::MatrixXd arm_covar_;
    };
}

#endif // KALMAN_FILTER_MULTIARM_BANDIT_HPP
