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
                    const Eigen::VectorXd& prior_mean = Eigen::VectorXd::Zero( 1 ),
                    const Eigen::MatrixXd& prior_covar = Eigen::MatrixXd::Identity( 1, 1 ) )
                : arm_mean_( prior_mean )
                , arm_covar_( prior_covar )
            {
                assert( arm_mean_.rows() == arm_covar_.cols() );
                assert( arm_covar_.cols() == arm_covar_.rows() );
            }

            /**
             * @brief selectArmToPull Perform Thompson sampling on the bandits,
             *                        and select the bandit with the largest sample.
             * @param generator
             * @return
             */
            ssize_t selectArmToPull( Generator& generator )
            {
                // Sample from the current distribuition
                arc_helpers::MultivariteGaussianDistribution distribution(
                            arm_mean_, arm_covar_ );
                const Eigen::VectorXd sample = distribution( generator );

                // Find the arm with the highest sample
                ssize_t best_arm = -1;
                sample.maxCoeff( &best_arm );

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
                    ssize_t arm_pulled,
                    double obs_reward,
                    double obs_var )
            {
                assert( arm_pulled >= 0 && arm_pulled < arm_mean_.rows() );

                // Construct the observation matrix based on the arm pulled
                Eigen::RowVectorXd C = Eigen::RowVectorXd::Zero( arm_mean_.rows() );
                C( arm_pulled ) = 1;


                // Kalman predict
                const Eigen::VectorXd& predicted_mean = arm_mean_;                                  // No change to mean
                const Eigen::MatrixXd& predicted_covariance = arm_covar_ + transition_covariance;   // Add process noise

                // Kalman update - symbols from wikipedia article
                const double innovation = obs_reward - C * predicted_mean;                                          // tilde y_k
                const double innovation_covariance = C * predicted_covariance * C.transpose() + obs_var;            // S_k
                const Eigen::MatrixXd kalman_gain = predicted_covariance * C.transpose() / innovation_covariance;   // K_k
                arm_mean_ = predicted_mean + kalman_gain * innovation;                                              // hat x_k|k
                arm_covar_ = predicted_covariance - kalman_gain * C * predicted_covariance;                         // P_k|k
            }

        private:
            Eigen::VectorXd arm_mean_;
            Eigen::MatrixXd arm_covar_;
    };
}

#endif // KALMAN_FILTER_MULTIARM_BANDIT_HPP
