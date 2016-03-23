#ifndef KALMAN_FILTER_MULTIARM_BANDIT_HPP
#define KALMAN_FILTER_MULTIARM_BANDIT_HPP

#include <assert.h>
#include <vector>
#include <random>
#include <utility>

namespace smmap
{
    template<typename Generator>
    class KalmanFilterMultiarmBandit
    {
        public:
            KalmanFilterMultiarmBandit(
                    const std::vector < double >& prior_mean = std::vector< double >(),
                    const std::vector < double >& prior_var = std::vector< double >() )
                : arm_mean_( prior_mean )
                , arm_var_( prior_var )
            {
                assert( arm_mean_.size() == arm_var_.size() );
            }

            /**
             * @brief selectArmToPull Perform Thompson sampling on the bandits,
             *                        and select the bandit with the largest sample.
             * @param generator
             * @return
             */
            size_t selectArmToPull( Generator& generator )
            {
                assert( arm_mean_.size() == arm_var_.size() );
                assert( arm_mean_.size() > 0 );

                size_t best_arm = -1;
                double best_sample = -std::numeric_limits< double >::infinity();

                for ( size_t arm_ind = 0; arm_ind < arm_mean_.size(); arm_ind++ )
                {
                    std::normal_distribution< double > distribution(
                                arm_mean_[arm_ind], std::sqrt( arm_var_[arm_ind] ) );
                    double sample = distribution( generator );

                    if ( sample > best_sample )
                    {
                        best_sample = sample;
                        best_arm = arm_ind;
                    }
                }

                return best_arm;
            }

            /**
             * @brief updateArms
             * @param arm_pulled
             * @param reward
             * @param obs_var
             * @param tr_var
             */
            void updateArms(
                    size_t arm_pulled,
                    double reward,
                    double obs_var,
                    double tr_var )
            {
                assert( arm_pulled < arm_mean_.size() );

                for ( size_t arm_ind = 0; arm_ind < arm_mean_.size(); arm_ind++ )
                {
                    if ( arm_ind == arm_pulled )
                    {
                        arm_mean_[arm_ind] =
                                ( ( arm_var_[arm_ind] + tr_var ) * reward
                                  + obs_var * arm_mean_[arm_ind] )
                                / ( arm_var_[arm_ind] + tr_var + obs_var );

                        arm_var_[arm_ind] =
                                ( arm_var_[arm_ind] + tr_var ) * obs_var
                                / ( arm_var_[arm_ind] + tr_var + obs_var );
                    }
                    else
                    {
                        arm_var_[arm_ind] += tr_var;
                    }
                }
            }

        private:
            std::vector < double > arm_mean_;
            std::vector < double > arm_var_;
    };
}

#endif // KALMAN_FILTER_MULTIARM_BANDIT_HPP
