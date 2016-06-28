#ifndef UCB_MULTIARM_BANDIT_HPP
#define UCB_MULTIARM_BANDIT_HPP

#include <stdlib.h>
#include <assert.h>
#include <limits>
#include <cmath>
#include <vector>

namespace smmap
{
    class UCB1Normal
    {
        public:
            UCB1Normal(size_t num_bandits)
                : num_arms_(num_bandits)
                , total_reward_(num_arms_, 0.0)
                , sum_of_squared_reward_(num_arms_, 0.0)
                , num_pulls_(num_arms_, 0)
                , total_pulls_(0)
            {}

            ssize_t selectArmToPull() const
            {
                if (total_pulls_ == 0UL)
                {
                    return 0L;
                }

                const double explore_threshold = 8.0 * std::log((double)total_pulls_ + 1.0);
                double highest_ucb = -std::numeric_limits<double>::infinity();
                ssize_t best_arm = -1;

                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    if (num_pulls_[arm_ind] < explore_threshold)
                    {
                        return (ssize_t)arm_ind;
                    }
                    else
                    {
                        assert(num_pulls_[arm_ind] > 1);

                        const double average_reward = total_reward_[arm_ind] / (double)num_pulls_[arm_ind];
                        const double term1_numer = (sum_of_squared_reward_[arm_ind] - (double)num_pulls_[arm_ind] * average_reward * average_reward);
                        const double term1_denom = (double)(num_pulls_[arm_ind] - 1);
                        const double term1 = term1_numer/term1_denom;
                        const double term2 = std::log((double)total_pulls_) / (double)num_pulls_[arm_ind];
                        const double ucb = average_reward + std::sqrt(16.0 * term1 * term2);

                        assert(std::isfinite(ucb));

                        if (ucb > highest_ucb)
                        {
                            highest_ucb = ucb;
                            best_arm = (ssize_t)arm_ind;
                        }
                    }
                }

                assert(best_arm >= 0);
                return best_arm;
            }

            void updateArms(const ssize_t arm_pulled, const double reward)
            {
                total_reward_[(size_t)arm_pulled] += reward;
                sum_of_squared_reward_[(size_t)arm_pulled] += reward * reward;
                num_pulls_[(size_t)arm_pulled]++;
                total_pulls_++;
            }

        private:
            const size_t num_arms_;

            std::vector<double> total_reward_;
            std::vector<double> sum_of_squared_reward_;
            std::vector<size_t> num_pulls_;
            size_t total_pulls_;
    };
}

#endif // UCB_MULTIARM_BANDIT_HPP
