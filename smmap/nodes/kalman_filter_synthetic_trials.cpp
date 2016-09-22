#include <Python.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <omp.h>
#include <cstdlib>

#include <arc_utilities/log.hpp>

//#include "smmap/cvxopt_solvers.h"
#include "smmap/gurobi_solvers.h"
#include "smmap/kalman_filter_multiarm_bandit.hpp"
#include "smmap/ucb_multiarm_bandit.hpp"
#include "smmap/timing.hpp"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;
using namespace arc_helpers;

////////////////////////////////////////////////////////////////////////////////
// Helper objects
////////////////////////////////////////////////////////////////////////////////

typedef std::map<std::string, double> TrialParams;

struct RewardObservation
{
    public:
        VectorXd true_reward;
        VectorXd observed_reward;
};

template<typename ActionType, typename ResultType>
struct ArmSuggestedAction
{
    public:
        ActionType suggested_action;
        ResultType predicted_result;
};

template<typename ActionType, typename ResultType>
std::ostream& operator<<(std::ostream& os, const ArmSuggestedAction<ActionType, ResultType>& actions)
{
    os << "Suggsted action: " << PrettyPrint::PrettyPrint(actions.suggested_action)
       << " Predicted result: " << PrettyPrint::PrettyPrint(actions.predicted_result) << std::endl;
    return os;
}


template<typename ActionResultType>
struct ActionResult
{
    public:
        double true_reward_;
        double true_regret_;
        ActionResultType action_result_;
        VectorXd all_possible_arms_true_reward_;
};


struct TrialResults
{
    public:
        TrialResults(const ssize_t num_trials)
            : num_trials_(num_trials)
            , kfrdbsimple_average_regret_(num_trials)
            , average_kfrdbsimple_avg_regret_(0)
            , variance_kfrdbsimple_avg_regret_(0)
            , kfrdbnormsim_average_regret_(num_trials)
            , average_kfrdbnormsim_avg_regret_(0)
            , variance_kfrdbnormsim_avg_regret_(0)
            , kfrdbanglesim_average_regret_(num_trials)
            , average_kfrdbanglesim_avg_regret_(0)
            , variance_kfrdbanglesim_avg_regret_(0)
            , kfmanb_average_regret_(num_trials)
            , average_kfmanb_avg_regret_(0)
            , variance_kfmanb_avg_regret_(0)
            , ucb1normal_average_regret_(num_trials)
            , average_ucb1normal_avg_regret_(0)
            , variance_ucb1normal_avg_regret_(0)
        {}

        const ssize_t num_trials_;

        Eigen::VectorXd kfrdbsimple_average_regret_;
        double average_kfrdbsimple_avg_regret_;
        double variance_kfrdbsimple_avg_regret_;

        Eigen::VectorXd kfrdbnormsim_average_regret_;
        double average_kfrdbnormsim_avg_regret_;
        double variance_kfrdbnormsim_avg_regret_;

        Eigen::VectorXd kfrdbanglesim_average_regret_;
        double average_kfrdbanglesim_avg_regret_;
        double variance_kfrdbanglesim_avg_regret_;

        Eigen::VectorXd kfmanb_average_regret_;
        double average_kfmanb_avg_regret_;
        double variance_kfmanb_avg_regret_;

        Eigen::VectorXd ucb1normal_average_regret_;
        double average_ucb1normal_avg_regret_;
        double variance_ucb1normal_avg_regret_;

        void calculateStatistics()
        {
            average_kfrdbsimple_avg_regret_ = kfrdbsimple_average_regret_.mean();
            average_kfrdbnormsim_avg_regret_ = kfrdbnormsim_average_regret_.mean();
            average_kfrdbanglesim_avg_regret_ = kfrdbanglesim_average_regret_.mean();
            average_kfmanb_avg_regret_ = kfmanb_average_regret_.mean();
            average_ucb1normal_avg_regret_ = ucb1normal_average_regret_.mean();

            variance_kfrdbsimple_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfrdbsimple_average_regret_.squaredNorm() - std::pow(average_kfrdbsimple_avg_regret_, 2);
            variance_kfrdbnormsim_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfrdbnormsim_average_regret_.squaredNorm() - std::pow(average_kfrdbnormsim_avg_regret_, 2);
            variance_kfrdbanglesim_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfrdbanglesim_average_regret_.squaredNorm() - std::pow(average_kfrdbanglesim_avg_regret_, 2);
            variance_kfmanb_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfmanb_average_regret_.squaredNorm() - std::pow(average_kfmanb_avg_regret_, 2);
            variance_ucb1normal_avg_regret_ = 1.0/(double)(num_trials_ - 1) * ucb1normal_average_regret_.squaredNorm() - std::pow(average_ucb1normal_avg_regret_, 2);
        }
};

std::ostream& operator<<(std::ostream& os, const TrialResults& tr)
{
    os << std::setw(9) << std::setprecision(6) << std::fixed;
    os << "KF-RDB (simple) average regret:    " << tr.average_kfrdbsimple_avg_regret_   << "   Std dev: " << std::sqrt(tr.variance_kfrdbsimple_avg_regret_) << std::endl
       << "KF-RDB (norm-sim) average regret:  " << tr.average_kfrdbnormsim_avg_regret_  << "   Std dev: " << std::sqrt(tr.variance_kfrdbnormsim_avg_regret_) << std::endl
       << "KF-RDB (angle-sim) average regret: " << tr.average_kfrdbanglesim_avg_regret_ << "   Std dev: " << std::sqrt(tr.variance_kfrdbanglesim_avg_regret_) << std::endl
       << "KF-MANB average regret:            " << tr.average_kfmanb_avg_regret_        << "   Std dev: " << std::sqrt(tr.variance_kfmanb_avg_regret_) << std::endl
       << "UCB1-Normal average regret:        " << tr.average_ucb1normal_avg_regret_    << "   Std dev: " << std::sqrt(tr.variance_ucb1normal_avg_regret_) << std::endl;
    os << std::setw(1) << std::setprecision(6);
    return os;
}

////////////////////////////////////////////////////////////////////////////////
// Bandits
////////////////////////////////////////////////////////////////////////////////

template <class Generator>
class MultiarmGaussianBandit
{
    public:
        MultiarmGaussianBandit(
                Generator &generator,
                const VectorXd& reward_mean,
                const MatrixXd& reward_covariance,
                const MatrixXd& transition_covariance,
                const MatrixXd& observation_covariance)
            : num_arms_(reward_mean.rows())
            , reward_mean_(reward_mean)
            , reward_covariance_(reward_covariance)
            , transition_covariance_(transition_covariance)
            , observation_covariance_(observation_covariance)
            , generator_(generator)
            , reward_distribution_(std::make_shared<MultivariteGaussianDistribution>(reward_mean_, reward_covariance_))
            , transition_distribution_(std::make_shared<MultivariteGaussianDistribution>(VectorXd::Zero(num_arms_), transition_covariance_))
            , observation_distribution_(std::make_shared<MultivariteGaussianDistribution>(VectorXd::Zero(num_arms_), observation_covariance_))
        {
            assert(reward_mean_.rows() == num_arms_);
            assert(reward_covariance_.rows() == num_arms_);
            assert(reward_covariance_.cols() == num_arms_);
            assert(transition_covariance_.rows() == num_arms_);
            assert(transition_covariance_.cols() == num_arms_);
            assert(observation_covariance_.rows() == num_arms_);
            assert(observation_covariance_.cols() == num_arms_);
        }

        ////////////////////////////////////////////////////////////////////////
        // Getters and Setters
        ////////////////////////////////////////////////////////////////////////

        // Getters and Setters: reward_mean_ ///////////////////////////////////

        const VectorXd& getRewardMean() const
        {
            return reward_mean_;
        }

        void setRewardMean(const VectorXd& reward_mean)
        {
            assert(reward_mean.rows() == num_arms_);
            reward_mean_ = reward_mean;
            reward_distribution_ = std::make_shared<MultivariteGaussianDistribution>(reward_mean_, reward_covariance_);
        }

        // Getters and Setters: reward_covariance_ /////////////////////////////

        const MatrixXd& getRewardCovariance() const
        {
            return reward_covariance_;
        }

        void setRewardCovariance(const MatrixXd& reward_covariance)
        {
            assert(reward_covariance.rows() == num_arms_);
            assert(reward_covariance.cols() == num_arms_);
            reward_covariance_ = reward_covariance;
            reward_distribution_ = std::make_shared<MultivariteGaussianDistribution>(reward_mean_, reward_covariance_);
        }

        // Getters and Setters: transition_covariance_ /////////////////////////

        const MatrixXd& getTransitionCovariance() const
        {
            return transition_covariance_;
        }

        void setTransitionCovariance(const MatrixXd& transition_covariance)
        {
            assert(transition_covariance.rows() == num_arms_);
            assert(transition_covariance.cols() == num_arms_);
            transition_covariance_ = transition_covariance;
            transition_distribution_ = std::make_shared<MultivariteGaussianDistribution>(VectorXd::Zero(), transition_covariance_);
        }

        // Getters and Setters: reward_mean_ ///////////////////////////////////

        const MatrixXd& getObservationCovariance() const
        {
            return observation_covariance_;
        }

        void setObservationCovariance(const MatrixXd& observation_covariance)
        {
            assert(observation_covariance.rows() == num_arms_);
            assert(observation_covariance.cols() == num_arms_);
            observation_covariance_ = observation_covariance;
            observation_distribution_ = MultivariteGaussianDistribution(VectorXd::Zero(), transition_covariance_);
        }

        ////////////////////////////////////////////////////////////////////////
        // Functions to be used most of the time
        ////////////////////////////////////////////////////////////////////////

        RewardObservation pullArms()
        {
            RewardObservation observation;

            observation.true_reward = (*reward_distribution_)(generator_);
            observation.observed_reward = observation.true_reward + (*observation_distribution_)(generator_);

//            MatrixXd output;
//            output.resize(2, num_arms_);
//            output << observation.true_reward.transpose(), observation.observed_reward.transpose();

//            std::cout << output << std::endl << std::endl;

            setRewardMean(reward_mean_ + (*transition_distribution_)(generator_));

            return observation;
        }

    private:
        const ssize_t num_arms_;

        VectorXd reward_mean_;
        MatrixXd reward_covariance_;
        MatrixXd transition_covariance_;
        MatrixXd observation_covariance_;

        Generator& generator_;
        std::shared_ptr<MultivariteGaussianDistribution> reward_distribution_;
        std::shared_ptr<MultivariteGaussianDistribution> transition_distribution_;
        std::shared_ptr<MultivariteGaussianDistribution> observation_distribution_;
};

/**
 * w'x = y
 */
template <class Generator>
class LinearRegressionBandit
{
    public:
        LinearRegressionBandit(
                Generator &generator,
                const size_t num_arms,
                const VectorXd& starting_weights,
                const MatrixXd& feature_covariance,
                const MatrixXd& weights_transition_covariance,
                const double observation_covariance)
            : num_arms_(num_arms)
            , true_weights_(starting_weights)
            , arm_weights_(num_arms_)
            , feature_covariance_(feature_covariance)
            , weights_transition_covariance_(weights_transition_covariance)
            , true_regression_variance_(observation_covariance)
            , generator_(generator)
            , feature_distribution_(std::make_shared<MultivariteGaussianDistribution>(VectorXd::Zero(feature_covariance_.rows()), feature_covariance_))
            , weights_transition_distribution_(std::make_shared<MultivariteGaussianDistribution>(VectorXd::Zero(weights_transition_covariance_.cols()), weights_transition_covariance_))
            , true_regression_distribution_(std::make_shared<std::normal_distribution<double>>(0.0, true_regression_variance_))
        {
            assert(feature_covariance.rows() == starting_weights.rows());
            assert(feature_covariance.rows() == feature_covariance.cols());
            assert(weights_transition_covariance.rows() == starting_weights.rows());
            assert(weights_transition_covariance.rows() == weights_transition_covariance.cols());

            // Fill the bandit arm weights in an uniformly sampled sphere
            // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8671&rep=rep1&type=pdf
            MultivariteGaussianDistribution arm_distribution(VectorXd::Zero(feature_covariance_.rows()), MatrixXd::Identity(feature_covariance_.rows(), feature_covariance_.rows()));
            for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                VectorXd weights;
                do
                {
                    weights = arm_distribution(generator);
                }
                while (weights.norm() < 1e-10);
                arm_weights_[arm_ind] = weights / weights.norm();
            }

            // Calculate how similar our bandits are to each other
            MatrixXd bandit_similarity_matrix = MatrixXd::Identity(num_arms_, num_arms_);
            for (ssize_t i = 0; i < num_arms_; i++)
            {
                for (ssize_t j = i + 1; j < num_arms_; j++)
                {
                    bandit_similarity_matrix(i, j) = arm_weights_[i].dot(arm_weights_[j]);
                    bandit_similarity_matrix(j, i) = bandit_similarity_matrix(i, j);
                }
            }

//            std::cout << "Bandit similarity matrix:\n";
//            if (num_arms_ <= 10)
//            {
//                std::cout << bandit_similarity_matrix << std::endl << std::endl;
//            }
//            else
//            {
//                std::cout << "Max similarity: " << (bandit_similarity_matrix - MatrixXd::Identity(num_arms_, num_arms_)).maxCoeff() << std::endl
//                          << "Min similarity: " << (bandit_similarity_matrix - MatrixXd::Identity(num_arms_, num_arms_)).minCoeff() << std::endl << std::endl;
//            }
        }

        VectorXd getFeatures()
        {
            return (*feature_distribution_)(generator_);
        }

        std::vector<VectorXd> getActions(const VectorXd& features) const
        {
            (void)features;
            return arm_weights_;
        }

        std::vector<double> getPredictions(const VectorXd& features) const
        {
            std::vector<double> predictions(num_arms_);
            for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                predictions[arm_ind] = arm_weights_[arm_ind].dot(features);
            }
            return predictions;
        }

        std::pair<RewardObservation, double> pullArms(const VectorXd& features)
        {
            RewardObservation reward;
            reward.true_reward.resize(num_arms_);
            reward.observed_reward.resize(num_arms_);

            const double true_regression = true_weights_.dot(features);
            const double observed_true_regression = true_regression + (*true_regression_distribution_)(generator_);

            #pragma omp parallel for
            for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                const double true_loss = std::abs(true_regression - arm_weights_[arm_ind].dot(features));
                reward.true_reward[arm_ind] = -true_loss;

                const double observed_loss = std::abs(observed_true_regression - arm_weights_[arm_ind].dot(features));
                reward.observed_reward[arm_ind] = -observed_loss;
            }

            setTrueWeights(true_weights_ + (*weights_transition_distribution_)(generator_));

            return std::make_pair(reward, observed_true_regression);
        }

        void setTrueWeights(const VectorXd& new_weights)
        {
            assert(new_weights.rows() == true_weights_.rows());
            true_weights_ = new_weights;
            true_weights_.normalize();
        }

        double getObservationCovariance() const
        {
            return true_regression_variance_;
        }

    private:
        const size_t num_arms_;
        VectorXd true_weights_;
        std::vector<VectorXd> arm_weights_;
        MatrixXd feature_covariance_;
        MatrixXd weights_transition_covariance_;
        double true_regression_variance_;

        Generator& generator_;
        std::shared_ptr<MultivariteGaussianDistribution> feature_distribution_;
        std::shared_ptr<MultivariteGaussianDistribution> weights_transition_distribution_;
        std::shared_ptr<std::normal_distribution<double>> true_regression_distribution_;
};

template <class Generator>
class JacobianBandit
{
    public:
        typedef ArmSuggestedAction<VectorXd, VectorXd> ArmSuggestedActionInner;

        JacobianBandit(
                Generator& generator,
                const size_t num_arms,
                const ssize_t num_jacobian_rows,
                const ssize_t num_jacobian_cols,
                const bool y_desesired_moves,
                const bool true_jacobian_moves)
            : num_arms_(num_arms)
            , y_desired_moves_(y_desesired_moves)
            , true_jacobian_moves_(true_jacobian_moves)
            , arm_jacobians_(num_arms_)
            , max_action_norm_(0.1)
            , generator_(generator)
        {
            y_desired_ = VectorXd::Zero(num_jacobian_rows);
            y_current_ = VectorXd::Ones(num_jacobian_rows) * 10;

            std::uniform_real_distribution<double> uniform_dist(-0.1, 0.1);

            true_jacobian_ = MatrixXd::Identity(num_jacobian_rows, num_jacobian_cols);
            for (ssize_t col_ind = 0; col_ind < num_jacobian_cols; col_ind++)
            {
                true_jacobian_(col_ind, col_ind) *= std::exp(-(double)col_ind * 2.0 / (num_jacobian_cols - 1.0));
                for (ssize_t row_ind = 0; row_ind < num_jacobian_rows; row_ind++)
                {
                    true_jacobian_(row_ind, col_ind) += uniform_dist(generator_);
                }
            }

            for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                arm_jacobians_[arm_ind].resize(num_jacobian_rows, num_jacobian_cols);
                for (ssize_t row_ind = 0; row_ind < num_jacobian_rows; row_ind++)
                {
                    for (ssize_t col_ind = 0; col_ind < num_jacobian_cols; col_ind++)
                    {
                        arm_jacobians_[arm_ind](row_ind, col_ind) = true_jacobian_(row_ind, col_ind) + 2.5 * uniform_dist(generator_);
                    }
                }

//                if (arm_ind == 0)
//                {
//                    std::cout << "First arm is the true jacobian\n";
//                    arm_jacobians_[0] = true_jacobian_;
//                }
            }

            assert(y_current_.rows() == y_desired_.rows());
            assert(y_current_.rows() == true_jacobian_.rows());
        }

        VectorXd getTargetMovement() const
        {
            return y_desired_ - y_current_;
        }

        VectorXd getArmPrediction(const size_t arm_ind, const VectorXd& action) const
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wconversion"
            return arm_jacobians_[arm_ind] * action;
            #pragma GCC diagnostic pop
        }

        VectorXd getTrueMovement(const VectorXd& action) const
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wconversion"
            return true_jacobian_ * action;
            #pragma GCC diagnostic pop
        }

        std::vector<ArmSuggestedActionInner> getArmSuggestedActions(const VectorXd& target_movement, const bool use_optimization) const
        {
            std::vector<ArmSuggestedActionInner> arm_suggestions(num_arms_);
            const Eigen::VectorXd weights = Eigen::VectorXd::Ones(target_movement.rows());

            if (use_optimization)
            {
                #pragma omp parallel for
                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    arm_suggestions[arm_ind].suggested_action =
                            minSquaredNorm(arm_jacobians_[arm_ind], target_movement, max_action_norm_, weights);
//                            CVXOptSolvers::qcqp_jacobian_least_squares(arm_jacobians_[arm_ind], weights, target_movement, max_action_norm_);

//                    std::cout << arm_suggestions[arm_ind].suggested_action.transpose() << std::endl;
//                    std::cout << minSquaredNorm(arm_jacobians_[arm_ind], target_movement, max_action_norm_).transpose() << std::endl;
//                    std::cout << std::endl;

                    arm_suggestions[arm_ind].predicted_result = getArmPrediction(arm_ind, arm_suggestions[arm_ind].suggested_action);
                }
            }
            else
            {
                #pragma omp parallel for
                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    arm_suggestions[arm_ind].suggested_action =
                            arm_jacobians_[arm_ind].jacobiSvd(ComputeThinU | ComputeThinV).solve(target_movement);
//                            WeightedLeastSquaresSolver(arm_jacobians_[arm_ind], target_movement, VectorXd::Ones(target_movement.rows()), 1e-3, 1e-2);
                    arm_suggestions[arm_ind].suggested_action = ClampNorm(arm_suggestions[arm_ind].suggested_action, max_action_norm_);
                    arm_suggestions[arm_ind].predicted_result = getArmPrediction(arm_ind, arm_suggestions[arm_ind].suggested_action);
                }
            }

            return arm_suggestions;
        }

        ArmSuggestedActionInner getTrueJacobianSuggestedAction(const VectorXd& target_movement, const bool use_optimization) const
        {
            ArmSuggestedActionInner suggestion;
            const Eigen::VectorXd weights = Eigen::VectorXd::Ones(target_movement.rows());

            if (use_optimization)
            {
                suggestion.suggested_action =
                        minSquaredNorm(true_jacobian_, target_movement, max_action_norm_, weights);
//                        CVXOptSolvers::qcqp_jacobian_least_squares(true_jacobian_, weights, target_movement, max_action_norm_);
            }
            else
            {
                suggestion.suggested_action =
                        true_jacobian_.jacobiSvd(ComputeThinU | ComputeThinV).solve(target_movement);
//                        WeightedLeastSquaresSolver(true_jacobian_, target_movement, VectorXd::Ones(target_movement.rows()), 1e-3, 1e-2);
                suggestion.suggested_action = ClampNorm(suggestion.suggested_action, max_action_norm_);
            }

            suggestion.predicted_result = true_jacobian_ * suggestion.suggested_action;

            return suggestion;
        }

        ActionResult<VectorXd> takeAction(const VectorXd& action, const std::vector<ArmSuggestedActionInner>& all_actions_to_consider)
        {
            assert(all_actions_to_consider.size() == num_arms_);

            // Apply the action and calculate the reward
            ActionResult<VectorXd> result;
            result.action_result_ = true_jacobian_ * action;
            const VectorXd y_next = y_current_ + result.action_result_;
            const double prev_error = (y_desired_ - y_current_).norm();
            const double next_error = (y_desired_ - y_next).norm();
            result.true_reward_ = prev_error - next_error;

            // Consider all the arms as possible actions, and find the best arm that could have been chosen
            double best_reward = -std::numeric_limits<double>::infinity();
            result.all_possible_arms_true_reward_.resize(num_arms_);
            for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                const VectorXd possible_y_next = y_current_ + true_jacobian_ * all_actions_to_consider[arm_ind].suggested_action;
                const double possible_next_error = (y_desired_ - possible_y_next).norm();
                const double possible_reward = prev_error - possible_next_error;
                result.all_possible_arms_true_reward_(arm_ind) = possible_reward;
                best_reward = std::max(possible_reward, best_reward);
            }
            // Calculate the true regret
            result.true_regret_ = best_reward - result.true_reward_;
            assert(result.true_regret_ >= 0);

            if (y_desired_moves_)
            {
                assert(false && "Not yet implemented");
            }
            if (true_jacobian_moves_)
            {
                assert(false && "Not yet implemented");
            }

            y_current_ = y_next;

            return result;
        }

        double takeAction(const VectorXd& action)
        {
            const double prev_error = (y_desired_ - y_current_).norm();
            y_current_ = y_current_ + true_jacobian_ * action;
            const double next_error = (y_desired_ - y_current_).norm();
            return prev_error - next_error;
        }

        const VectorXd& getYCurrent() const
        {
            return y_current_;
        }

        const VectorXd& getYDesired() const
        {
            return y_desired_;
        }

        VectorXd getBestYFinal() const
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wconversion"
            return true_jacobian_ * Pinv(true_jacobian_, SuggestedRcond()) * y_desired_;
            #pragma GCC diagnostic pop
        }

        const MatrixXd& getTrueJacobian() const
        {
            return true_jacobian_;
        }

        const MatrixXd& getArmJacobian(const size_t arm_ind) const
        {
            return arm_jacobians_.at(arm_ind);
        }

    private:
        const size_t num_arms_;
        const bool y_desired_moves_;
        const bool true_jacobian_moves_;

        VectorXd y_current_;
        VectorXd y_desired_;
        MatrixXd true_jacobian_;
        std::vector<MatrixXd> arm_jacobians_;
        const double max_action_norm_;

        Generator& generator_;
};

////////////////////////////////////////////////////////////////////////////////
// Trial functions
////////////////////////////////////////////////////////////////////////////////

/*
template <class Generator>
TrialResults IndependantGaussianBanditsTrials(Generator& generator, const TrialParams& params)
{
    const size_t num_arms   = (size_t)params.at("Number of arms:   ");
    const size_t num_trials = (size_t)params.at("Number of trials: ");
    const size_t num_pulls  = (size_t)params.at("Number of pulls:  ");

    const double initial_reward_variance_scale_factor   = params.at("Initial Reward Variance Scale Factor: ");
    const double transition_covariance_scale_factor     = params.at("Transition Covariance Scale Factor:   ");
    const double observation_covariance_scale_factor    = params.at("Observation Covariance Scale Factor:  ");

    TrialResults results(num_trials);

    for (size_t trial_ind = 0; trial_ind < num_trials; trial_ind++)
    {
        // Generate the bandit itself
        VectorXd reward_mean = VectorXd::Zero(num_arms);
        for (ssize_t bandit_ind = 0; bandit_ind < num_trials; bandit_ind++)
        {
            reward_mean(bandit_ind) = (double)(num_arms - bandit_ind) * 50.0;
        }
        const MatrixXd initial_reward_covariance = MatrixXd::Identity(num_arms, num_arms) * initial_reward_variance_scale_factor;
        const MatrixXd transition_covariance = MatrixXd::Identity(num_arms, num_arms) * transition_covariance_scale_factor;
        const MatrixXd observation_covariance = MatrixXd::Identity(num_arms, num_arms) * observation_covariance_scale_factor;
        MultiarmGaussianBandit<Generator> bandit(generator, reward_mean, initial_reward_covariance, transition_covariance, observation_covariance);

        // Create the algorithms
        KalmanFilterRDB<Generator> kfrdb_alg(VectorXd::Zero(num_arms), MatrixXd::Identity(num_arms, num_arms) * 1e100);
        double kfrdb_total_regret = 0;

        KalmanFilterMANB<Generator> kfmanb_alg(VectorXd::Zero(num_arms), VectorXd::Ones(num_arms) * 1e100);
        double kfmanb_total_regret = 0;

        UCB1Normal ucb1normal_alg(num_arms);
        double ucb1normal_total_regret = 0;

        // Pull the arms
        for (size_t pull_ind = 0; pull_ind < num_pulls; pull_ind++)
        {
            // Determine which arm each algorithm pulls
            const ssize_t kfrdb_arm_pulled = kfrdb_alg.selectArmToPull(generator);
            const ssize_t kfmanb_arm_pulled = kfmanb_alg.selectArmToPull(generator);
            const ssize_t ucb1normal_arm_pulled = ucb1normal_alg.selectArmToPull();

            // Pull all the arms to determine true and observed rewards
            const double best_expected_reward = bandit.getRewardMean().maxCoeff();
            const auto rewards = bandit.pullArms();

            // Give rewards for each algorithm
            kfrdb_total_regret += best_expected_reward - rewards.true_reward(kfrdb_arm_pulled);
            kfmanb_total_regret += best_expected_reward - rewards.true_reward(kfmanb_arm_pulled);
            ucb1normal_total_regret += best_expected_reward - rewards.true_reward(ucb1normal_arm_pulled);

            // Update each algorithm
            const MatrixXd kfrdb_observation_matrix = MatrixXd::Identity(num_arms, num_arms);
            const VectorXd kfrdb_observed_reward = rewards.observed_reward;
            kfrdb_alg.updateArms(bandit.getTransitionCovariance(), kfrdb_observation_matrix, kfrdb_observed_reward, bandit.getObservationCovariance());

            const VectorXd kfmanb_transition_variance = bandit.getTransitionCovariance().diagonal();
            const double kfmanb_observed_reward = rewards.observed_reward(kfmanb_arm_pulled);
            const double kfmanb_observation_variance = bandit.getObservationCovariance()(kfmanb_arm_pulled, kfmanb_arm_pulled);
            kfmanb_alg.updateArms(kfmanb_transition_variance, kfmanb_arm_pulled, kfmanb_observed_reward, kfmanb_observation_variance);

            ucb1normal_alg.updateArms(ucb1normal_arm_pulled, rewards.observed_reward(ucb1normal_arm_pulled));
        }

        // Record the results
        results.kfrdbsimple_average_regret_(trial_ind) = kfrdb_total_regret / (double)num_pulls;
        results.kfmanb_average_regret_(trial_ind) = kfmanb_total_regret / (double)num_pulls;
        results.ucb1normal_average_regret_(trial_ind) = ucb1normal_total_regret / (double)num_pulls;

        std::cout << "Trial Num: " << trial_ind
                  << " KF-RDB: " << results.kfrdbsimple_average_regret_(trial_ind)
                  << " KF-MANB: " << results.kfmanb_average_regret_(trial_ind)
                  << " UCB1-Normal: " << results.ucb1normal_average_regret_(trial_ind)
                  << std::endl;
    }

    results.calculateStatistics();
    return results;
}

template <class Generator>
TrialResults LinearRegressionBanditsTrials(Generator& generator, const TrialParams& params)
{
    const size_t num_arms   = (size_t)params.at("Number of arms:   ");
    const size_t num_trials = (size_t)params.at("Number of trials: ");
    const size_t num_pulls  = (size_t)params.at("Number of pulls:  ");

    const ssize_t feature_vector_length                 = (ssize_t)params.at("Feature Vector Length: ");
    const double feature_covariance_scale_factor        = params.at("Feature Covariance Scale Factor:      ");
    const double transition_covariance_scale_factor     = params.at("Transition Covariance Scale Factor:   ");
    const double observation_covariance_scale_factor    = params.at("Observation Covariance Scale Factor:  ");

    TrialResults results(num_trials);

    for (size_t trial_ind = 0; trial_ind < num_trials; trial_ind++)
    {
        // Create the bandit itself
        VectorXd starting_weights(feature_vector_length);
        starting_weights(0) = 1.0;
        const MatrixXd feature_covariance = MatrixXd::Identity(feature_vector_length, feature_vector_length) * feature_covariance_scale_factor;
        const MatrixXd weights_transition_covariance = MatrixXd::Identity(feature_vector_length, feature_vector_length) * transition_covariance_scale_factor;
        const double observation_covariance = observation_covariance_scale_factor;
        LinearRegressionBandit<Generator> bandit(generator, num_arms, starting_weights, feature_covariance, weights_transition_covariance, observation_covariance);

        // Create the algorithms
        KalmanFilterRDB<Generator> kfrdb_alg(VectorXd::Zero(num_arms), MatrixXd::Identity(num_arms, num_arms) * 1e100);
        double kfrdb_total_regret = 0;
        double kfrdb_current_reward_std_dev_scale = 1.0;

        KalmanFilterMANB <Generator>kfmanb_alg(VectorXd::Zero(num_arms), VectorXd::Ones(num_arms) * 1e100);
        double kfmanb_total_regret = 0;

        UCB1Normal ucb1normal_alg(num_arms);
        double ucb1normal_total_regret = 0;

        // Pull the arms
        for (size_t pull_ind = 0; pull_ind < num_pulls; pull_ind++)
        {
            // Determine which arm each algorithm pulls
            const ssize_t kfrdb_arm_pulled = kfrdb_alg.selectArmToPull(generator);
            const ssize_t kfmanb_arm_pulled = kfmanb_alg.selectArmToPull(generator);
            const ssize_t ucb1normal_arm_pulled = ucb1normal_alg.selectArmToPull();

            // Pull all the arms to determine true and observed rewards
            const VectorXd features = bandit.getFeatures();
            std::vector<VectorXd> actions = bandit.getActions(features);
            std::vector<double> predictions = bandit.getPredictions(features);
            const std::pair<RewardObservation, double> arm_results = bandit.pullArms(features);
            const RewardObservation& rewards = arm_results.first;
            const double& observed_regression = arm_results.second;
            const double best_expected_reward = 0;

            // Give rewards for each algorithm
            kfrdb_total_regret += best_expected_reward - rewards.true_reward(kfrdb_arm_pulled);
            kfmanb_total_regret += best_expected_reward - rewards.true_reward(kfmanb_arm_pulled);
            ucb1normal_total_regret += best_expected_reward - rewards.true_reward(ucb1normal_arm_pulled);

            // Update each algorithm

            // KFRDB - process noise
            MatrixXd kfrdb_process_noise = MatrixXd::Identity(num_arms, num_arms);
            for (ssize_t i = 0; i < num_arms; i++)
            {
                for (ssize_t j = i + 1; j < num_arms; j++)
                {
                    kfrdb_process_noise(i, j) = actions[i].dot(actions[j]) / (actions[i].norm() * actions[j].norm());
                    kfrdb_process_noise(j, i) = kfrdb_process_noise(i, j);
                }
            }

            // KFRDB - observation matrix
            MatrixXd kfrdb_observation_matrix = MatrixXd::Identity(num_arms, num_arms);

            // KFRDB - observed reward
            VectorXd kfrdb_observed_reward = VectorXd::Zero(num_arms);
            const double equivalent_distance = std::abs(observed_regression - predictions[(size_t)kfrdb_arm_pulled]);
            kfrdb_current_reward_std_dev_scale = 0.9 * kfrdb_current_reward_std_dev_scale + 0.1 * equivalent_distance;
            for (ssize_t arm_ind = 0; arm_ind < num_arms; arm_ind++)
            {
                const double current_arm_distance = std::abs(observed_regression - predictions[(size_t)arm_ind]);
                kfrdb_observed_reward(arm_ind) = rewards.observed_reward(kfrdb_arm_pulled)
                        + (equivalent_distance - current_arm_distance) * std::pow(kfrdb_current_reward_std_dev_scale, 2);
            }

//            MatrixXd formatted_output;
//            formatted_output.resize(2, rewards.true_reward.rows());
//            formatted_output << rewards.true_reward.transpose(), kfrdb_observed_reward.transpose();

//            std::cout << " True Reward: " << rewards.true_reward(kfrdb_arm_pulled)
//                      << " Obs Reward: " << rewards.observed_reward(kfrdb_arm_pulled)
//                      << std::endl;
//            std::cout << formatted_output << std::endl << std::endl;

            // KFRDB - observation noise
            MatrixXd kfrdb_observation_noise = MatrixXd::Identity(num_arms, num_arms);
            for (ssize_t i = 0; i < num_arms; i++)
            {
                kfrdb_observation_noise(i, i) = std::exp(-kfrdb_process_noise(i, kfrdb_arm_pulled));
            }
            for (ssize_t i = 0; i < num_arms; i++)
            {
                for (ssize_t j = i + 1; j < num_arms; j++)
                {
                    kfrdb_observation_noise(i, j) = kfrdb_process_noise(i, j) * std::sqrt(kfrdb_observation_noise(i, i)) * std::sqrt(kfrdb_observation_noise(j, j));
                    kfrdb_observation_noise(j, i) = kfrdb_process_noise(i, j);
                }
            }

            kfrdb_alg.updateArms(kfrdb_process_noise, kfrdb_observation_matrix, kfrdb_observed_reward, kfrdb_observation_noise);

            const VectorXd kfmanb_transition_variance = VectorXd::Ones(num_arms);
            const double kfmanb_observed_reward = rewards.observed_reward(kfmanb_arm_pulled);
            const double kfmanb_observation_variance = bandit.getObservationCovariance();
            kfmanb_alg.updateArms(kfmanb_transition_variance, kfmanb_arm_pulled, kfmanb_observed_reward, kfmanb_observation_variance);

            ucb1normal_alg.updateArms(ucb1normal_arm_pulled, rewards.observed_reward(ucb1normal_arm_pulled));
        }

        // Record the results
        results.kfrdbsimple_average_regret_(trial_ind) = kfrdb_total_regret / (double)num_pulls;
        results.kfmanb_average_regret_(trial_ind) = kfmanb_total_regret / (double)num_pulls;
        results.ucb1normal_average_regret_(trial_ind) = ucb1normal_total_regret / (double)num_pulls;

        std::cout << "Trial Num: " << trial_ind;
        std::cout << std::setw(9) << std::setprecision(6) << std::fixed;
        std::cout << " KF-RDB: " << results.kfrdbsimple_average_regret_(trial_ind)
                  << " KF-MANB: " << results.kfmanb_average_regret_(trial_ind)
                  << " UCB1-Normal: " << results.ucb1normal_average_regret_(trial_ind);
        std::cout << std::setw(1) << std::setprecision(6)
                  << std::endl;
    }

    results.calculateStatistics();
    return results;
}
*/

#define LOG_DATA logData(trial_log, bandit.getYDesired() - target_movement, target_movement.norm(), arm_suggested_actions[arm_to_pull].suggested_action, pull_result.true_reward_, pull_result.true_regret_)

class JacobianTrackingTrials
{
    public:
        static TrialResults run(std::mt19937_64& generator, const TrialParams& params)
        {
            num_arms_   = (ssize_t)params.at("Number of arms:   ");
            num_trials_ = (ssize_t)params.at("Number of trials: ");
            num_pulls_  = (ssize_t)params.at("Number of pulls:  ");
            num_jacobian_rows_ = (ssize_t)params.at("Num Jacobian rows: ");
            num_jacobian_cols_ = (ssize_t)params.at("Num Jacobian cols: ");

            use_optimization_ = params.at("Optimize:          ") > 0.5 ? true : false;
            min_reward_scale_ = 1e-10;

            log_folder_ = std::string("/tmp/kalman_regret_trials/")
                    + "num_arms_" + std::to_string(num_arms_)
                    + "__num_trials_" + std::to_string(num_trials_)
                    + "__num_pulls_" + std::to_string(num_pulls_)
                    + "__num_rows_" + std::to_string(num_jacobian_rows_)
                    + "__num_cols_" + std::to_string(num_jacobian_cols_)
                    + "__optimization_" + std::to_string(use_optimization_)
                    + "/";

            Log::Log aggregate_log(log_folder_ + "aggregate_log.txt", true);

            TrialResults results(num_trials_);

            for (trial_ind_ = 0; trial_ind_ < num_trials_; trial_ind_++)
            {
                // Advance the generator some arbitrary number of draws to get a new number stream
                generator.discard(num_pulls_ * (num_arms_ + 1) + 1000);

                logJacobians(generator);
                trueJacobianTrial(generator);
                results.kfrdbsimple_average_regret_(trial_ind_)      = kfrdbSimpleTrial(generator);
                results.kfrdbnormsim_average_regret_(trial_ind_)     = kfrdbNormSimTrial(generator);
                results.kfrdbanglesim_average_regret_(trial_ind_)    = kfrdbAngleSimTrial(generator);
                results.kfmanb_average_regret_(trial_ind_)           = kfmanbTrial(generator);
                results.ucb1normal_average_regret_(trial_ind_)       = ucb1NormalTrial(generator);

                std::stringstream ss;
                ss << "Trial Ind: " << trial_ind_;
                ss << std::setw(9) << std::setprecision(6) << std::fixed;
                ss << "   KF-RDB (simple): " << results.kfrdbsimple_average_regret_(trial_ind_)
                   << "   KF-RDB (norm sim): " << results.kfrdbnormsim_average_regret_(trial_ind_)
                   << "   KF-RDB (angle sim): " << results.kfrdbanglesim_average_regret_(trial_ind_)
                   << "   KF-MANB: " << results.kfmanb_average_regret_(trial_ind_)
                   << "   UCB1-Normal: " << results.ucb1normal_average_regret_(trial_ind_);

                std::cout << ss.str() << std::endl;
                LOG(aggregate_log, ss.str());
            }

            results.calculateStatistics();

            LOG(aggregate_log, results);
            return results;
        }

    private:
        JacobianTrackingTrials();

        static double rewardScaleEstimator(const double prev_reward_scale, const double current_reward)
        {
            return std::max(min_reward_scale_, 0.9 * prev_reward_scale + 0.1 * std::abs(current_reward));
        }

        static void trueJacobianTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "true_jacobian_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            JacobianBandit<std::mt19937_64> bandit(generator, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                const VectorXd target_movement = bandit.getTargetMovement();
                std::vector<JacobianBandit<std::mt19937_64>::ArmSuggestedActionInner> arm_suggested_actions(1);
                arm_suggested_actions[0] = bandit.getTrueJacobianSuggestedAction(target_movement, use_optimization_);
                const size_t arm_to_pull = 0;
                ActionResult<VectorXd> pull_result;
                pull_result.true_reward_ = bandit.takeAction(arm_suggested_actions[0].suggested_action);
                pull_result.true_regret_ = 0;

                LOG_DATA;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);
        }

        static double kfrdbSimpleTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "kfrdbsimple_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            std::mt19937_64 generator_copy = generator;
            JacobianBandit<std::mt19937_64> bandit(generator_copy, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            KalmanFilterRDB<std::mt19937_64> kfrdb_alg(VectorXd::Zero(num_arms_), MatrixXd::Identity(num_arms_, num_arms_) * 1e6);
            double total_regret = 0;
            double estimated_reward_scale = min_reward_scale_;

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                // Pull an arm, collect data
                const VectorXd target_movement = bandit.getTargetMovement();
                const auto arm_suggested_actions = bandit.getArmSuggestedActions(target_movement, use_optimization_);
                const size_t arm_to_pull = kfrdb_alg.selectArmToPull(generator_copy);
                const auto pull_result = bandit.takeAction(arm_suggested_actions[arm_to_pull].suggested_action, arm_suggested_actions);
                estimated_reward_scale = rewardScaleEstimator(estimated_reward_scale, pull_result.true_reward_);

                LOG_DATA;

                // Action norms - calced here to avoid calculating repeatedly
                std::vector<double> action_norms((size_t)num_arms_);
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    action_norms[arm_ind] = arm_suggested_actions[arm_ind].suggested_action.norm();
                }

                // KFRDB - Process noise
                MatrixXd process_noise = 1.1 * MatrixXd::Identity(num_arms_, num_arms_);
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    for (ssize_t j = i + 1; j < num_arms_; j++)
                    {
                        double action_similarity;
                        if (action_norms[(size_t)i] != 0 && action_norms[(size_t)j] != 0)
                        {
                            const VectorXd& action_i = arm_suggested_actions[i].suggested_action;
                            const VectorXd& action_j = arm_suggested_actions[j].suggested_action;
                            action_similarity = action_i.dot(action_j) / (action_norms[(size_t)i] * action_norms[(size_t)j]);
                        }
                        else if (action_norms[(size_t)i] == 0 && action_norms[(size_t)j] == 0)
                        {
                            action_similarity = 1;
                        }
                        else
                        {
                            action_similarity = 0;
                        }

                        process_noise(i, j) = action_similarity;
                        process_noise(j, i) = action_similarity;
                    }
                }
                process_noise *= std::pow(estimated_reward_scale, 2);

                // KFRDB - Observation matrix
                MatrixXd observation_matrix = RowVectorXd::Zero(num_arms_);
                observation_matrix(0, arm_to_pull) = 1;

                // KFRDB - Observed Reward
                VectorXd observed_reward = VectorXd::Ones(1) * pull_result.true_reward_;

                // KFRDB - Observation Noise
                MatrixXd observation_noise = MatrixXd::Zero(1, 1);
                observation_noise *= std::pow(estimated_reward_scale, 2);

                kfrdb_alg.updateArms(process_noise, observation_matrix, observed_reward, observation_noise);
                total_regret += pull_result.true_regret_;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);

            return total_regret / (double)num_pulls_;
        }

        static double kfrdbNormSimTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "kfrdbnormsim_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            std::mt19937_64 generator_copy = generator;
            JacobianBandit<std::mt19937_64> bandit(generator_copy, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            KalmanFilterRDB<std::mt19937_64> kfrdb_alg(VectorXd::Zero(num_arms_), MatrixXd::Identity(num_arms_, num_arms_) * 1e6);
            double total_regret = 0;
            double estimated_reward_scale = min_reward_scale_;

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                const VectorXd target_movement = bandit.getTargetMovement();
                const auto arm_suggested_actions = bandit.getArmSuggestedActions(target_movement, use_optimization_);
                const size_t arm_to_pull = kfrdb_alg.selectArmToPull(generator_copy);
                const auto pull_result = bandit.takeAction(arm_suggested_actions[arm_to_pull].suggested_action, arm_suggested_actions);
                estimated_reward_scale = rewardScaleEstimator(estimated_reward_scale, pull_result.true_reward_);

                LOG_DATA;

                // Action norms - calced here to avoid calculating repeatedly
                std::vector<double> action_norms((size_t)num_arms_);
                for (size_t arm_ind = 0; arm_ind < (size_t)num_arms_; arm_ind++)
                {
                    action_norms[arm_ind] = arm_suggested_actions[arm_ind].suggested_action.norm();
                }

                // KFRDB - Process noise
                MatrixXd process_noise = 1.1 * MatrixXd::Identity(num_arms_, num_arms_);
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    for (ssize_t j = i + 1; j < num_arms_; j++)
                    {
                        double action_similarity;
                        if (action_norms[(size_t)i] != 0 && action_norms[(size_t)j] != 0)
                        {
                            const VectorXd& action_i = arm_suggested_actions[i].suggested_action;
                            const VectorXd& action_j = arm_suggested_actions[j].suggested_action;
                            action_similarity = action_i.dot(action_j) / (action_norms[(size_t)i] * action_norms[(size_t)j]);
                        }
                        else if (action_norms[(size_t)i] == 0 && action_norms[(size_t)j] == 0)
                        {
                            action_similarity = 1;
                        }
                        else
                        {
                            action_similarity = 0;
                        }

                        process_noise(i, j) = action_similarity;
                        process_noise(j, i) = action_similarity;
                    }
                }
                process_noise *= std::pow(estimated_reward_scale, 2);

                // KFRDB - observation matrix
                MatrixXd observation_matrix = MatrixXd::Identity(num_arms_, num_arms_);


                //                std::cout << "\n\n\n\n\n";
                // KFRDB - observed reward
                VectorXd norm_between_true_movement_and_predicted_movement(num_arms_);
                const VectorXd& true_movement = pull_result.action_result_;
                //                std::cout << "True movement:            " << true_movement.transpose() << std::endl;
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const VectorXd predicted_movement = bandit.getArmPrediction(arm_ind, arm_suggested_actions[arm_to_pull].suggested_action);
                    //                    std::cout << "Arm " << arm_ind << " predicted movement: " << predicted_movement.transpose() << std::endl;
                    norm_between_true_movement_and_predicted_movement(arm_ind) = (true_movement - predicted_movement).norm();
                }

                //                for (size_t arm_ind = 0; arm_ind < num_arms; arm_ind++)
                //                {
                //                    #pragma GCC diagnostic push
                //                    #pragma GCC diagnostic ignored "-Wconversion"
                //                    const VectorXd current_arm_true_movement_if_suggsted_action_is_taken = bandit.getTrueMovement(arm_suggested_actions[arm_ind].suggested_action);
                //                    const VectorXd current_arm_predicted_movement_if_suggsted_action_is_taken = bandit.getArmPrediction(arm_ind, arm_suggested_actions[arm_ind].suggested_action);
                //                    #pragma GCC diagnostic pop

                //                    std::cout << "Arm ind: " << arm_ind << std::endl
                //                              << "Suggsted action:                           " << arm_suggested_actions[arm_ind].suggested_action.transpose()            << " Norm: " << arm_suggested_actions[arm_ind].suggested_action.norm() << std::endl
                //                              << "True movement under suggested action:      " << current_arm_true_movement_if_suggsted_action_is_taken.transpose()      << " Norm: " << current_arm_true_movement_if_suggsted_action_is_taken.norm() <<  std::endl
                //                              << "Predicted movement under suggested action: " << current_arm_predicted_movement_if_suggsted_action_is_taken.transpose() << " Norm: " << current_arm_predicted_movement_if_suggsted_action_is_taken.norm() << std::endl
                //                              << std::endl;
                //                }


                //                std::cout << "\n\n\n\n\n";


                bool print_data = false;
                std::vector<ssize_t> bad_inds;
                VectorXd estimated_reward_improvements = VectorXd::Zero(num_arms_);
                const VectorXd true_reward_improvements = pull_result.all_possible_arms_true_reward_.array() - pull_result.all_possible_arms_true_reward_(arm_to_pull);

                VectorXd observed_reward = VectorXd::Zero(num_arms_);
                VectorXd norm_improvements = VectorXd::Zero(num_arms_);
                const double norm_to_arm_chosen = norm_between_true_movement_and_predicted_movement(arm_to_pull);
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const double norm_improvement = norm_to_arm_chosen - norm_between_true_movement_and_predicted_movement(arm_ind);
                    norm_improvements(arm_ind) = norm_improvement;
                    estimated_reward_improvements(arm_ind) = 1.0 * norm_improvement * std::abs(pull_result.true_reward_);
                    observed_reward(arm_ind) = pull_result.true_reward_ + estimated_reward_improvements(arm_ind);

                    if (estimated_reward_improvements(arm_ind) * true_reward_improvements(arm_ind) < 0)
                    {
                        print_data = true;
                        bad_inds.push_back(arm_ind);
                    }
                }

                // KFRDB - observation noise
                MatrixXd observation_noise = 1.1 * MatrixXd::Identity(num_arms_, num_arms_);
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    observation_noise(i, i) = std::exp(-process_noise(i, arm_to_pull));
                }
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    for (ssize_t j = i + 1; j < num_arms_; j++)
                    {
                        const double observation_covariance = process_noise(i, j) * std::sqrt(observation_noise(i, i)) * std::sqrt(observation_noise(j, j));
                        observation_noise(i, j) = observation_covariance;
                        observation_noise(j, i) = observation_covariance;
                    }
                }
                observation_noise *= std::pow(estimated_reward_scale, 2);
                observation_noise = process_noise;

                print_data = false;
                if (print_data)
                {
                    std::cout << "y_prev         : " << (bandit.getYCurrent() - pull_result.action_result_).transpose() << std::endl;
                    std::cout << "y_desired      : " << bandit.getYDesired().transpose() << std::endl;
                    std::cout << "desired y_delta: " << target_movement.transpose() << std::endl;
                    std::cout << "true action result norm: " << pull_result.action_result_.norm() << std::endl;
                    std::cout << "bad inds: " << PrettyPrint::PrettyPrint(bad_inds) << std::endl;
                    std::cout << " true rewrd |"
                              << " est. rewrd |"
                              << " norm t vs p|"
                              << " norm-imp.  |"
                              << " true imp.  |"
                              << " est. imp.  |"
                                 //                              << " predicted action norm |"
                              << "\n";

                    MatrixXd formatted_output;
                    formatted_output.resize(num_arms_, 6);

                    formatted_output.col(0) = pull_result.all_possible_arms_true_reward_;
                    formatted_output.col(1) = observed_reward;
                    formatted_output.col(2) = norm_between_true_movement_and_predicted_movement;
                    formatted_output.col(3) = norm_improvements;
                    formatted_output.col(4) = true_reward_improvements;
                    formatted_output.col(5) = estimated_reward_improvements;

                    std::cout << formatted_output << std::endl << std::endl;

                    //                    std::cout << "Process Noise (no scale factor):\n"
                    //                              << process_noise
                    //                              << std::endl << std::endl;
                    //                    std::cout << "Observation Noise (no scale factor):\n"
                    //                              << observation_noise
                    //                              << std::endl << std::endl;

                    if (pull_ind == 2)
                        exit(-1);
                }


                kfrdb_alg.updateArms(process_noise, observation_matrix, observed_reward, observation_noise);
                total_regret += pull_result.true_regret_;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);

            return total_regret / (double)num_pulls_;
        }

        static double kfrdbAngleSimTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "kfrdbanglesim_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            JacobianBandit<std::mt19937_64> bandit(generator, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            KalmanFilterRDB<std::mt19937_64> kfrdb_alg(VectorXd::Zero(num_arms_), MatrixXd::Identity(num_arms_, num_arms_) * 1e6);
            double total_regret = 0;
            double estimated_reward_scale = min_reward_scale_;

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                // Pull an arm, collect data
                const VectorXd target_movement = bandit.getTargetMovement();
                const auto arm_suggested_actions = bandit.getArmSuggestedActions(target_movement, use_optimization_);
                const size_t arm_to_pull = kfrdb_alg.selectArmToPull(generator);
                const auto pull_result = bandit.takeAction(arm_suggested_actions[arm_to_pull].suggested_action, arm_suggested_actions);
                estimated_reward_scale = rewardScaleEstimator(estimated_reward_scale, pull_result.true_reward_);

                LOG_DATA;

                // Action norms - calced here to avoid calculating repeatedly
                std::vector<double> action_norms((size_t)num_arms_);
                for (size_t arm_ind = 0; arm_ind < (size_t)num_arms_; arm_ind++)
                {
                    action_norms[arm_ind] = arm_suggested_actions[arm_ind].suggested_action.norm();
                }

                // KFRDB - Process noise
                MatrixXd process_noise = 1.1 * MatrixXd::Identity(num_arms_, num_arms_);
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    for (ssize_t j = i + 1; j < num_arms_; j++)
                    {
                        double action_similarity;
                        if (action_norms[(size_t)i] != 0 && action_norms[(size_t)j] != 0)
                        {
                            const VectorXd& action_i = arm_suggested_actions[i].suggested_action;
                            const VectorXd& action_j = arm_suggested_actions[j].suggested_action;
                            action_similarity = action_i.dot(action_j) / (action_norms[(size_t)i] * action_norms[(size_t)j]);
                        }
                        else if (action_norms[(size_t)i] == 0 && action_norms[(size_t)j] == 0)
                        {
                            action_similarity = 1;
                        }
                        else
                        {
                            action_similarity = 0;
                        }

                        process_noise(i, j) = action_similarity;
                        process_noise(j, i) = action_similarity;
                    }
                }
                process_noise *= std::pow(estimated_reward_scale, 2);

                // KFRDB - observation matrix
                MatrixXd observation_matrix = MatrixXd::Identity(num_arms_, num_arms_);

                // KFRDB - observed reward
                VectorXd angle_between_true_movement_and_predicted_movement(num_arms_);
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const Eigen::VectorXd& true_movement = pull_result.action_result_;
                    const Eigen::VectorXd predicted_movement = bandit.getArmPrediction(arm_ind, arm_suggested_actions[arm_to_pull].suggested_action);

                    const double true_movement_norm = true_movement.norm();
                    const double predicted_movement_norm = predicted_movement.norm();

                    // Deal with the movement possibly being very small
                    if (true_movement_norm > 1e-10 && predicted_movement_norm > 1e-10)
                    {
                        angle_between_true_movement_and_predicted_movement(arm_ind) =
                                EigenHelpers::WeightedAngleBetweenVectors(true_movement, predicted_movement, VectorXd::Ones(true_movement.rows()));
                    }
                    else if (true_movement_norm <= 1e-10 && predicted_movement_norm <= 1e-10)
                    {
                        angle_between_true_movement_and_predicted_movement(arm_ind) = 0;
                    }
                    else
                    {
                        angle_between_true_movement_and_predicted_movement(arm_ind) = M_PI/2.0;
                    }
                }

                bool print_data = false;
                std::vector<ssize_t> bad_inds;
                VectorXd estimated_reward_improvements = VectorXd::Zero(num_arms_);
                const VectorXd true_reward_improvements = pull_result.all_possible_arms_true_reward_.array() - pull_result.all_possible_arms_true_reward_(arm_to_pull);

                VectorXd observed_reward = VectorXd::Zero(num_arms_);
                VectorXd angle_improvements = VectorXd::Zero(num_arms_);
                const double angle_to_arm_chosen = angle_between_true_movement_and_predicted_movement(arm_to_pull);
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const double angle_improvement = angle_to_arm_chosen - angle_between_true_movement_and_predicted_movement(arm_ind);
                    angle_improvements(arm_ind) = angle_improvement;
                    estimated_reward_improvements(arm_ind) = 0.2 * angle_improvement * std::abs(pull_result.true_reward_);
                    observed_reward(arm_ind) = pull_result.true_reward_ + estimated_reward_improvements(arm_ind);

                    if (estimated_reward_improvements(arm_ind) * true_reward_improvements(arm_ind) < 0)
                    {
                        print_data = true;
                        bad_inds.push_back(arm_ind);
                    }
                }

                // KFRDB - observation noise
                MatrixXd observation_noise(num_arms_, num_arms_);
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    observation_noise(i, i) = std::exp(-process_noise(i, arm_to_pull));
                }
                for (ssize_t i = 0; i < num_arms_; i++)
                {
                    for (ssize_t j = i + 1; j < num_arms_; j++)
                    {
                        observation_noise(i, j) = process_noise(i, j) * std::sqrt(observation_noise(i, i)) * std::sqrt(observation_noise(j, j));
                        observation_noise(j, i) = process_noise(i, j);
                    }
                }
                observation_noise *= std::pow(estimated_reward_scale, 2);
                observation_noise = process_noise;


                print_data = false;
                if (print_data)
                {
                    std::cout << "y_prev         : " << (bandit.getYCurrent() - pull_result.action_result_).transpose() << std::endl;
                    std::cout << "y_desired      : " << bandit.getYDesired().transpose() << std::endl;
                    std::cout << "desired y_delta: " << target_movement.transpose() << std::endl;
                    std::cout << "true action norm: " << pull_result.action_result_.norm() << std::endl;
                    std::cout << "bad inds: " << PrettyPrint::PrettyPrint(bad_inds) << std::endl;
                    std::cout << " true rewrd |"
                              << " est. rewrd |"
                              << " angl t vs p|"
                              << " angl-imp.  |"
                              << " true imp.  |"
                              << " est. imp.  |"
                              << "\n";

                    MatrixXd formatted_output;
                    formatted_output.resize(num_arms_, 6);

                    formatted_output.col(0) = pull_result.all_possible_arms_true_reward_;
                    formatted_output.col(1) = observed_reward;
                    formatted_output.col(2) = angle_between_true_movement_and_predicted_movement;
                    formatted_output.col(3) = angle_improvements;
                    formatted_output.col(4) = true_reward_improvements;
                    formatted_output.col(5) = estimated_reward_improvements;

                    std::cout << formatted_output << std::endl << std::endl;

                    //                    std::cout << "Process Noise (no scale factor):\n"
                    //                              << process_noise
                    //                              << std::endl << std::endl;
                    //                    std::cout << "Observation Noise (no scale factor):\n"
                    //                              << observation_noise
                    //                              << std::endl << std::endl;

                    //                    exit(-1);
                }


                kfrdb_alg.updateArms(process_noise, observation_matrix, observed_reward, observation_noise);
                total_regret += pull_result.true_regret_;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);

            return total_regret / (double)num_pulls_;
        }

        static double kfmanbTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "kfmanb_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            JacobianBandit<std::mt19937_64> bandit(generator, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            KalmanFilterMANB<std::mt19937_64> kfmanb_alg(VectorXd::Zero(num_arms_), VectorXd::Ones(num_arms_) * 1e6);
            double total_regret = 0;
            double estimated_reward_scale = min_reward_scale_;

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                const VectorXd target_movement = bandit.getTargetMovement();
                const auto arm_suggested_actions = bandit.getArmSuggestedActions(target_movement, use_optimization_);
                const size_t arm_to_pull = kfmanb_alg.selectArmToPull(generator);
                const auto pull_result = bandit.takeAction(arm_suggested_actions[arm_to_pull].suggested_action, arm_suggested_actions);
                estimated_reward_scale = rewardScaleEstimator(estimated_reward_scale, pull_result.true_reward_);

                LOG_DATA;

                kfmanb_alg.updateArms(std::pow(estimated_reward_scale, 2) * Eigen::VectorXd::Ones(num_arms_), arm_to_pull, pull_result.true_reward_, 0);
                total_regret += pull_result.true_regret_;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);

            return total_regret / (double)num_pulls_;
        }

        static double ucb1NormalTrial(std::mt19937_64 generator)
        {
            Log::Log trial_log(log_folder_ + "ucb1normal_trial_" + std::to_string(trial_ind_) + ".txt", true);
            trialLogHeader(trial_log);

            JacobianBandit<std::mt19937_64> bandit(generator, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            UCB1Normal<std::mt19937_64> ucb1normal_alg(num_arms_);
            double total_regret = 0;

            for (ssize_t pull_ind = 0; pull_ind < num_pulls_; pull_ind++)
            {
                const VectorXd target_movement = bandit.getTargetMovement();
                const auto arm_suggested_actions = bandit.getArmSuggestedActions(target_movement, use_optimization_);
                const size_t arm_to_pull = ucb1normal_alg.selectArmToPull(generator);
                const auto pull_result = bandit.takeAction(arm_suggested_actions[arm_to_pull].suggested_action, arm_suggested_actions);

                LOG_DATA;

                ucb1normal_alg.updateArms(arm_to_pull, pull_result.true_reward_);
                total_regret += pull_result.true_regret_;
            }

            logData(trial_log, bandit.getYCurrent(), bandit.getTargetMovement().norm(), VectorXd::Zero(num_jacobian_cols_), 0.0, 0.0);

            return total_regret / (double)num_pulls_;
        }

        static void trialLogHeader(Log::Log& log)
        {
            LOG_STREAM(log,
                       "position " << num_jacobian_rows_ << " elems"
                       << " | error"
                       << " | action " << num_jacobian_cols_ << " elems"
                       << " | reward"
                       << " | regret");
        }

        static void logJacobians(std::mt19937_64 generator)
        {
            Log::Log jacobian_log(log_folder_ + "jacobians_trial_" + std::to_string(trial_ind_) + ".txt", true);

            JacobianBandit<std::mt19937_64> bandit(generator, num_arms_, num_jacobian_rows_, num_jacobian_cols_, false, false);
            LOG(jacobian_log, "True");
            LOG(jacobian_log, bandit.getTrueJacobian());

            for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
            {
                LOG(jacobian_log, arm_ind);
                LOG(jacobian_log, bandit.getArmJacobian(arm_ind));
            }
        }

        static void logData(Log::Log& log, const VectorXd& position, const double error, const VectorXd& action, const double reward, const double regret)
        {
            LOG_STREAM(log, position.transpose() << " " << error << " " << action.transpose() << " " << reward << " " << regret);
        }

        static ssize_t num_arms_;
        static ssize_t num_trials_;
        static ssize_t num_pulls_;
        static ssize_t num_jacobian_rows_;
        static ssize_t num_jacobian_cols_;

        static bool use_optimization_;
        static double min_reward_scale_;

        static std::string log_folder_;
        static ssize_t trial_ind_;
};

ssize_t JacobianTrackingTrials::num_arms_;
ssize_t JacobianTrackingTrials::num_trials_;
ssize_t JacobianTrackingTrials::num_pulls_;
ssize_t JacobianTrackingTrials::num_jacobian_rows_;
ssize_t JacobianTrackingTrials::num_jacobian_cols_;

bool JacobianTrackingTrials::use_optimization_;
double JacobianTrackingTrials::min_reward_scale_;

std::string JacobianTrackingTrials::log_folder_;
ssize_t JacobianTrackingTrials::trial_ind_;

////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
//    CVXOptSolvers::Initialize();
    std::mt19937_64 generator(0xa8710913d2b5df6c); // a30cd67f3860ddb3) // MD5 sum of "Dale McConachie"
//    std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    // Purely Gaussian Bandits - Independant
    ////////////////////////////////////////////////////////////////////////////
    {
//        trial_params.num_arms_ = 10;
//        trial_num_trials = 100;
//        trial_num_pulls = 1000;
//        trial_params.initial_reward_variance_scale_factor_ = 0.0;
//        trial_params.transition_covariance_scale_factor_ = 0.0;
//        trial_params.observation_covariance_scale_factor_ = 0.0;

//        const std::vector<double> obs_std_dev_list = {12.5, 50.0/3.0, 25.0, 50.0*2.0/3.0, 50.0, 75.0, 100.0, 150.0, 200.0};
//        for (double obs_std_dev: obs_std_dev_list)
//        {
//            std::cout << "\n----------------------------------------------------------------------\n\n";
//            trial_params.observation_covariance_scale_factor_ = obs_std_dev * obs_std_dev;
//            std::cout << trial_params << std::endl;
//            const auto trial_results = IndependantGaussianBanditsTrials(trial_params);
//            std::cout << trial_results << std::endl;
//        }

//        trial_params.observation_covariance_scale_factor_ = 50.0 * 50.0;
//        const std::vector<double> transition_std_dev_list = {0.0, 12.5, 50.0/3.0, 25.0, 50.0*2.0/3.0, 50.0, 75.0, 100.0, 150.0, 200.0};
//        for (double transition_std_dev: transition_std_dev_list)
//        {
//            std::cout << "\n\n----------------------------------------------------------------------\n\n";
//            trial_params.transition_covariance_scale_factor_ = transition_std_dev * transition_std_dev;
//            std::cout << trial_params << std::endl;
//            const auto trial_results = IndependantGaussianBanditsTrials(trial_params);
//            std::cout << trial_results << std::endl;
//        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Linear Regression Bandits - Dependant
    ////////////////////////////////////////////////////////////////////////////
    {
//        trial_params.num_arms_ = 50;
//        trial_feature_vector_length = 10;
//        trial_params.feature_covariance_scale_factor_ = 1.0;
//        trial_params.observation_covariance_scale_factor_ = 0.2 * 0.2;
//        const std::vector<double> transition_std_dev_list = {0.0, 0.01, 0.02, 0.04, 0.08, 0.16};
//        for (double transition_std_dev: transition_std_dev_list)
//        {
//            std::cout << "\n\n----------------------------------------------------------------------\n\n";
//            trial_params.transition_covariance_scale_factor_ = transition_std_dev * transition_std_dev;
//            std::cout << trial_params << std::endl;
//            auto lr_results = LinearRegressionBanditsTrials(trial_params);
//            std::cout << lr_results << std::endl;
//        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Tracking trials
    ////////////////////////////////////////////////////////////////////////////
    {
        TrialParams params;

        params["Number of trials: "] = 10;
        params["Number of pulls:  "] = 400;

        params["Number of arms:   "] = 10;
        params["Num Jacobian rows: "] = 3;
        params["Num Jacobian cols: "] = 2;

        std::cout << "Usage:\n"
                  << "\tkalman_filter_synthetic_trials [numtrials [numpulls [numarms [numrows [numcols]]]]]\n"
                  << "Defaults:\n"
                  << PrettyPrint::PrettyPrint(params, false, "\n") << std::endl << std::endl;

        if (argc > 1)
        {
            params["Number of trials: "] = std::atoi(argv[1]);
        }

        if (argc > 2)
        {
            params["Number of pulls:  "] = std::atoi(argv[2]);
        }

        if (argc > 3)
        {
            params["Number of arms:   "] = std::atoi(argv[3]);
        }

        if (argc > 4)
        {
            params["Num Jacobian rows: "] = std::atoi(argv[4]);
        }

        if (argc > 5)
        {
            params["Num Jacobian cols: "] = std::atoi(argv[5]);
        }

        if (argc > 6)
        {
            for (int arg = 6; arg < argc; arg++)
            {
                std::cout << "Unused parameter " << argv[arg] << std::endl;
            }
        }


        // Test without optimization
        {
            std::cout << "\n\n---------------No Optimization----------------------------------------\n\n";
            params["Optimize:          "] = false;
            std::cout << PrettyPrint::PrettyPrint(params, false, "\n") << std::endl;

            std::mt19937_64 generator_copy = generator;
            auto results = JacobianTrackingTrials::run(generator_copy, params);
            std::cout << results << std::endl;
        }

        // Test with optimization
        {
            std::cout << "\n\n---------------With Optimization--------------------------------------\n\n";
            params["Optimize:          "] = true;
            std::cout << PrettyPrint::PrettyPrint(params, false, "\n") << std::endl;

            std::mt19937_64 generator_copy = generator;
            auto results = JacobianTrackingTrials::run(generator_copy, params);
            std::cout << results << std::endl;
        }
    }

//    CVXOptSolvers::Finalize();
    return 0;
}
