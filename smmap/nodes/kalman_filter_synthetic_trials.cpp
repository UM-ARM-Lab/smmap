#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>

#include "smmap/kalman_filter_multiarm_bandit.hpp"
#include "smmap/ucb_multiarm_bandit.hpp"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;
using namespace arc_helpers;

////////////////////////////////////////////////////////////////////////////////
// Helper objects
////////////////////////////////////////////////////////////////////////////////

struct RewardObservation
{
    public:
        VectorXd true_reward;
        VectorXd observed_reward;
};

template <class Generator>
struct TrialParams
{
    public:
        TrialParams(Generator& generator)
            : num_arms_(10)
            , num_trials_(1000)
            , num_pulls_(2000)
            , initial_reward_variance_scale_factor_(0.0)
            , feature_vector_length_(10)
            , feature_covariance_scale_factor_(0.0)
            , transition_covariance_scale_factor_(0.0)
            , observation_covariance_scale_factor_(0.0)
            , generator_(generator)
        {}

        ssize_t num_arms_;
        size_t num_trials_;
        size_t num_pulls_;

        double initial_reward_variance_scale_factor_;
        ssize_t feature_vector_length_;
        double feature_covariance_scale_factor_;
        double transition_covariance_scale_factor_;
        double observation_covariance_scale_factor_;

        Generator& generator_;
};

template <class Generator>
std::ostream& operator<<(std::ostream& os, const TrialParams<Generator>& tp)
{
    os << "Num bandits: " << tp.num_arms_ << std::endl
       << "Num trials:  " << tp.num_trials_ << std::endl
       << "Num pulls:   " << tp.num_pulls_ << std::endl;
    os << std::setw(9) << std::setprecision(6) << std::fixed;
    os << "Reward std dev factor:      " << std::sqrt(tp.initial_reward_variance_scale_factor_) << std::endl
       << "Feature vector length:      " << tp.feature_vector_length_ << std::endl
       << "Feature std dev factor:     " << std::sqrt(tp.feature_covariance_scale_factor_) << std::endl
       << "Transition std dev factor:  " << std::sqrt(tp.transition_covariance_scale_factor_) << std::endl
       << "Observation std dev factor: " << std::sqrt(tp.observation_covariance_scale_factor_) << std::endl;
    os << std::setw(1) << std::setprecision(6);
    return os;
}

struct TrialResults
{
    public:
        TrialResults(const ssize_t num_trials)
            : num_trials_(num_trials)
            , kfrdb_average_regret_(num_trials)
            , average_kfrdb_avg_regret_(0)
            , variance_kfrdb_avg_regret_(0)
            , kfmanb_average_regret_(num_trials)
            , average_kfmanb_avg_regret_(0)
            , variance_kfmanb_avg_regret_(0)
            , ucb1normal_average_regret_(num_trials)
            , average_ucb1normal_avg_regret_(0)
            , variance_ucb1normal_avg_regret_(0)
        {}

        const ssize_t num_trials_;

        Eigen::VectorXd kfrdb_average_regret_;
        double average_kfrdb_avg_regret_;
        double variance_kfrdb_avg_regret_;

        Eigen::VectorXd kfmanb_average_regret_;
        double average_kfmanb_avg_regret_;
        double variance_kfmanb_avg_regret_;

        Eigen::VectorXd ucb1normal_average_regret_;
        double average_ucb1normal_avg_regret_;
        double variance_ucb1normal_avg_regret_;

        void calculateStatistics()
        {
            average_kfrdb_avg_regret_ = kfrdb_average_regret_.mean();
            average_kfmanb_avg_regret_ = kfmanb_average_regret_.mean();
            average_ucb1normal_avg_regret_ = ucb1normal_average_regret_.mean();

            variance_kfrdb_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfrdb_average_regret_.squaredNorm() - std::pow(average_kfrdb_avg_regret_, 2);
            variance_kfmanb_avg_regret_ = 1.0/(double)(num_trials_ - 1) * kfmanb_average_regret_.squaredNorm() - std::pow(average_kfmanb_avg_regret_, 2);
            variance_ucb1normal_avg_regret_ = 1.0/(double)(num_trials_ - 1) * ucb1normal_average_regret_.squaredNorm() - std::pow(average_ucb1normal_avg_regret_, 2);
        }
};

std::ostream& operator<<(std::ostream& os, const TrialResults& tr)
{
    os << std::setw(9) << std::setprecision(6) << std::fixed;
    os << "KF-RDB average regret:      " << tr.average_kfrdb_avg_regret_
       << " KF-RDB std dev:      " << std::sqrt(tr.variance_kfrdb_avg_regret_) << std::endl
       << "KF-MANB average regret:     " << tr.average_kfmanb_avg_regret_
       << " KF-MANB std dev:     " << std::sqrt(tr.variance_kfmanb_avg_regret_) << std::endl
       << "UCB1-Normal average regret: " << tr.average_ucb1normal_avg_regret_
       << " UCB1-Normal std dev: " << std::sqrt(tr.variance_ucb1normal_avg_regret_) << std::endl;
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
            reward.true_reward.resize((ssize_t)num_arms_);
            reward.observed_reward.resize((ssize_t)num_arms_);

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

////////////////////////////////////////////////////////////////////////////////
// Trial functions
////////////////////////////////////////////////////////////////////////////////

template <class Generator>
TrialResults IndependantGaussianBanditsTrials(TrialParams<Generator>& params)
{
    TrialResults results(params.num_trials_);

    for (size_t trial_ind = 0; trial_ind < params.num_trials_; trial_ind++)
    {
        // Generate the bandit itself
        VectorXd reward_mean = VectorXd::Zero(params.num_arms_);
        for (ssize_t bandit_ind = 0; bandit_ind < params.num_arms_; bandit_ind++)
        {
            reward_mean(bandit_ind) = (double)(params.num_arms_ - bandit_ind) * 50.0;
        }
        const MatrixXd initial_reward_covariance = MatrixXd::Identity(params.num_arms_, params.num_arms_) * params.initial_reward_variance_scale_factor_;
        const MatrixXd transition_covariance = MatrixXd::Identity(params.num_arms_, params.num_arms_) * params.transition_covariance_scale_factor_;
        const MatrixXd observation_covariance = MatrixXd::Identity(params.num_arms_, params.num_arms_) * params.observation_covariance_scale_factor_;
        MultiarmGaussianBandit<Generator> bandits(params.generator_, reward_mean, initial_reward_covariance, transition_covariance, observation_covariance);

        // Create the algorithms
        KalmanFilterRDB<Generator> kfrdb_alg(VectorXd::Zero(params.num_arms_), MatrixXd::Identity(params.num_arms_, params.num_arms_) * 1e100);
        double kfrdb_total_regret = 0;

        KalmanFilterMANB<Generator> kfmanb_alg(VectorXd::Zero(params.num_arms_), VectorXd::Ones(params.num_arms_) * 1e100);
        double kfmanb_total_regret = 0;

        UCB1Normal ucb1normal_alg(params.num_arms_);
        double ucb1normal_total_regret = 0;

        // Pull the arms
        for (size_t pull_ind = 0; pull_ind < params.num_pulls_; pull_ind++)
        {
            // Determine which arm each algorithm pulls
            const ssize_t kfrdb_arm_pulled = kfrdb_alg.selectArmToPull(params.generator_);
            const ssize_t kfmanb_arm_pulled = kfmanb_alg.selectArmToPull(params.generator_);
            const ssize_t ucb1normal_arm_pulled = ucb1normal_alg.selectArmToPull();

            // Pull all the arms to determine true and observed rewards
            const double best_expected_reward = bandits.getRewardMean().maxCoeff();
            const auto rewards = bandits.pullArms();

            // Give rewards for each algorithm
            kfrdb_total_regret += best_expected_reward - rewards.true_reward(kfrdb_arm_pulled);
            kfmanb_total_regret += best_expected_reward - rewards.true_reward(kfmanb_arm_pulled);
            ucb1normal_total_regret += best_expected_reward - rewards.true_reward(ucb1normal_arm_pulled);

            // Update each algorithm
            const MatrixXd kfrdb_observation_matrix = MatrixXd::Identity(params.num_arms_, params.num_arms_);
            const VectorXd kfrdb_observed_reward = rewards.observed_reward;
            kfrdb_alg.updateArms(bandits.getTransitionCovariance(), kfrdb_observation_matrix, kfrdb_observed_reward, bandits.getObservationCovariance());

            const VectorXd kfmanb_transition_variance = bandits.getTransitionCovariance().diagonal();
            const double kfmanb_observed_reward = rewards.observed_reward(kfmanb_arm_pulled);
            const double kfmanb_observation_variance = bandits.getObservationCovariance()(kfmanb_arm_pulled, kfmanb_arm_pulled);
            kfmanb_alg.updateArms(kfmanb_transition_variance, kfmanb_arm_pulled, kfmanb_observed_reward, kfmanb_observation_variance);

            ucb1normal_alg.updateArms(ucb1normal_arm_pulled, rewards.observed_reward(ucb1normal_arm_pulled));
        }

        // Record the results
        results.kfrdb_average_regret_(trial_ind) = kfrdb_total_regret / (double)params.num_pulls_;
        results.kfmanb_average_regret_(trial_ind) = kfmanb_total_regret / (double)params.num_pulls_;
        results.ucb1normal_average_regret_(trial_ind) = ucb1normal_total_regret / (double)params.num_pulls_;

        std::cout << "Trial Num: " << trial_ind
                  << " KF-RDB: " << results.kfrdb_average_regret_(trial_ind)
                  << " KF-MANB: " << results.kfmanb_average_regret_(trial_ind)
                  << " UCB1-Normal: " << results.ucb1normal_average_regret_(trial_ind)
                  << std::endl;
    }

    results.calculateStatistics();

    return results;
}

template <class Generator>
TrialResults LinearRegressionBanditsTrials(TrialParams<Generator>& params)
{
    TrialResults results(params.num_trials_);

    for (size_t trial_ind = 0; trial_ind < params.num_trials_; trial_ind++)
    {
        // Create the bandit itself
        VectorXd starting_weights(params.feature_vector_length_);
        starting_weights(0) = 1.0;
        const MatrixXd feature_covariance = MatrixXd::Identity(params.feature_vector_length_, params.feature_vector_length_) * params.feature_covariance_scale_factor_;
        const MatrixXd weights_transition_covariance = MatrixXd::Identity(params.feature_vector_length_, params.feature_vector_length_) * params.transition_covariance_scale_factor_;
        const double observation_covariance = params.observation_covariance_scale_factor_;
        LinearRegressionBandit<Generator> bandits(params.generator_, params.num_arms_, starting_weights, feature_covariance, weights_transition_covariance, observation_covariance);

        // Create the algorithms
        KalmanFilterRDB<Generator> kfrdb_alg(VectorXd::Zero(params.num_arms_), MatrixXd::Identity(params.num_arms_, params.num_arms_) * 1e100);
        double kfrdb_total_regret = 0;
        double kfrdb_current_reward_std_dev_scale = 1.0;

        KalmanFilterMANB<Generator> kfmanb_alg(VectorXd::Zero(params.num_arms_), VectorXd::Ones(params.num_arms_) * 1e100);
        double kfmanb_total_regret = 0;

        UCB1Normal ucb1normal_alg(params.num_arms_);
        double ucb1normal_total_regret = 0;

        // Pull the arms
        for (size_t pull_ind = 0; pull_ind < params.num_pulls_; pull_ind++)
        {
            // Determine which arm each algorithm pulls
            const ssize_t kfrdb_arm_pulled = kfrdb_alg.selectArmToPull(params.generator_);
            const ssize_t kfmanb_arm_pulled = kfmanb_alg.selectArmToPull(params.generator_);
            const ssize_t ucb1normal_arm_pulled = ucb1normal_alg.selectArmToPull();

            // Pull all the arms to determine true and observed rewards
            const VectorXd features = bandits.getFeatures();
            std::vector<VectorXd> actions = bandits.getActions(features);
            std::vector<double> predictions = bandits.getPredictions(features);
            const std::pair<RewardObservation, double> arm_results = bandits.pullArms(features);
            const RewardObservation& rewards = arm_results.first;
            const double& observed_regression = arm_results.second;
            const double best_expected_reward = 0;

            // Give rewards for each algorithm
            kfrdb_total_regret += best_expected_reward - rewards.true_reward(kfrdb_arm_pulled);
            kfmanb_total_regret += best_expected_reward - rewards.true_reward(kfmanb_arm_pulled);
            ucb1normal_total_regret += best_expected_reward - rewards.true_reward(ucb1normal_arm_pulled);

            // Update each algorithm

            // KFRDB - process noise
            MatrixXd kfrdb_process_noise = MatrixXd::Identity(params.num_arms_, params.num_arms_);
            for (ssize_t i = 0; i < params.num_arms_; i++)
            {
                for (ssize_t j = i + 1; j < params.num_arms_; j++)
                {
                    kfrdb_process_noise(i, j) = actions[i].dot(actions[j]) / (actions[i].norm() * actions[j].norm());
                    kfrdb_process_noise(j, i) = kfrdb_process_noise(i, j);
                }
            }

            // KFRDB - observation matrix
            MatrixXd kfrdb_observation_matrix = MatrixXd::Identity(params.num_arms_, params.num_arms_);

            // KFRDB - observed reward
            VectorXd kfrdb_observed_reward = VectorXd::Zero(params.num_arms_);
            const double equivalent_distance = std::abs(observed_regression - predictions[(size_t)kfrdb_arm_pulled]);
            kfrdb_current_reward_std_dev_scale = 0.9 * kfrdb_current_reward_std_dev_scale + 0.1 * equivalent_distance;
            for (ssize_t arm_ind = 0; arm_ind < params.num_arms_; arm_ind++)
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
            MatrixXd kfrdb_observation_noise = MatrixXd::Identity(params.num_arms_, params.num_arms_);
            for (ssize_t i = 0; i < params.num_arms_; i++)
            {
                kfrdb_observation_noise(i, i) = std::exp(-kfrdb_process_noise(i, kfrdb_arm_pulled));
            }
            for (ssize_t i = 0; i < params.num_arms_; i++)
            {
                for (ssize_t j = i + 1; j < params.num_arms_; j++)
                {
                    kfrdb_observation_noise(i, j) = kfrdb_process_noise(i, j) * std::sqrt(kfrdb_observation_noise(i, i)) * std::sqrt(kfrdb_observation_noise(j, j));
                    kfrdb_observation_noise(j, i) = kfrdb_process_noise(i, j);
                }
            }

            kfrdb_alg.updateArms(kfrdb_process_noise, kfrdb_observation_matrix, kfrdb_observed_reward, kfrdb_observation_noise);

            const VectorXd kfmanb_transition_variance = VectorXd::Ones(params.num_arms_);
            const double kfmanb_observed_reward = rewards.observed_reward(kfmanb_arm_pulled);
            const double kfmanb_observation_variance = bandits.getObservationCovariance();
            kfmanb_alg.updateArms(kfmanb_transition_variance, kfmanb_arm_pulled, kfmanb_observed_reward, kfmanb_observation_variance);

            ucb1normal_alg.updateArms(ucb1normal_arm_pulled, rewards.observed_reward(ucb1normal_arm_pulled));
        }

        // Record the results
        results.kfrdb_average_regret_(trial_ind) = kfrdb_total_regret / (double)params.num_pulls_;
        results.kfmanb_average_regret_(trial_ind) = kfmanb_total_regret / (double)params.num_pulls_;
        results.ucb1normal_average_regret_(trial_ind) = ucb1normal_total_regret / (double)params.num_pulls_;

        std::cout << "Trial Num: " << trial_ind;
        std::cout << std::setw(9) << std::setprecision(6) << std::fixed;
        std::cout << " KF-RDB: " << results.kfrdb_average_regret_(trial_ind)
                  << " KF-MANB: " << results.kfmanb_average_regret_(trial_ind)
                  << " UCB1-Normal: " << results.ucb1normal_average_regret_(trial_ind);
        std::cout << std::setw(1) << std::setprecision(6)
                  << std::endl;
    }

    results.calculateStatistics();

    return results;
}

////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::mt19937_64 generator(0xa8710913d2b5df6c); // a30cd67f3860ddb3) // MD5 sum of "Dale McConachie"
//    std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count());
    TrialParams<std::mt19937_64> trial_params(generator);

    ////////////////////////////////////////////////////////////////////////////
    // Purely Gaussian Bandits - Independant
    ////////////////////////////////////////////////////////////////////////////

    trial_params.num_arms_ = 10;
    trial_params.num_trials_ = 100;
    trial_params.num_pulls_ = 1000;
    trial_params.initial_reward_variance_scale_factor_ = 0.0;
    trial_params.transition_covariance_scale_factor_ = 0.0;
    trial_params.observation_covariance_scale_factor_ = 0.0;

//    const std::vector<double> obs_std_dev_list = {12.5, 50.0/3.0, 25.0, 50.0*2.0/3.0, 50.0, 75.0, 100.0, 150.0, 200.0};
//    for (double obs_std_dev: obs_std_dev_list)
//    {
//        std::cout << "\n----------------------------------------------------------------------\n\n";
//        trial_params.observation_covariance_scale_factor_ = obs_std_dev * obs_std_dev;
//        std::cout << trial_params << std::endl;
//        const auto trial_results = IndependantGaussianBanditsTrials(trial_params);
//        std::cout << trial_results << std::endl;
//    }

//    trial_params.observation_covariance_scale_factor_ = 50.0 * 50.0;
//    const std::vector<double> transition_std_dev_list = {0.0, 12.5, 50.0/3.0, 25.0, 50.0*2.0/3.0, 50.0, 75.0, 100.0, 150.0, 200.0};
//    for (double transition_std_dev: transition_std_dev_list)
//    {
//        std::cout << "\n\n----------------------------------------------------------------------\n\n";
//        trial_params.transition_covariance_scale_factor_ = transition_std_dev * transition_std_dev;
//        std::cout << trial_params << std::endl;
//        const auto trial_results = IndependantGaussianBanditsTrials(trial_params);
//        std::cout << trial_results << std::endl;
//    }

    ////////////////////////////////////////////////////////////////////////////
    // Linear Regression Bandits - Dependant
    ////////////////////////////////////////////////////////////////////////////

//    trial_params.num_arms_ = 50;
//    trial_params.feature_vector_length_ = 10;
//    trial_params.feature_covariance_scale_factor_ = 1.0;
//    trial_params.observation_covariance_scale_factor_ = 0.2 * 0.2;
//    const std::vector<double> transition_std_dev_list = {0.0, 0.01, 0.02, 0.04, 0.08, 0.16};
//    for (double transition_std_dev: transition_std_dev_list)
//    {
//        std::cout << "\n\n----------------------------------------------------------------------\n\n";
//        trial_params.transition_covariance_scale_factor_ = transition_std_dev * transition_std_dev;
//        std::cout << trial_params << std::endl;
//        auto lr_results = LinearRegressionBanditsTrials(trial_params);
//        std::cout << lr_results << std::endl;
//    }

    ////////////////////////////////////////////////////////////////////////////
    // Tracking trials
    ////////////////////////////////////////////////////////////////////////////

    trial_params.num_arms_ = 50;
    trial_params.feature_vector_length_ = 10;
    trial_params.feature_covariance_scale_factor_ = 1.0;
    trial_params.observation_covariance_scale_factor_ = 0.2 * 0.2;
    const std::vector<double> transition_std_dev_list = {0.0, 0.01, 0.02, 0.04, 0.08, 0.16};
    for (double transition_std_dev: transition_std_dev_list)
    {
        std::cout << "\n\n----------------------------------------------------------------------\n\n";
        trial_params.transition_covariance_scale_factor_ = transition_std_dev * transition_std_dev;
        std::cout << trial_params << std::endl;
        auto lr_results = LinearRegressionBanditsTrials(trial_params);
        std::cout << lr_results << std::endl;
    }


    return 0;
}

