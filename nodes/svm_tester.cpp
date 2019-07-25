#include "smmap/svm_classifier.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "smmap_svm_tester");

    auto nh = std::make_shared<ros::NodeHandle>();
    auto ph = std::make_shared<ros::NodeHandle>("~");

    const auto clf = smmap::SVMClassifier(nh, ph);
    {
        Eigen::VectorXd vec(4);
        vec << 0.634196, 0.609892, 0.634196, 0.609892;
        std::cout << "SKLearn prediction: -1  Prediction: " << clf.predict(vec) << "  Vec: " << vec.transpose() << std::endl;
    }
    {
        Eigen::VectorXd vec(4);
        vec << 0.526977, 0.502574, 0.531208, 0.510087;
        std::cout << "SKLearn prediction:  1  Prediction: " << clf.predict(vec) << "  Vec: " << vec.transpose() << std::endl;
    }

    return EXIT_SUCCESS;
}
