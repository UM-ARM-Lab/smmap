#include <torch/script.h>
#include <ros/ros.h>
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "smmap_torch_tester");

//    auto nh = std::make_shared<ros::NodeHandle>();
//    auto ph = std::make_shared<ros::NodeHandle>("~");

    torch::jit::script::Module model = torch::jit::load(
                "/home/dmcconac/Dropbox/catkin_ws/src/smmap/logs/rope_hooks/13feature_torch_model_cpu.pt");
//    module->to(torch::kCUDA);

    Eigen::VectorXd vec(13);
    vec << 0.66673756,  0.6558349 ,  0.7069159 ,  0.7233157 ,  0.33333334,  0.33333334,  0.        ,  0.33333334,  0.33333334,  0.        , -1.        , -1.        ,  0. ;

    auto vec_torch = torch::empty({13});
    for (int idx = 0; idx < 13; ++idx)
    {
        vec_torch[idx] = vec[idx];
    }
    std::vector<torch::jit::IValue> const query(1, vec_torch);

//    inputs.push_back(torch::ones({13}).to(torch::kCUDA));
//    inputs.push_back(torch::ones({13}));

    std::cout << "Output: " << model.forward(query).toTensor().item().toFloat() << std::endl;

    return EXIT_SUCCESS;
}
