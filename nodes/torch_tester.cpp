#include <torch/script.h>
#include <ros/ros.h>

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "smmap_torch_tester");

//    auto nh = std::make_shared<ros::NodeHandle>();
//    auto ph = std::make_shared<ros::NodeHandle>("~");

    torch::jit::script::Module model = torch::jit::load("/home/dmcconac/Dropbox/catkin_ws/src/smmap_jupyter/src/smmap_jupyter/13feature_torch_model_cpu.pt");
    const auto params = model.get_parameters();
    std::cout << params.size() << std::endl;
    for(const auto& param : params)
    {
        std::cout << param.name() << std::endl;
    }
//    module->to(torch::kCUDA);
//    module = torch::jit::load("/home/dmcconac/Dropbox/catkin_ws/src/smmap_jupyter/src/smmap_jupyter/13feature_torch_model.pt");
//    std::cout << "ok\n";

    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(torch::ones({13}).to(torch::kCUDA));
    inputs.push_back(torch::ones({13}));
    at::Tensor output = model.forward(inputs).toTensor();
    std::cout << output.item().toFloat() << std::endl;

    return EXIT_SUCCESS;
}
