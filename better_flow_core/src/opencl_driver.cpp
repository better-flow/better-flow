#include <better_flow/opencl_driver.h>


bool OpenCLDriver::enabled = false;

#if OPENCL_ENABLED
cl::Context *OpenCLDriver::context = NULL;
cl::Program *OpenCLDriver::program = NULL;
cl::CommandQueue *OpenCLDriver::queue = NULL;
cl::Kernel OpenCLDriver::pr_4param_ker = cl::Kernel();
cl::Kernel OpenCLDriver::model_helper_ker = cl::Kernel();
#endif

void OpenCLDriver::init (std::string fname_) {
    OpenCLDriver::enabled = false;

#if OPENCL_ENABLED
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size () == 0) {
        std::cout<<" No OpenCL platforms found!\n";
        return;
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
 
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout<<" No OpenCL devices found!\n";
        return;
    }

    cl::Device default_device=all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    OpenCLDriver::context = new cl::Context({default_device});
    cl::Program::Sources sources;

    std::ifstream file(fname_);
    if (file.is_open() ) {std::cout << "Kernel file successfully opened.\n"; }
    else { std::cout << "ERROR: Unable to open kernel file. Please, try again.\n"; exit(1); };    
    std::string kernel_code(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
 
    OpenCLDriver::program = new cl::Program(*OpenCLDriver::context, sources);
    if (OpenCLDriver::program->build({default_device}) != CL_SUCCESS) {
        std::cout << " Error building: " << OpenCLDriver::program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    
    OpenCLDriver::queue = new cl::CommandQueue(*OpenCLDriver::context, default_device);

    OpenCLDriver::pr_4param_ker = cl::Kernel(*OpenCLDriver::program, "pr_4param");
    OpenCLDriver::model_helper_ker = cl::Kernel(*OpenCLDriver::program, "model_helper");

    OpenCLDriver::enabled = true;
#else
    (void)fname_;
#endif
}
