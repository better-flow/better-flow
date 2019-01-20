#ifndef OPENCL_DRIVER_H
#define OPENCL_DRIVER_H

#include <better_flow/common.h>

#if OPENCL_ENABLED
#include <CL/cl.hpp>
#endif

class OpenCLDriver {
public:

#if OPENCL_ENABLED
    static cl::Context *context;
    static cl::Program *program;
    static cl::CommandQueue *queue;
    static cl::Kernel pr_4param_ker, model_helper_ker;
#endif

    static bool enabled;
    static void init (std::string fname_ = "/home/alice/better_flow/better_flow_core/src/gpu_impl.cl");
};


#endif // OPENCL_DRIVER_H
