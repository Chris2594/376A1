//
//  main.cpp
//  376A1
//
//  Created by Chris on 12/8/19.
//  Copyright © 2019 BScCompSci. All rights reserved.
//

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS    // using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS                // enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// functions to handle errors
#include "error.h"

int main(int argc, const char * argv[]) {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    
    cl::Context context;
    std::vector<cl::Device> contextDevices;
    cl::CommandQueue queue;
    
    std::vector<cl::Kernel> allKernels;
    
    int deviceSelect = 0;

    try{
        // PART 1: DEVICE TYPE SELECT
        std::cout << "==============================" << std::endl;
        std::cout << "Would you like to select:\n\t(1) CPU\n\t(2) GPU\nSelect device >> ";
        std::cin >> deviceSelect;
        cl::Platform::get(&platforms);
//        std::cout << "==============================" << std::endl;
//        std::cout << "(DEBUG) num plat: " << platforms.size() << std::endl;
//        for(int i=0; i<platforms.size(); i++){
//            std::cout << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
//        }
        
        for(int i=0; i < platforms.size(); i++){
            switch(deviceSelect){
                case 1:
                    std::cout << "\nYou have selected CPU" << std::endl;
                    platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &devices);
                    break;
                case 2:
                    std::cout << "\nYou have selected GPU" << std::endl;
                    platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                    break;
                default:
                    std::cerr << "\nIncorrect selection, Terminating!" << std::endl;
                    exit(EXIT_FAILURE);
                    break;
            }
        }
        //std::cout << "==============================" << std::endl;
        //std::cout << "(DEBUG) num dev: " << devices.size() << std::endl;
        
        //PART 2 DEVICE INFO DISPLAY
        for(int i = 0; i < devices.size(); i++){
            std::cout << "==============================" << std::endl;
            std::cout << "Device " << i <<"\n--------" << std::endl;
            cl::Platform A(devices[i].getInfo<CL_DEVICE_PLATFORM>());
            std::cout << "Platform:\t\t\t\t" << A.getInfo<CL_PLATFORM_NAME>() << std::endl;
            if (devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
                std::cout << "Type:\t\t\t\t\t" << "CPU" << std::endl;
            else if (devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
                std::cout << "Type:\t\t\t\t\t" << "GPU" << std::endl;
            else
                std::cout << "Type:\t\t\t\t\t" << "Other" << std::endl;
            std::cout << "Name:\t\t\t\t\t" << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "Max Compute Units:\t\t" << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "Max Work Group Size:\t" << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "Max Work Item Sizes:\t";
            std::vector<std::size_t> a = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            std::cout << a[0];
            for(int j = 1; j < a.size(); j++){
                std::cout << " / " << a[j];
            }
            std::cout << "\nLocal Memory Size:\t\t" <<  devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
        }
        std::cout << "==============================" << std::endl;
        
        //PART 3 DEVICE SELECTION
        std::cout << "Select a device >> ";
        deviceSelect = -1;
        std::cin >> deviceSelect;
        std::string deviceExt = devices[deviceSelect].getInfo<CL_DEVICE_EXTENSIONS>();
        
        //std::cout << deviceExt << std::endl;
        std::cout << "\nDevice ";
        if(!(deviceExt.find("cl_khr_icd") == std::string::npos))
            std::cout << "supports ";
        else
            std::cout << "does not support ";
        std::cout << "cl_khr_icd extension\n" << std::endl;

        context = cl::Context(devices[deviceSelect]);
        contextDevices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, contextDevices[0]);
        std::cout << "==============================" << std::endl;

        
        //PART 4 SOURCE FILE
        std::ifstream programFile("source.cl");
        if(!programFile.is_open()){
            quit_program("File not found");
        }
        std::string programString(
                                   std::istreambuf_iterator<char>(programFile),
                                   (std::istreambuf_iterator<char>())
                                 );
        programFile.close();
        
        //std::cout << programString << std::endl;
        
        cl::Program::Sources source(1, std::make_pair(programString.c_str(), programString.length() + 1));
        cl::Program program(context, source);
        try{
            program.build(contextDevices);
            std::cout << "Build Successful" << std::endl;
            for(int i = 0; i<contextDevices.size();i++){
                std::cout << "Device - " << contextDevices[i].getInfo<CL_DEVICE_NAME>() << ", build log:" << std::endl;
                std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[i]) << std::endl;
            }
        }
        catch (cl::Error e){
            if (e.err() == CL_BUILD_PROGRAM_FAILURE){
                std::cout << e.what() << ": Failed to build program." << std::endl;
                for(int i=0;i<contextDevices.size();i++){
                std::cout << "Device ‐ " << contextDevices[i].getInfo<CL_DEVICE_NAME>() << ", build log:"<< std::endl;
                std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[i]) << "‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐" << std::endl;
                }
            }
        }
        std::cout << "==============================" << std::endl;

        //PART 5 KERNEL OPERATIONS
        program.createKernels(&allKernels);
        std::cout << "Number of kernels: " << allKernels.size() << std::endl;
        for(int i=0; i<allKernels.size(); i++){
            std::cout << "Kernel " << i << " - " << allKernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
        }
    }
    // catch any OpenCL function errors
    catch (cl::Error e) {
        // call function to handle errors
        handle_error(e);
    }
    return 0;
    
}
