#ifndef CLMANAGER_HPP
#define CLMANAGER_HPP

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
	#include <opencl/cl.h>
	#include <opencl/cl_platform.h>
#else
	#include <CL/cl.h>
	#include <CL/cl_platform.h>
#endif

#include "CLWrappers.hpp"

using namespace std;

class CLManager {

	public:

		CLManager();
		~CLManager();

		void                  createCLPlatforms();
		void                  createCLContexts();
		cl_context            createCLContext(OpenCLPlatformWrapper* clPlatformWrapper,string contextType);
		void                  createCLDevices(OpenCLContextWrapper* contextWrapper);
		void			      createCLCommandQueues();
		cl_command_queue      createCLCommandQueue(OpenCLContextWrapper* contextWrapper, OpenCLDeviceIDWrapper* deviceWrapper);
		OpenCLKernelWrapper*  createCLKernel(OpenCLProgramWrapper* programWrapper, const char* kernelName);
		OpenCLProgramWrapper* createCLProgram(OpenCLContextWrapper* contextWrapper, 
			                                  OpenCLDeviceIDWrapper* deviceWrapper, 
											  const char* fileName, 
											  const char* programName);

		OpenCLMemoryObjectWrapper* createCLMemoryObject(OpenCLContextWrapper* contextWrapper,
														cl_mem_flags flags,
														size_t bufferSize,
														void* hostPtr,
														char* name);

		OpenCLMemoryObjectWrapper* loadImage(OpenCLContextWrapper* contextWrapper, char *fileName);
		bool                       saveImage(OpenCLMemoryObjectWrapper* imageWrapper,char* filename, OpenCLCommandQueueWrapper* cqWrapper);

		OpenCLMemoryObjectWrapper* createBlankCLImage(OpenCLContextWrapper* contextWrapper, 
			                                        cl_mem_flags const    flags, 
													cl_uint const         format, 
													cl_uint const         dataType,
													cl_uint const         width, 
													cl_uint const         height, 
													char* const           name);

		OpenCLSamplerWrapper* createCLSampler(OpenCLContextWrapper* contextWrapper, cl_bool normalized, cl_addressing_mode addressingMode, cl_filter_mode filterMode, char* name);
	
		void			  initCL();
		void			  printDebugInfo();
		bool			  deviceSupportsExtension(OpenCLDeviceIDWrapper* device, string extension);
		
		OpenCLPlatformWrapper*     getCLPlatform(string name);
		OpenCLContextWrapper*      getCLContext(string name, string type);
		OpenCLDeviceIDWrapper*     getCLDevice(string name);
		OpenCLCommandQueueWrapper* getCLCommandQueue(string name);

		OpenCLKernelWrapper*       getCLKernel(string name);
		OpenCLProgramWrapper*      getCLProgram(string name);
		OpenCLMemoryObjectWrapper* getCLMemoryObject(string name);

	private:

		//platforms
		vector<OpenCLPlatformWrapper*> CLPlatforms;
		OpenCLPlatformWrapper CurrentCLPlatform;

		//contexts
		vector<OpenCLContextWrapper*> CLContexts;
		OpenCLContextWrapper       CurrentCLContext;

		//command queues
		vector<OpenCLCommandQueueWrapper*> CLCommandQueues;
		OpenCLCommandQueueWrapper CurrentCLCommandQueue;

		//programs
		vector<OpenCLProgramWrapper*> CLPrograms;
		OpenCLProgramWrapper       CurrentCLProgram;

		//devices
		vector<OpenCLDeviceIDWrapper*> CLDevices;
		OpenCLDeviceIDWrapper     CurrentCLDevice;

		//kernels
		vector<OpenCLKernelWrapper*> CLKernels;
		OpenCLKernelWrapper        CurrentCLKernel;

		//memory objects
		vector<OpenCLMemoryObjectWrapper*> CLMemoryObjects;
		OpenCLMemoryObjectWrapper CurrentCLMemoryObject;

		//samplers
		vector<OpenCLSamplerWrapper*> CLSamplers;
		//OpenCLMemoryObjectWrapper CurrentCLMemoryObject;

};


#endif 