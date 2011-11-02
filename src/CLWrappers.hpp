#ifndef CLWRAPPERS_HPP
#define CLWRAPPERS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>

#ifdef __APPLE__
	#include <opencl/cl.h>
	#include <opencl/cl_platform.h>
#else
	#include <CL/cl.h>
	#include <CL/cl_platform.h>
#endif

using namespace std;

#define CL_ASSERT(_expr)                                                         \
do {                                                                         \
    cl_int _err = _expr;                                                       \
    if (_err == CL_SUCCESS)                                                    \
    break;                                                                   \
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
    abort();                                                                   \
} while (0)


class OpenCLDeviceIDWrapper;

class OpenCLPlatformWrapper {
public:
	cl_platform_id platformID;
	string name;
};

class OpenCLContextWrapper {
public:
	cl_context              context;
	string                  name;
	OpenCLPlatformWrapper*  parentPlatform;
	vector<OpenCLDeviceIDWrapper*> childrenDevices;
};

class OpenCLProgramWrapper {
	public:
	cl_program             program;
	string                 name;
	OpenCLContextWrapper*  parentContext;
};

class OpenCLKernelWrapper {
	public:
	cl_kernel              kernel;
	string                 name;
	OpenCLProgramWrapper*  parentProgram;
};

class OpenCLDeviceIDWrapper {
	public:

	cl_device_id           deviceID;	
	string                 name;
	string vendor;
	string deviceVersion;
	string driverVersion;
	string profile;
	unsigned int maxComputeUnits;
	unsigned int maxClockFrequency;
	unsigned long long globalMemSize;
	vector<string> deviceExtensions;
	bool imageSupport;
	size_t maxWorkgroupSize;

	OpenCLPlatformWrapper* parentPlatform;
};

class OpenCLCommandQueueWrapper {
public:
	cl_command_queue       commandQueue;
	string                 name;
	OpenCLDeviceIDWrapper* parentDeviceID;
	OpenCLContextWrapper*  parentContext;
};

class OpenCLMemoryObjectWrapper {
public:
	cl_mem                 memoryObject;
	string                 name;
	OpenCLContextWrapper*  parentContext;
	cl_mem_flags           flags;
	size_t                 bufferSize;
	void*                  hostPtr;
	int					   width;
	int					   height;
	string                 type; //usually == 'image'	
	cl_image_format        imageFormat;   //CL_RGBA, etc
};

class OpenCLSamplerWrapper {
public:
	cl_sampler             sampler;
	string                 name;
	OpenCLContextWrapper*  parentContext;
	cl_bool                normalized;
	cl_addressing_mode     addressingMode;
	cl_filter_mode         filterMode;
};

#endif