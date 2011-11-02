// CLInfo.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>

#ifdef __APPLE__
	#include <opencl/cl.h>
	#include <opencl/cl_platform.h>
#else
	#include <CL/cl.h>
	#include <CL/cl_platform.h>
#endif

#include "FreeImage.h"

#include "CLManager.hpp"

using namespace std;

const enum OPENCL_SYSTEM_TO_USE {OPENCL_CPU_SYSTEM,OPENCL_GPU_SYSTEM};

int main(int argc, char** argv)
{
	CLManager clManager;
	
	clManager.initCL();
	clManager.printDebugInfo();

	const int systemPreference = OPENCL_GPU_SYSTEM; //change this variable (and only this variable) to switch between CPU and GPU usage

	OpenCLContextWrapper* gpuContext=NULL;
	OpenCLContextWrapper* cpuContext=NULL;	

	OpenCLContextWrapper* contextToUse=NULL;
	OpenCLCommandQueueWrapper* commandQueueToUse=NULL;	
    
    cpuContext = clManager.getCLContext("intel", "cpu");
	if(!cpuContext) {
		cpuContext = clManager.getCLContext("amd", "cpu");
	}

#ifdef __APPLE__
	gpuContext = clManager.getCLContext("apple", "gpu");
	
#else
    gpuContext = clManager.getCLContext("nvidia", "gpu");
	if(!gpuContext) {
		gpuContext = clManager.getCLContext("radeon", "gpu");
	}
	
#endif

	if(!gpuContext && ! cpuContext) {
		cout << "Error! Could not get a CPU or GPU context, aborting." << endl;
		getchar();
		return 0;
	}

	//attempt GPU
	OpenCLCommandQueueWrapper* gpuCommandQueue = NULL;

    if(gpuContext) {
		gpuCommandQueue = clManager.getCLCommandQueue("geforce");
		if(!gpuCommandQueue) {
			gpuCommandQueue = clManager.getCLCommandQueue("radeon");
			if(!gpuCommandQueue) {
				cout << "Could not get gpuCommandQueue" << endl;
				getchar();
				return 1;
			}
		}		
	}

	//attempt CPU
	OpenCLCommandQueueWrapper* cpuCommandQueue = NULL;
	
	if(cpuContext) {
		cpuCommandQueue = clManager.getCLCommandQueue("intel");
		if(!cpuCommandQueue) {
			cpuCommandQueue = clManager.getCLCommandQueue("amd");
			if(!cpuCommandQueue) {
				cout << "Could not get cpuCommandQueue" << endl;
				getchar();
				return 1;
			}
		}		
	}

	switch(systemPreference) {
		case OPENCL_CPU_SYSTEM:
			if(cpuContext && cpuCommandQueue) {
				contextToUse=cpuContext;
				commandQueueToUse=cpuCommandQueue;
			}
			break;
		case OPENCL_GPU_SYSTEM:
			if(gpuContext && gpuCommandQueue) {
				contextToUse=gpuContext;
				commandQueueToUse=gpuCommandQueue;
			}
			break;
	}

	if(!contextToUse || !commandQueueToUse) {
		cout <<"Error initializing OpenCL system." << endl;
		getchar();
		return 0;
	}

	cout << "=== Begin OpenCL Execution ===" << endl;
	cout << "Using " << contextToUse->name << endl;
	cout << "Using " << commandQueueToUse->name << endl;

	cl_int errorCode=-1;			
	
	//create program
	OpenCLProgramWrapper* colorTransferProgram = clManager.createCLProgram(contextToUse,contextToUse->childrenDevices[0],"ColorTransfer.cl","colortransfer_program");
	if(!colorTransferProgram){
		cout << "Error creating program, aborting." << endl;
		getchar();
		return 1;
	}

	//create kernels
	OpenCLKernelWrapper* rgbToLABKernel = clManager.createCLKernel(colorTransferProgram, "rgbToLAB");
	if(!rgbToLABKernel){
		cout << "Error creating kernel, aborting." << endl;
		getchar();
		return 1;
	}
	OpenCLKernelWrapper* colorTransferKernel = clManager.createCLKernel(colorTransferProgram, "colorTransfer");
	if(!colorTransferKernel){
		cout << "Error creating kernel, aborting." << endl;
		getchar();
		return 1;
	}

	cout << endl;

	//samplers
	OpenCLSamplerWrapper* samplerWrapper = clManager.createCLSampler(contextToUse,CL_FALSE,CL_ADDRESS_CLAMP_TO_EDGE,CL_FILTER_NEAREST,"sampler1024clampnearest");

	//load image(s)			
		 
	if(argc < 3)
	{
		cout << "Error! Must specify at least 2 arguments. \nFor example, \"clcolortransfer imageA.png imageB.png\"" << endl;
		getchar();
		return 0;
	}

	OpenCLMemoryObjectWrapper* inputImageSource = clManager.loadImage(contextToUse, argv[1]);		
		
	if (!inputImageSource || (inputImageSource->memoryObject == 0))
	{
		std::cerr << "Error loading: " << std::string(argv[1]) << std::endl;
		//Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		getchar();
		return 1;
	}
		
	OpenCLMemoryObjectWrapper* inputImageTarget = clManager.loadImage(contextToUse, argv[2]);		
		
	if (!inputImageTarget || (inputImageTarget->memoryObject == 0))
	{
		std::cerr << "Error loading: " << std::string(argv[2]) << std::endl;
		//Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		getchar();
		return 1;
	}
		
	//intermediate buffers needed to store the LAB colorspace image with 32bits per channel
	OpenCLMemoryObjectWrapper* inputImageSource_LAB32 = clManager.createBlankCLImage(contextToUse,CL_MEM_READ_WRITE,CL_RGBA,CL_FLOAT,1024,1024,"inputImageSource_LAB32");
	OpenCLMemoryObjectWrapper* inputImageTarget_LAB32 = clManager.createBlankCLImage(contextToUse,CL_MEM_READ_WRITE,CL_RGBA,CL_FLOAT,1024,1024,"inputImageTarget_LAB32");

	//store the final image in the r32g32b32a32 format (will be downsampled when saving to disk)
	OpenCLMemoryObjectWrapper* outputImage_RGB32 = clManager.createBlankCLImage(contextToUse,CL_MEM_READ_WRITE,CL_RGBA,CL_FLOAT,1024,1024,"outputImage_RGB32");

	cout << endl;
		
	//first process the source image
	errorCode  = clSetKernelArg(rgbToLABKernel->kernel,0,sizeof(cl_mem),&inputImageSource->memoryObject);
	errorCode |= clSetKernelArg(rgbToLABKernel->kernel,1,sizeof(cl_mem),&inputImageSource_LAB32->memoryObject);
	errorCode |= clSetKernelArg(rgbToLABKernel->kernel,2,sizeof(cl_sampler),&samplerWrapper->sampler);

	if(errorCode != CL_SUCCESS) {
		cout << "Error setting kernel arguments in "<< rgbToLABKernel->name << ", aborting." << endl;
		return 1;
	}		

	//dimensions of CL workload 
	size_t globalWorkSize[2] = {1024,1024}; // (1024x1024)
	size_t localWorkSize[2] = {1,1};	    // 1 pixel per work item

	//rock and roll is here to stay
	CL_ASSERT(clEnqueueNDRangeKernel(commandQueueToUse->commandQueue,rgbToLABKernel->kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL));
		
	//now do the same RGB->LAB conversion for the target image
	errorCode  = clSetKernelArg(rgbToLABKernel->kernel,0,sizeof(cl_mem),&inputImageTarget->memoryObject);
	errorCode |= clSetKernelArg(rgbToLABKernel->kernel,1,sizeof(cl_mem),&inputImageTarget_LAB32->memoryObject);
	errorCode |= clSetKernelArg(rgbToLABKernel->kernel,2,sizeof(cl_sampler),&samplerWrapper->sampler);

	if(errorCode != CL_SUCCESS) {
		cout << "Error setting kernel arguments in "<< rgbToLABKernel->name << ", aborting." << endl;
		return 1;
	}
		
	CL_ASSERT(clEnqueueNDRangeKernel(commandQueueToUse->commandQueue,rgbToLABKernel->kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL));

	float* fbufferSource = new float[inputImageSource_LAB32->width * inputImageSource_LAB32->width * 4];
	float* fbufferTarget = new float[inputImageSource_LAB32->width * inputImageSource_LAB32->width * 4];
		
	size_t origin[3] = {0,0,0};
	size_t region[3] = {inputImageSource_LAB32->width,inputImageSource_LAB32->height,1};	
		
	errorCode = clEnqueueReadImage(commandQueueToUse->commandQueue,inputImageSource_LAB32->memoryObject,CL_TRUE,origin,region,0,0,fbufferSource,0,NULL,NULL);
	
	if(errorCode != CL_SUCCESS) {
		cout << "Error doing clEnqueueReadImage(inputImageSource_LAB32) in main()" << endl;
	}

	errorCode = clEnqueueReadImage(commandQueueToUse->commandQueue,inputImageTarget_LAB32->memoryObject,CL_TRUE,origin,region,0,0,fbufferTarget,0,NULL,NULL);
	
	if(errorCode != CL_SUCCESS) {
		cout << "Error doing clEnqueueReadImage(inputImageTarget_LAB32) in main()"<< endl;
	}

	float sourceMeanL,sourceMeanA,sourceMeanB,targetMeanL,targetMeanA,targetMeanB;
	sourceMeanL=sourceMeanA=sourceMeanB=targetMeanL=targetMeanA=targetMeanB=0.0;
		
	//compute means
	for(unsigned int i=0; i < (inputImageSource_LAB32->width * inputImageSource_LAB32->height*4); i+=4) {
		sourceMeanL += fbufferSource[i];
		sourceMeanA += fbufferSource[i+1];
		sourceMeanB += fbufferSource[i+2];
	}

	sourceMeanL /= (inputImageSource_LAB32->width * inputImageSource_LAB32->height );
	sourceMeanA /= (inputImageSource_LAB32->width * inputImageSource_LAB32->height );
	sourceMeanB /= (inputImageSource_LAB32->width * inputImageSource_LAB32->height );
	
	cout << "sourceMeanL: " << sourceMeanL << endl;
	cout << "sourceMeanA: " << sourceMeanA << endl;
	cout << "sourceMeanB: " << sourceMeanB << endl <<endl;

	float totalSquaredSourceL=0.0f;
	float totalSquaredSourceA=0.0f;
	float totalSquaredSourceB=0.0f;

	float stddevSourceL=0.0f;
	float stddevSourceA=0.0f;
	float stddevSourceB=0.0f;

	//compute standard deviation
	for(unsigned int i=0; i < (inputImageSource_LAB32->width * inputImageSource_LAB32->height*4); i+=4) {
		//subtract means
		fbufferSource[i] -= sourceMeanL;
		fbufferSource[i+1] -= sourceMeanA;
		fbufferSource[i+2]-= sourceMeanB;	

		//compute squares
		fbufferSource[i] *= fbufferSource[i];
		fbufferSource[i+1] *= fbufferSource[i+1];
		fbufferSource[i+2] *= fbufferSource[i+2];

		//add squares
		totalSquaredSourceL += fbufferSource[i];
		totalSquaredSourceA += fbufferSource[i+1];
		totalSquaredSourceB += fbufferSource[i+2];	
	}

	stddevSourceL = sqrt(totalSquaredSourceL/((inputImageSource_LAB32->width * inputImageSource_LAB32->height)-1));
	stddevSourceA = sqrt(totalSquaredSourceA/((inputImageSource_LAB32->width * inputImageSource_LAB32->height)-1));
	stddevSourceB = sqrt(totalSquaredSourceB/((inputImageSource_LAB32->width * inputImageSource_LAB32->height)-1));

	cout << "stddevSourceL: " << stddevSourceL << endl;
	cout << "stddevSourceA: " << stddevSourceA << endl;
	cout << "stddevSourceB: " << stddevSourceB << endl <<endl;

	//now do the same for the target

	//compute means
	for(unsigned int i=0; i < (inputImageTarget_LAB32->width * inputImageTarget_LAB32->height*4); i+=4) {
		targetMeanL += fbufferTarget[i];
		targetMeanA += fbufferTarget[i+1];
		targetMeanB += fbufferTarget[i+2];
	}

	targetMeanL /= (inputImageTarget_LAB32->width * inputImageTarget_LAB32->height );
	targetMeanA /= (inputImageTarget_LAB32->width * inputImageTarget_LAB32->height );
	targetMeanB /= (inputImageTarget_LAB32->width * inputImageTarget_LAB32->height );
	
	cout << "targetMeanL: " << targetMeanL << endl;
	cout << "targetMeanA: " << targetMeanA << endl;
	cout << "targetMeanB: " << targetMeanB << endl <<endl;

	float totalSquaredTargetL=0.0f;
	float totalSquaredTargetA=0.0f;
	float totalSquaredTargetB=0.0f;

	float stddevTargetL=0.0f;
	float stddevTargetA=0.0f;
	float stddevTargetB=0.0f;

	//compute standard deviation
	for(unsigned int i=0; i < (inputImageTarget_LAB32->width * inputImageTarget_LAB32->height*4); i+=4) {
		//subtract means
		fbufferTarget[i] -= sourceMeanL;
		fbufferTarget[i+1] -= sourceMeanA;
		fbufferTarget[i+2]-= sourceMeanB;	

		//compute squares
		fbufferTarget[i] *= fbufferTarget[i];
		fbufferTarget[i+1] *= fbufferTarget[i+1];
		fbufferTarget[i+2] *= fbufferTarget[i+2];

		//add squares
		totalSquaredTargetL += fbufferTarget[i];
		totalSquaredTargetA += fbufferTarget[i+1];
		totalSquaredTargetB += fbufferTarget[i+2];	
	}

	stddevTargetL = sqrt(totalSquaredTargetL/((inputImageTarget_LAB32->width * inputImageTarget_LAB32->height)-1));
	stddevTargetA = sqrt(totalSquaredTargetA/((inputImageTarget_LAB32->width * inputImageTarget_LAB32->height)-1));
	stddevTargetB = sqrt(totalSquaredTargetB/((inputImageTarget_LAB32->width * inputImageTarget_LAB32->height)-1));

	cout << "stddevTargetL: " << stddevTargetL << endl;
	cout << "stddevTargetA: " << stddevTargetA << endl;
	cout << "stddevTargetB: " << stddevTargetB << endl;

	//ok, all required data has been calculated, lets do the color transfer

	float data[] = {stddevSourceL,stddevSourceA,stddevSourceB,stddevTargetL,stddevTargetA,stddevTargetB,sourceMeanL,sourceMeanA,sourceMeanB,targetMeanL,targetMeanA,targetMeanB};
	
	OpenCLMemoryObjectWrapper* colorTransferDataBuffer;	
	colorTransferDataBuffer =  clManager.createCLMemoryObject(contextToUse,
											CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											sizeof(float)*12,
											data,
											"colorTransferDataBuffer");

	errorCode  = clSetKernelArg(colorTransferKernel->kernel,0,sizeof(cl_mem),&inputImageSource_LAB32->memoryObject);
	errorCode |= clSetKernelArg(colorTransferKernel->kernel,1,sizeof(cl_mem),&outputImage_RGB32->memoryObject);
	errorCode |= clSetKernelArg(colorTransferKernel->kernel,2,sizeof(cl_sampler),&samplerWrapper->sampler);
	errorCode |= clSetKernelArg(colorTransferKernel->kernel,3,sizeof(cl_mem) ,&colorTransferDataBuffer->memoryObject);

	if(errorCode != CL_SUCCESS) {
		cout << "Error setting kernel arguments in kernel "<< colorTransferKernel->name << ", aborting." << endl;
		getchar();
		return 1;
	}

	CL_ASSERT(clEnqueueNDRangeKernel(commandQueueToUse->commandQueue,colorTransferKernel->kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL));	

	clManager.saveImage(inputImageSource_LAB32,"outputSourceLAB.png", commandQueueToUse);
	clManager.saveImage(inputImageTarget_LAB32,"outputTargetLAB.png", commandQueueToUse);
	clManager.saveImage(outputImage_RGB32,"output.png", commandQueueToUse);


	cout << "===The Ending===" << endl;

	getchar();

	return 0;
}


