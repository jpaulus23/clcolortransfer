#include "CLManager.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <algorithm>

#include "FreeImage.h"



//stupid helper function
std::string toUpperCase(const std::string & s)
{
    std::string ret(s.size(), char());
    for(unsigned int i = 0; i < s.size(); ++i)
        ret[i] = (s[i] <= 'z' && s[i] >= 'a') ? s[i]-('a'-'A') : s[i];
    return ret;
}

CLManager::CLManager() {

}

CLManager::~CLManager() {

}

void CLManager::initCL() {
	//query platforms
	createCLPlatforms();
	
	if(CLPlatforms.size() == 0)
		return ;	

	//now contexts
	createCLContexts();

	if(CLContexts.size() == 0)
		return;
	
	//now devices
	for(unsigned int i=0;i<CLContexts.size(); i++) {
		createCLDevices(CLContexts[i]);
	}	

	createCLCommandQueues();
}

void CLManager::printDebugInfo() {
	//platforms
	cout << "===CL Platforms Found===" << endl;

	for(unsigned int i=0; i< CLPlatforms.size(); i++) {
		cout << " - " << CLPlatforms[i]->name << endl;
	}
	cout << endl;

	cout << "===CL Contexts Created===" << endl;
	for(unsigned int i=0; i< CLContexts.size(); i++) {
		cout << " - " << CLContexts[i]->name << endl;
	}
	cout << endl;

	cout << "===CL Devices Found===" << endl;
	for(unsigned int i=0; i< CLDevices.size(); i++) {
		cout << " - " << CLDevices[i]->name << endl;
		cout << "    * Device Version: " << CLDevices[i]->deviceVersion << endl;
		cout << "    * Driver Version: " << CLDevices[i]->driverVersion << endl;
		cout << "    * Global Memory: " << CLDevices[i]->globalMemSize << " bytes" << endl;
		cout << "    * Max Frequency: " << CLDevices[i]->maxClockFrequency << " Mhz" << endl;
		cout << "    * Max Compute Units: " << CLDevices[i]->maxComputeUnits << endl;
		cout << "    * Max Workgroup Size: " << CLDevices[i]->maxWorkgroupSize << endl;
		cout << "    * Profile: " << CLDevices[i]->profile << endl;
		
		if(CLDevices[i]->imageSupport)
			cout << "    * Image Support: Yes" << endl;
		else
			cout << "    * Image Support: No" << endl;

		cout << "    * Extensions: " << endl;
		for(unsigned int j=0; j < CLDevices[i]->deviceExtensions.size(); j++) {		
			cout << "       * " << CLDevices[i]->deviceExtensions[j] << endl;
		}
		cout << endl;
	}

	cout << "===Command Queues Found===" << endl;
	for(unsigned int i=0; i< CLCommandQueues.size(); i++) {
		cout << " - " << CLCommandQueues[i]->name << endl;
	}
	cout << endl;
}

void CLManager::createCLPlatforms() {
	
	cl_uint numPlatforms;
	cl_platform_id* platformIDs;
	size_t size;

	//get num of platforms
	CL_ASSERT(clGetPlatformIDs(0,NULL,&numPlatforms));	
	
	platformIDs = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);

	//now get the platforms
	CL_ASSERT(clGetPlatformIDs(numPlatforms, platformIDs, NULL));	

	//get platform info for each platform
	for(int i = 0; i < numPlatforms; i++) {
		
		CL_ASSERT(clGetPlatformInfo(platformIDs[i],CL_PLATFORM_NAME,0,NULL,&size));
		char* name = (char *) alloca(sizeof(char) * size);
		CL_ASSERT(clGetPlatformInfo(platformIDs[i],CL_PLATFORM_NAME,size,name,NULL));
		
		OpenCLPlatformWrapper* platform = new OpenCLPlatformWrapper;
		platform->platformID =  platformIDs[i];
		platform->name = name;
		
		

			CLPlatforms.push_back(platform);
	}

	cout << endl;	
}

void CLManager::createCLContexts() {	

	//create a context for each platform
	for(int i=0;i < CLPlatforms.size(); i++) {	
		OpenCLContextWrapper* contextWrapper = new OpenCLContextWrapper();
		
		//attempt GPU context creation
		contextWrapper->context = createCLContext(CLPlatforms[i],string("gpu"));
		if(contextWrapper->context)	{
			contextWrapper->name = CLPlatforms[i]->name + " - GPU Context";
			contextWrapper->parentPlatform = CLPlatforms[i];
		}
		else {
			//attempt CPU context creation
			contextWrapper->context = createCLContext(CLPlatforms[i],string("cpu"));			
			if(contextWrapper->context)	{
				contextWrapper->name = CLPlatforms[i]->name + " - CPU Context";
				contextWrapper->parentPlatform = CLPlatforms[i];
			}
		}
		
		if(contextWrapper->context) {
			CLContexts.push_back(contextWrapper);			
		}
	}	
}

cl_context CLManager::createCLContext(OpenCLPlatformWrapper* clPlatformWrapper,string contextType) {
	cl_context context;
	cl_int errorCode=-1;

	cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) clPlatformWrapper->platformID,
		0 //last element must be zero
	};
	string upperCaseType = toUpperCase(contextType);

	if(upperCaseType.compare("GPU") == 0)	
		context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_GPU,NULL,NULL,&errorCode);
	else if(upperCaseType.compare("CPU") == 0)
		context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_CPU,NULL,NULL,&errorCode);
	else if(upperCaseType.compare("ACCELERATOR") == 0)
		context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_ACCELERATOR,NULL,NULL,&errorCode);

	if(errorCode != CL_SUCCESS) {
		return NULL;
	}
	
	return context;
}

void CLManager::createCLDevices(OpenCLContextWrapper* contextWrapper) {
	
	cl_device_id* devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;
	int numDevices=0;

	CL_ASSERT(clGetContextInfo(contextWrapper->context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize));	
	
	if(deviceBufferSize <=0) {
		cout << " Error! No devices available for context" <<  contextWrapper->name << endl;
		return;
	}

	numDevices = deviceBufferSize / sizeof(cl_device_id);
	devices = new cl_device_id[numDevices];
	
	//cout << numDevices << " devices found:" << endl;	

	CL_ASSERT(clGetContextInfo(contextWrapper->context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL));

	for(int i=0; i < numDevices; i++) {

		OpenCLDeviceIDWrapper* deviceWrapper = new OpenCLDeviceIDWrapper();

		deviceWrapper->deviceID = devices[i];
		deviceWrapper->parentPlatform = contextWrapper->parentPlatform;

		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;		

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));	
		deviceWrapper->name = buffer;
		
		//remove leading white space from name (damn you Intel!)
		unsigned int pos=0;		
		do{
			pos = deviceWrapper->name.find(" ");
			if( pos ==0 )
			deviceWrapper->name.erase( pos, 1 );
		}while(pos==0 && pos != string::npos);

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));	
		deviceWrapper->vendor = buffer;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));		
		deviceWrapper->deviceVersion = buffer;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));		
		deviceWrapper->driverVersion = buffer;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_PROFILE, sizeof(buffer), buffer, NULL));		
		deviceWrapper->profile = buffer;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));		
		deviceWrapper->maxComputeUnits = (unsigned int)buf_uint;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));		
		deviceWrapper->maxClockFrequency = (unsigned int)buf_uint;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));		
		deviceWrapper->globalMemSize = (unsigned long long)buf_ulong;

		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL));	
		
		char *p = strtok(buffer, " ");

		while (p) {			
			deviceWrapper->deviceExtensions.push_back(string(p));
			p = strtok(NULL, " ");
		}		

		cl_bool imageSupport = CL_FALSE;
		
		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL));	
		deviceWrapper->imageSupport = imageSupport;
        
        if(!deviceWrapper->imageSupport) {
            cout << "Error! Image support was not found on CL device " << deviceWrapper->name << ". Aborting :(" << endl;
            exit(0);
        }
            

		size_t maxWorkgroupSize;
		CL_ASSERT(clGetDeviceInfo(deviceWrapper->deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkgroupSize, NULL));	
		deviceWrapper->maxWorkgroupSize = maxWorkgroupSize;
		
		//context keeps a list of all devices it uses
		contextWrapper->childrenDevices.push_back(deviceWrapper);
		
		//global vector of all devices
		CLDevices.push_back(deviceWrapper);

	}	

	delete [] devices;	

}

bool CLManager::deviceSupportsExtension(OpenCLDeviceIDWrapper* device, string extension) {
	if(!device || !device->deviceID)
		return false;
	size_t found;

	for(unsigned int i=0; i < device->deviceExtensions.size();i++) {
		found = toUpperCase(device->deviceExtensions[i]).find(toUpperCase(extension));
		if(found != string::npos)
			return true;
	}
	return false;
}



void CLManager::createCLCommandQueues() {
	
	// create all possible command queues
	for(unsigned int i = 0; i < CLContexts.size(); i++)
		for(unsigned int j=0; j < CLDevices.size(); j++)
			if(CLContexts[i]->parentPlatform->platformID == CLDevices[j]->parentPlatform->platformID) {
				cl_command_queue queue = createCLCommandQueue(CLContexts[i],CLDevices[j]);
				if(queue) {
				OpenCLCommandQueueWrapper* queueWrapper = new OpenCLCommandQueueWrapper();
				queueWrapper->commandQueue = queue;
				queueWrapper->name = CLDevices[j]->name + " Command Queue";
				queueWrapper->parentDeviceID= CLDevices[j];
				queueWrapper->parentContext = CLContexts[i];

				CLCommandQueues.push_back(queueWrapper);
				}
			}
}

cl_command_queue CLManager::createCLCommandQueue(OpenCLContextWrapper* contextWrapper, OpenCLDeviceIDWrapper* deviceWrapper) {
	cl_int errorCode=-1;
	cl_command_queue queue;
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	queue = clCreateCommandQueue(contextWrapper->context,deviceWrapper->deviceID,NULL,&errorCode);
	
	if(errorCode == CL_SUCCESS)
		return queue;
	else return NULL;
}

OpenCLProgramWrapper* CLManager::createCLProgram(OpenCLContextWrapper* contextWrapper, OpenCLDeviceIDWrapper* deviceWrapper, const char* fileName, const char* programName) {
	
	OpenCLProgramWrapper* programWrapper = new OpenCLProgramWrapper();
	cl_int errorCode=-1;

	ifstream kernelFile(fileName, ios::in);

	if(!kernelFile.is_open()) {
		cout << "Error opening file " << fileName <<endl;
		return NULL;
	}

	ostringstream oss;
	oss << kernelFile.rdbuf();
	string srcStdStr = oss.str();
	const char* srcStr = srcStdStr.c_str();

	programWrapper->program = clCreateProgramWithSource(contextWrapper->context,1,(const char**)&srcStr,NULL,NULL);
	
	if(programWrapper->program == NULL) {
		cout << "Failed to create CL program from source given in " << fileName << endl;
		return NULL;
	}

	errorCode = clBuildProgram(programWrapper->program,0,NULL,NULL,NULL,NULL);
	
	if(errorCode != CL_SUCCESS) {
		char buildLog[16384];
		clGetProgramBuildInfo(programWrapper->program,deviceWrapper->deviceID,CL_PROGRAM_BUILD_LOG,sizeof(buildLog),buildLog,NULL);
		cout << "Error in kernel: " << endl << buildLog << endl << endl;

		clReleaseProgram(programWrapper->program);
		delete programWrapper;
		return NULL;
	}

	if(programName)
		programWrapper->name = programName;
	else
		programWrapper->name = fileName;

	programWrapper->parentContext = contextWrapper;
	
	CLPrograms.push_back(programWrapper);
	cout << "Created program named " << programName << endl;
	return programWrapper;
}

OpenCLKernelWrapper*  CLManager::createCLKernel(OpenCLProgramWrapper* programWrapper, const char* kernelName) {
	
	cl_kernel kernel = clCreateKernel(programWrapper->program,kernelName,NULL);
	
	if(kernel == NULL) {
		cout << "Failed to create CL kernel named " << kernelName << endl;
		return NULL;
	}
	
	OpenCLKernelWrapper* kernelWrapper = new OpenCLKernelWrapper();
	kernelWrapper->kernel = kernel;
	kernelWrapper->name = kernelName;
	kernelWrapper->parentProgram = programWrapper;

	CLKernels.push_back(kernelWrapper);
	cout << "Created kernel named " << kernelName << endl;
	
	return kernelWrapper;

}

OpenCLMemoryObjectWrapper* CLManager::createCLMemoryObject(OpenCLContextWrapper* contextWrapper,
														cl_mem_flags flags,
														size_t bufferSize,
														void* hostPtr,
														char* name) 
{
    cl_mem memObj;
	if(flags & CL_MEM_READ_ONLY ) 
        memObj = clCreateBuffer(contextWrapper->context,flags,bufferSize,hostPtr,NULL);
    else if(flags & CL_MEM_READ_WRITE)
        memObj = clCreateBuffer(contextWrapper->context,flags,bufferSize,NULL,NULL);
	
	if(memObj == NULL) {
		cout << "Error creating buffer named " << name << endl;
		return NULL;
	}

	OpenCLMemoryObjectWrapper* memObjWrapper = new OpenCLMemoryObjectWrapper();

	memObjWrapper->memoryObject = memObj;
	memObjWrapper->name = name;
	memObjWrapper->parentContext = contextWrapper;
	return memObjWrapper;
}

// name (string) based getters
OpenCLPlatformWrapper* CLManager::getCLPlatform(string name) {
	for(unsigned int i = 0; i < CLPlatforms.size(); i++) {
		if(toUpperCase(CLPlatforms[i]->name).find(toUpperCase(name)) != string::npos)
			return CLPlatforms[i];
	}
	return NULL;
}

OpenCLContextWrapper* CLManager::getCLContext(string name, string type) {
	size_t found;
	for(unsigned int i = 0; i < CLContexts.size(); i++) {
		found = toUpperCase(CLContexts[i]->name).find(toUpperCase(name));
		if(found != string::npos) {
			found = toUpperCase(CLContexts[i]->name).find(toUpperCase(name));
			if(found != string::npos) 
				return CLContexts[i];
		}		
	}
	return NULL;
}

OpenCLDeviceIDWrapper* CLManager::getCLDevice(string name) {
	for(unsigned int i = 0; i < CLDevices.size(); i++) {
		if(toUpperCase(CLDevices[i]->name).find(toUpperCase(name)) != string::npos)
			return CLDevices[i];
	}
	return NULL;
}

OpenCLCommandQueueWrapper* CLManager::getCLCommandQueue(string name) {
	for(unsigned int i = 0; i < CLCommandQueues.size(); i++) {
		if(toUpperCase(CLCommandQueues[i]->name).find(toUpperCase(name)) != string::npos)
			return CLCommandQueues[i];
	}
	return NULL;
}

OpenCLKernelWrapper* CLManager::getCLKernel(string name) {
	for(unsigned int i = 0; i < CLKernels.size(); i++) {
		if(toUpperCase(CLKernels[i]->name).find(toUpperCase(name)) != string::npos)
			return CLKernels[i];
	}
	return NULL;
}

OpenCLProgramWrapper* CLManager::getCLProgram(string name) {
	for(unsigned int i = 0; i < CLPrograms.size(); i++) {
		if(toUpperCase(CLPrograms[i]->name).find(toUpperCase(name)) != string::npos)
			return CLPrograms[i];
	}
	return NULL;
}

OpenCLMemoryObjectWrapper* CLManager::getCLMemoryObject(string name) {
	for(unsigned int i = 0; i < CLMemoryObjects.size(); i++) {
		if(toUpperCase(CLMemoryObjects[i]->name).find(toUpperCase(name)) != string::npos)
			return CLMemoryObjects[i];
	}
	return NULL;
}

OpenCLMemoryObjectWrapper* CLManager::loadImage(OpenCLContextWrapper* contextWrapper, char *fileName) {
	
	OpenCLMemoryObjectWrapper* imageWrapper = new OpenCLMemoryObjectWrapper();

    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName, 0);
    FIBITMAP* image = FreeImage_Load(format, fileName);

    // Convert to 32-bit image
    FIBITMAP* temp = image;
    image = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(temp);

    imageWrapper->width = FreeImage_GetWidth(image);
    imageWrapper->height = FreeImage_GetHeight(image);

    char *buffer = new char[imageWrapper->width * imageWrapper->height * 4];
    memcpy(buffer, FreeImage_GetBits(image), imageWrapper->width * imageWrapper->height * 4);

    FreeImage_Unload(image);

    // Create OpenCL image
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;



    cl_int errNum;   
	imageWrapper->memoryObject = clCreateImage2D(contextWrapper->context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            &clImageFormat,
                            imageWrapper->width,
                            imageWrapper->height,
                            0,
                            buffer,
                            &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error creating CL image object" << std::endl;
        return 0;
    }
	imageWrapper->name = string(fileName);
	imageWrapper->type = string("image");
	imageWrapper->imageFormat.image_channel_order = CL_RGBA;
	imageWrapper->imageFormat.image_channel_data_type = CL_UNORM_INT8;

	cout << "Loaded image " << fileName << " successfully." << endl;
	CLMemoryObjects.push_back(imageWrapper);
    return imageWrapper;
}

OpenCLMemoryObjectWrapper* CLManager::createBlankCLImage(OpenCLContextWrapper* contextWrapper, cl_mem_flags const flags, cl_uint const format, cl_uint const dataType, cl_uint const width, cl_uint const height, char* const name) {
	// format = CL_RGBA, CL_RGB, etc
	// type = CL_FLOAT, CL_CL_UNORM_INT8, etc

	OpenCLMemoryObjectWrapper* imageWrapper = new OpenCLMemoryObjectWrapper();
	
	imageWrapper->width = width;
    imageWrapper->height = height;
	imageWrapper->name = string(name);
	imageWrapper->flags = flags;
	imageWrapper->imageFormat.image_channel_data_type = dataType;
	imageWrapper->imageFormat.image_channel_order= format;

	cl_int errNum;   	

	imageWrapper->memoryObject = clCreateImage2D(contextWrapper->context,
                            imageWrapper->flags,
                            &imageWrapper->imageFormat,
                            imageWrapper->width,
                            imageWrapper->height,
                            0,
                            0,
                            &errNum);
	 
	if (errNum != CL_SUCCESS)    {
        std::cerr << "Error creating blank CL image" << std::endl;
		delete imageWrapper;
        return 0;
    }	

	CLMemoryObjects.push_back(imageWrapper);

	cout << "Created blank image " << name << endl;
    return imageWrapper;
}

OpenCLSamplerWrapper* CLManager::createCLSampler(OpenCLContextWrapper* contextWrapper, cl_bool normalized, cl_addressing_mode addressingMode, cl_filter_mode filterMode,char* name) {
	
	cl_int errNum;  
	OpenCLSamplerWrapper* samplerWrapper = new OpenCLSamplerWrapper();

	samplerWrapper->sampler = clCreateSampler(contextWrapper->context,normalized,addressingMode,filterMode,&errNum);
	
	if(errNum != CL_SUCCESS) {
			cout << "Error creating sampler named " << name << endl;
			delete samplerWrapper;
			return NULL;
	}
	samplerWrapper->addressingMode = addressingMode;
	samplerWrapper->filterMode = filterMode;
	samplerWrapper->normalized = normalized;
	samplerWrapper->name = string(name);

	CLSamplers.push_back(samplerWrapper);
	cout << "Created sampler names " << name << endl;
    return samplerWrapper;

}

bool CLManager::saveImage(OpenCLMemoryObjectWrapper* imageWrapper,char* filename, OpenCLCommandQueueWrapper* cqWrapper) {
	
	//save a 128bpp image to a 32bpp file image (typically .png)

	cl_int errNum; 
	
	float* fbuffer=NULL;
	unsigned char* cbuffer=NULL;

	size_t origin[3] = {0,0,0};
	size_t region[3] = {imageWrapper->width,imageWrapper->height,1};	

	switch(imageWrapper->imageFormat.image_channel_data_type) {
	
		case CL_UNORM_INT8:
			cbuffer = new unsigned char[imageWrapper->width * imageWrapper->width * 4];
			errNum = clEnqueueReadImage(cqWrapper->commandQueue,imageWrapper->memoryObject,CL_TRUE,origin,region,0,0,cbuffer,0,NULL,NULL);
	
			if(errNum != CL_SUCCESS) {
				cout << "Error doing clEnqueueReadImage() in CLManager::saveImage() " << endl;
			}
			break;

		case CL_FLOAT:
			//must down sample the 128bpp image to a 32bpp image (4 float --> 4 byte)
			fbuffer = new float[imageWrapper->width * imageWrapper->width *4];
			cbuffer = new unsigned char[imageWrapper->width * imageWrapper->width * 4];

			errNum = clEnqueueReadImage(cqWrapper->commandQueue,imageWrapper->memoryObject,CL_TRUE,origin,region,0,0,fbuffer,0,NULL,NULL);
	
			if(errNum != CL_SUCCESS) {
				cout << "Error doing clEnqueueReadImage() in CLManager::saveImage() " << endl;
			}

			//convert from float value to unsigned char value 
			for(unsigned int i =0; i< imageWrapper->width * imageWrapper->width * 4; i+=4) {
				
				if(fbuffer[i] !=  fbuffer[i]) //check for NaN
					cout << "Found NaN while saving image :(" << endl;
				if(fbuffer[i+1] !=  fbuffer[i+1]) //check for NaN
					cout << "Found NaN while saving image :(" << endl;
				if(fbuffer[i+2] !=  fbuffer[i+2]) //check for NaN
					cout << "Found NaN while saving image :(" << endl;

				//also convert from RGBA to BGRA
				//float R = fbuffer[i];
				//float G = fbuffer[i+1];
				//float B = fbuffer[i+2];
				//float A = 1.0f;
				/*fbuffer[i] = G;
				fbuffer[i+1] = B;
				fbuffer[i+2]=R;
				fbuffer[i+3]=A;*/
				//fbuffer[i] = G;
				//fbuffer[i+1]=R;
				//fbuffer[i+2]=B;



				//red
				int ibuf = fbuffer[i]*255;
		
				if(ibuf>255)
					ibuf=255;
				else if(ibuf <0) {
					ibuf=0;
				}
				cbuffer[i] = (char)(int)(ibuf);

				//green
				ibuf = fbuffer[i+1]*255;
		
				if(ibuf>255)
					ibuf=255;
				else if(ibuf <0) {
					ibuf=0;
				}
				cbuffer[i+1] = (char)(int)(ibuf);

				//blue

				ibuf = fbuffer[i+2]*255;
		
				if(ibuf>255)
					ibuf=255;
				else if(ibuf <0) {
					ibuf=0;
				}
				cbuffer[i+2] = (char)(int)(ibuf);
				
				//alpha
				cbuffer[i+3] = 255; 
			}
			break;
	}

	//errNum = clEnqueueReadImage(cqWrapper->commandQueue,imageWrapper->memoryObject,CL_TRUE,origin,region,0,0,buffer,0,NULL,NULL);
	
	//if(errNum != CL_SUCCESS) {
	//	cout << "Error doing clEnqueueReadImage() in CLManager::saveImage() " << endl;
	//}

	FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
	
	FIBITMAP* image = FreeImage_ConvertFromRawBits((BYTE*)cbuffer,imageWrapper->width,imageWrapper->height,imageWrapper->width*4,32,0xFF000000,0x00FF0000,0x0000FF00);

	return FreeImage_Save(format,image,filename);

}