__kernel void rgbToLAB(__read_only image2d_t srcImg,
	                   __write_only image2d_t dstImg,
					   sampler_t sampler) {
	
	int2 coords = (int2) (get_global_id(0),get_global_id(1));
		
	//working: RGB -> LogLMS
	/*
	float4 buf = read_imagef(srcImg,sampler,coords);
	float4 buf2 = (float4)(0.0,0.0,0.0,0.0);
	
	float matRGBToLMS[9] = {0.3811, 0.5783, 0.0402,
						    0.1967, 0.7244, 0.0782,
						    0.0241, 0.1228, 0.8444};
	
	buf2.x = matRGBToLMS[0]*buf.x + matRGBToLMS[1]*buf.y + matRGBToLMS[2]*buf.z;
	buf2.y = matRGBToLMS[3]*buf.x + matRGBToLMS[4]*buf.y + matRGBToLMS[5]*buf.z;
	buf2.z = matRGBToLMS[6]*buf.x + matRGBToLMS[7]*buf.y + matRGBToLMS[8]*buf.z;
	buf2.w = 1.0;	

	
	//log LMS color space
	buf = log10(buf2);

	//do some simple clamping
	if(buf.x < -12)
		buf.x = -12;
	if(buf.y < -12)
		buf.y = -12;
	if(buf.z < -12)
		buf.z = -12;

	buf.w=1.0;
	
	write_imagef(dstImg,coords,buf);
	*/

	float4 buf = read_imagef(srcImg,sampler,coords);
	float4 buf2 = (float4)(0.0,0.0,0.0,0.0);
	
	float matRGBToLMS[9] = {0.3811, 0.5783, 0.0402,
						    0.1967, 0.7244, 0.0782,
						    0.0241, 0.1228, 0.8444};
	
	buf2.x = matRGBToLMS[0]*buf.x + matRGBToLMS[1]*buf.y + matRGBToLMS[2]*buf.z;
	buf2.y = matRGBToLMS[3]*buf.x + matRGBToLMS[4]*buf.y + matRGBToLMS[5]*buf.z;
	buf2.z = matRGBToLMS[6]*buf.x + matRGBToLMS[7]*buf.y + matRGBToLMS[8]*buf.z;
	buf2.w = 1.0;	

	
	//log LMS color space
	buf = log10(buf2);

	//do some simple clamping
	if(buf.x < -12)
		buf.x = -12;
	if(buf.y < -12)
		buf.y = -12;
	if(buf.z < -12)
		buf.z = -12;

	buf.w=1.0;
	//begin experiment
	/*
	//LAB color space
	// A = []
	//    1/sqrt(3),0,0
	//    0,1/sqrt(6),0
	//    0,0,1/sqrt(2)
	//
	//B = []
	//	1,1,1
	//	1,1,-2
	//	1,-1,0

	A*B = 0.577350, 0.577350, 0.577350,
	      0.408248, 0.408248, -0.816496,
	      0.707106, -0.707106, 0

	B*A = 0.57735, 0.408248, 0.707106,
	      0.57735, 0.408248, -1.414212,
	      0.57735, -0.408248, 0.0}
	*/

	float matLogLMSToLAB[9] = {0.577350, 0.577350, 0.577350,
							   0.408248, 0.408248, -0.816496,
							   0.707106, -0.707106, 0};
   
	
	buf2.x = matLogLMSToLAB[0]*buf.x + matLogLMSToLAB[1]*buf.y + matLogLMSToLAB[2]*buf.z;
	buf2.y = matLogLMSToLAB[3]*buf.x + matLogLMSToLAB[4]*buf.y + matLogLMSToLAB[5]*buf.z;
	buf2.z = matLogLMSToLAB[6]*buf.x + matLogLMSToLAB[7]*buf.y + matLogLMSToLAB[8]*buf.z; 
	
	buf2.w = 1.0;

	write_imagef(dstImg,coords,buf2);
	
	
}

__kernel void colorTransfer(__read_only image2d_t srcImg,
	                        __write_only image2d_t dstImg,
							sampler_t sampler,
							constant float* data) 
{
								
	// (l*) = l- meanL;


	//...same for A,B, etc

	// l' = (stddevTargetL)
	//      (-------------) * (l*)
	//      (stddevSourceL)

	//...same for A,B,etc

	//note: format of data[] = {stddevSourceL,stddevSourceA,stddevSourceB,stddevTargetL,stddevTargetA,stddevTargetB,sourceMeanL,sourceMeanA,sourceMeanB,targetMeanL,targetMeanA,targetMeanB};
	
	float stddevSourceL = data[0];
	float stddevSourceA = data[1];
	float stddevSourceB = data[2];

	float stddevTargetL = data[3];
	float stddevTargetA = data[4];
	float stddevTargetB = data[5];

	float sourceMeanL = data[6];
	float sourceMeanA = data[7];
	float sourceMeanB = data[8];

	float targetMeanL = data[9];
	float targetMeanA = data[10];
	float targetMeanB = data[11];
	
	int2 coords = (int2) (get_global_id(0),get_global_id(1));	

	float4 buf = read_imagef(srcImg,sampler,coords); //buf is in LAB color space
	float4 buf2 = (float4) (0.0,0.0,0.0,0.0);

	//do the color transfer in LAB color space
	buf2.x = (stddevTargetL/stddevSourceL) * (buf.x - sourceMeanL); //L
	buf2.y = (stddevTargetA/stddevSourceA) * (buf.y - sourceMeanA); //A
	buf2.z = (stddevTargetB/stddevSourceB) * (buf.z - sourceMeanB); //B

	buf2.x += targetMeanL;
	buf2.y += targetMeanA;
	buf2.z += targetMeanB;

	buf2.w = 1.0;

	float matLABToLogLMS[9] = {0.577350, 0.408248, 0.707106,
							   0.577350, 0.408248, -0.707106,
							   0.577350, -0.816496, 0};

	buf.x = matLABToLogLMS[0]*buf2.x + matLABToLogLMS[1]*buf2.y + matLABToLogLMS[2]*buf2.z;
	buf.y = matLABToLogLMS[3]*buf2.x + matLABToLogLMS[4]*buf2.y + matLABToLogLMS[5]*buf2.z;
	buf.z = matLABToLogLMS[6]*buf2.x + matLABToLogLMS[7]*buf2.y + matLABToLogLMS[8]*buf2.z;
	buf.w=1;	
	
	//logLMS to LMS
	buf2 = exp10(buf);
	buf2.w =1;
	
	float matLMSToRGB[9] = {4.4679, -3.5873, 0.1193,
						   -1.2186, 2.3809, -0.1624,
						    0.0497, -0.2439, 1.2045};	
	
	buf.x = matLMSToRGB[0]*buf2.x + matLMSToRGB[1]*buf2.y + matLMSToRGB[2]*buf2.z;
	buf.y = matLMSToRGB[3]*buf2.x + matLMSToRGB[4]*buf2.y + matLMSToRGB[5]*buf2.z;
	buf.z = matLMSToRGB[6]*buf2.x + matLMSToRGB[7]*buf2.y + matLMSToRGB[8]*buf2.z;
	buf.w = 1;

	write_imagef(dstImg,coords,buf);

}