/*************************************************
Copyright: SIFT::panorama
Author: Sally
Date:2016-07-26
Description: Implement Extract SIFT features.
**************************************************/

#include "sift.h"

/*************************************************
Function:    // SIFT
Description: // Constructor of class SIFT
Calls:		
Inputs:      // src, input image
             // sigma, the parameter of Gaussian
			 // scale, the height of scale space
**************************************************/
SIFT::SIFT(Mat& img,int octave, int scale, double sigma){
	// Initial the parameters
	this->octave = octave;
	this->scale = scale;
	
	// Construct the scale space
	vector<Mat> gauPyr = generateGaussianPyramid(img,octave,scale,sigma);
	this->DoGs = generateDoGPyramid(gauPyr,octave,scale,sigma);

	// TODO:
}

/*************************************************
Function:    convertRGBToGray64F
Description: // convert the input RGB image to a 
                64F Gray image
Inputs:      // src, input image
		     // dst, output image
**************************************************/
void SIFT::convertRGBToGray64F(const Mat& src, Mat& dst){
	// Determine the type of input image
	if(src.channels() == 1){
		cout << "The input image is a gray image." << endl;
		return;
	}

	// Initial the size of output image.
    dst = cvCreateMat(src.rows, src.cols, CV_64F);
	
	// Convert the image 
	for(int x = 0; x < src.rows; x++){
		for(int y = 0; y < src.cols; y++){
			int b = (int)((uchar*)src.data)[x * src.step + src.channels() * y + 0] ;
			int g = (int)((uchar*)src.data)[x * src.step + src.channels() * y + 1] ;
			int r = (int)((uchar*)src.data)[x * src.step + src.channels() * y + 2] ;
			// Scale the range from 0 to 1
			((double*)dst.data)[x * dst.cols + y] = (double)(r + g + b) / (double)(255 * 3);
		}
	}
}


/*************************************************
Function:    upSample
Description: // Use linear interpolation to up sample 
				the input image and convert its result 
				to the output image 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // dst, output image
**************************************************/ 
void SIFT::upSample(const Mat& src, Mat& dst){
	// Determine the type of input image
	if(src.channels() != 1){
		cout << "The input image is not a gray image." << endl;
		return;
	}
	if(src.type() != CV_64F){
		cout << "The bit depth of input image is not CV_64F." << endl;
		return;
	}

	// Initial the size of output image.
	dst = cvCreateMat(src.rows * 2, src.cols * 2, CV_64F);

	// Up sample with  linear interpolation
	for(int srcX = 0; srcX < src.rows - 1; srcX++){
		for (int srcY = 0; srcY < src.cols - 1; srcY++){
			((double*)dst.data)[(srcX * 2) * dst.cols + (srcY * 2)] = ((double*)src.data)[srcX * src.cols + srcY];
			// interpolate x
			double dx = ((double*)src.data)[srcX * src.cols + srcY] + ((double*)src.data)[(srcX + 1) * src.cols + srcY];
			((double*)dst.data)[(srcX * 2 + 1) * dst.cols + (srcY * 2)] = dx / 2.0;
			// interpolate y
			double dy =  ((double*)src.data)[srcX * src.cols + srcY] +  ((double*)src.data)[srcX * src.cols + (srcY + 1)];
			((double*)dst.data)[(srcX * 2) * dst.cols + (srcY * 2 + 1)] = dy / 2.0;
			// interpolate xy
			/*double dxy = ((double*)src.data)[srcX * src.cols + srcY] + ((double*)src.data)[(srcX + 1) * src.cols + (srcY + 1)];
			((double*)dst.data)[(srcX * 2 + 1) * dst.cols + (srcY * 2 + 1)] = dxy / 2.0;*/
			double dxy = ((double*)src.data)[srcX * src.cols + srcY] + ((double*)src.data)[(srcX + 1) * src.cols + srcY]
			+  ((double*)src.data)[srcX * src.cols + (srcY + 1)] + ((double*)src.data)[(srcX + 1) * src.cols + (srcY + 1)];
			((double*)dst.data)[(srcX * 2 + 1) * dst.cols + (srcY * 2 + 1)] = dxy / 4.0;
		}
	}
}
 

/*************************************************
Function:    downSample
Description: // Down sample the input image and convert 
				its result to the output image 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // dst, output image
**************************************************/  
void SIFT::downSample(const Mat& src, Mat& dst){
	// Determine the type of input image
	if(src.channels() != 1){
		cout << "The input image is not a gray image." << endl;
		return;
	}
	if(src.type() != CV_64F){
		cout << "The bit depth of input image is not CV_64F." << endl;
		return;
	}

	// Initial the size of output image.
	dst = cvCreateMat(src.rows / 2, src.cols / 2, CV_64F);

	for(int dstX = 0; dstX < dst.rows; dstX++){
		for(int dstY = 0; dstY < dst.cols; dstY++){
			double sum = ((double*)src.data)[(dstX * 2)* src.cols + (dstY * 2)] 
						+ ((double*)src.data)[(dstX * 2 + 1 )* src.cols + (dstY * 2)]
						+ ((double*)src.data)[(dstX * 2)* src.cols + (dstY * 2 + 1)]
						+ ((double*)src.data)[(dstX * 2 + 1 )* src.cols + (dstY * 2 + 1)];
			((double*)dst.data)[dstX * dst.cols + dstY] = sum / 4.0;
		}
	}
}


/*************************************************
Function:    convolve
Description: // Convolution of the input image with the filter
Inputs:      // src, input gray image and its bit depth
				is CV_64F
			 // filter
             // dst, output image
			 // a, the width of the filter
			 // b, the height of the filter
**************************************************/  
void SIFT::convolve(const Mat& src, double filter[], Mat& dst, int a, int b){
	// Initialization
	dst = cvCreateMat(src.rows, src.cols, CV_64F);
	int filterSize = min(2 * a + 1, 2 * b + 1);

	for(int x = a; x < src.rows - a; x++){
		for(int y = b; y < src.cols - b; y++){
			double sum = 0.0;
			for(int s = -a; s <= a; s++){
				for(int t = -b; t <= b; t++){
					sum += filter[(s + a) * filterSize + (t + b)] 
							* ((double*)src.data)[(x + s) * src.cols + (y + t)];
				}
			}
			((double*)dst.data)[x * src.cols + y] = sum;
		}
	}
}

/*************************************************
Function:    gaussianSmoothing
Description: // Gaussian Smoothing of the input image 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // dst, output image
			 // sigma
**************************************************/  
void SIFT::gaussianSmoothing(const Mat& src, Mat& dst, double sigma){
	// Generate the gaussian mask
	GaussianMask gaussianMask(sigma);
	double* filter = gaussianMask.getFullMask();
	int filterSize = gaussianMask.getSize();
	Mat dstCx;
	convolve(src, filter,dstCx, filterSize / 2,0);
	convolve(dstCx, filter,dst, 0, filterSize / 2);
}


/*************************************************
Function:    gaussianSmoothing
Description: // Gaussian Smoothing of the input image 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // dst, output image
			 // sigma
**************************************************/  
void SIFT::substruction(const Mat& src1, const Mat& src2, Mat& dst){
	if(src1.cols != src2.cols || src1.cols != src2.cols ){
		cout << "The sizes are not same." << endl;
		return;
	}
	dst = cvCreateMat(src1.rows, src1.cols, CV_64F);
	for(int x = 0 ; x < src1.rows; x++){
		for(int y = 0; y < src1.cols; y++){
			((double*)dst.data)[x * dst.cols + y] = ((double*)src1.data)[x * src1.cols + y] 
													- ((double*)src2.data)[x * src2.cols + y] ;
		}
	}
}


/*************************************************
Function:    generateGaussianPyramid
Description: // generate the Gaussian Pyramid 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // octaves, the number of octaves
			 // scale, the height of scale space
			 // sigma, the parameter of Gaussian
Outputs:	 // 1D vector of Gaussian Pyramid
**************************************************/  
vector<Mat> SIFT::generateGaussianPyramid(Mat& src, int octaves, int scales, double sigma){
	double k = pow(2, (1 / (double)this->scale));
	int intervalGaus = scale + 3;

	vector<Mat> gaussPyr;
	// Convert to Gray image and Up sample 
	Mat initGrayImg, initUpSampling;
	convertRGBToGray64F(src,initGrayImg);
	upSample(initGrayImg, initUpSampling);
	
	// Generate a list of series of sigma
	double *sigmas = new double[intervalGaus];
	sigmas[0] = sigma;
	for(int i = 1; i < intervalGaus; i++){
		sigmas[i] = sigmas[i - 1] * k;
	}

	// Generate smoothing images in the first octave
	for(int i = 0; i < intervalGaus; i++){
		Mat smoothing;
		//cout << sigmas[i] << endl;
		gaussianSmoothing(initUpSampling,smoothing,sigmas[i]);
		gaussPyr.insert(gaussPyr.end(),smoothing);
	}
	// Use down sample to generate the rest octave
	vector<Mat>::iterator it;
	for(int i = 1; i < octaves; i++){
		int currentLayer = intervalGaus * i;
		int prevLayer = intervalGaus * (i - 1);
		for(int j = 0; j < intervalGaus; j++){
			it = gaussPyr.begin() + prevLayer + j;
			Mat gauss;
			downSample(*it, gauss);
			gaussPyr.insert(gaussPyr.end(),gauss);

		}
	}
	// Test:
	/*int i;
	for(it = gaussPyr.begin(), i = 0; it != gaussPyr.end(); it++, i++){
		char buffer[20];
		itoa(i,buffer,10);
		string number(buffer);
		string name = "Image " + number;
		namedWindow(name);  
		imshow(name,*it);  
	}*/
	return gaussPyr;
}


/*************************************************
Function:    generateDoGPyramid
Description: // generate the DoG Pyramid 
Inputs:      // src, input gray image and its bit depth
				is CV_64F
             // octaves, the number of octaves
			 // scale, the height of scale space
			 // sigma, the parameter of Gaussian
Outputs:	 // 1D vector of DoG Pyramid
**************************************************/ 
vector<Mat> SIFT::generateDoGPyramid(vector<Mat>& gaussPyr, int octaves, int scales, double sigma){
	int intervalDoGs = scale + 2;
	int intervalGaus = scale + 3;
	vector<Mat> dogPyr;

	vector<Mat>::iterator currIt;
	vector<Mat>::iterator nextIt;
	for(int i = 0; i < octaves; i++){
		for(int j = 0; j < intervalDoGs; j++){
			int number = i * intervalGaus + j;
			currIt = gaussPyr.begin() + number;
			nextIt = gaussPyr.begin() + number + 1;
			Mat subImg;
			substruction(*nextIt, *currIt, subImg);
			dogPyr.insert(dogPyr.end(), subImg);
		}
	}
	int num;
	// Test:
	/*for(currIt = dogPyr.begin(), num = 0; currIt != dogPyr.end(); currIt++, num++){
		char buffer[20];
		itoa(num,buffer,10);
		string number(buffer);
		string name = "Image " + number;
		namedWindow(name);  
		imshow(name,*currIt);  
	}	return dogPyr;*/
}


/*************************************************
Function:   GaussianMask
Description: // Constructor, Calculate the Gaussian mask
Inputs:      // sigma
**************************************************/
GaussianMask::GaussianMask(double sigma){
	double gaussianDieOff = 0.001;
	vector<double> halfMask;
	for(int i = 0; true; i++){
		double d = exp(- i * i / (2 * sigma * sigma));
		if(d < gaussianDieOff) 
			break;
		halfMask.insert(halfMask.begin(),d);
	}

	this->maskSize = 2 * halfMask.size() - 1;
	this->fullMask = new double[this->maskSize];
	vector<double>::iterator it;
	int num = 0;
	for(it = halfMask.begin(); it != halfMask.end(); it++, num++){
		fullMask[num] = *it;
	}
	for(num -= 1; num < this->maskSize ; num++){
		fullMask[num] = fullMask[this->maskSize  - 1 - num];
	}

	double sum = 0.0;
	for(int i = 0; i < maskSize; i++){
		sum += fullMask[i];
	}
	for(int i = 0; i < maskSize; i++){
		fullMask[i] /= sum;
	}
}