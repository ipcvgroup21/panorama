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
	
	cout << SIGMA;
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
	
	// Test:
	/*int num;
	for(currIt = dogPyr.begin(), num = 0; currIt != dogPyr.end(); currIt++, num++){
		char buffer[20];
		itoa(num,buffer,10);
		string number(buffer);
		string name = "Image " + number;
		namedWindow(name);  
		imshow(name,*currIt);  
	} */
	return dogPyr;
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





// detect features by finding extrema in DoG scale space
void SIFT::findKeypoints() {
	for (int o = 0; o < this->octave; o++)
		for (int i = 0; i < intervals; i++)
			for (int r = SIFT_IMG_BORDER; r < dog_pyr[o * (intervals + 2) + i + 1].rows - SIFT_IMG_BORDER; r++)
				for (int c = SIFT_IMG_BORDER; c < dog_pyr[o * (intervals + 2) + i + 1].cols - SIFT_IMG_BORDER; c++) {
					if (isExtremum(o, i, r, c)) {
						if (interpolateExtrema(o, i, r, c) != NULL) {
							// TODO: add feature
						}
					}
				}
				// TODO: remove duplicated features
}

// check its 26 neighbors in 3x3 regions at the current and adjacent scales
// to find out if it's an extremum

bool Sift::isExtremum(int octave, int interval, int row, int column) {
	bool maximum = true, minimum = true;

	float cur_value = dog_pyr[octave * (intervals + 2) + interval + 1].at<float>(row, column);

	// (for convenience the loop check include current pixel itself)
	for (int i = interval - 1; i <= interval + 1; i++)
		for (int r = row - 1; r <= row + 1; r++)
			for (int c = column - 1; c <= column + 1; c++) {
				// current image = dog_pyr[octave * (intervals + 2) + i + 1]
				if (cur_value < dog_pyr[octave * (intervals + 2) + i + 1].at<float>(r, c))
					maximum = false;
				if (cur_value > dog_pyr[octave * (intervals + 2) + i + 1].at<float>(r, c))
					minimum = false;
				if (!maximum && !minimum)
					return false;
			}
			return true;
}
 
// interpolate extrema to sub-pixel accuracy

Feature* SIFT::interpolateExtrema(int octave, int interval, int row, int column){
	int i = 0;
	int cur_interval = interval;
	int cur_row = row;
	int cur_col = column;
	Vec3f dD;
	Vec3f offset;
	while (i < SIFT_MAX_INTERP_STEPS) {
		int layer_index = octave * (intervals + 2) + cur_interval + 1;

		// first derivative
		float dx = (dog_pyr[layer_index].at<float>(cur_row, cur_col + 1) -
			dog_pyr[layer_index].at<float>(cur_row, cur_col - 1)) / 2;
		float dy = (dog_pyr[layer_index].at<float>(cur_row + 1, cur_col) -
			dog_pyr[layer_index].at<float>(cur_row - 1, cur_col)) / 2;
		float ds = (dog_pyr[layer_index + 1].at<float>(cur_row, cur_col) -
			dog_pyr[layer_index - 1].at<float>(cur_row, cur_col)) / 2;
		//Vec3f dD(dx, dy, ds);
		dD[0] = dx;
		dD[1] = dy;
		dD[2] = ds;

		// second partial derivative (3 * 3 hessian matrix)
		float dxx = dog_pyr[layer_index].at<float>(cur_row, cur_col + 1) +
			dog_pyr[layer_index].at<float>(cur_row, cur_col - 1) -
			dog_pyr[layer_index].at<float>(cur_row, cur_col) * 2;
		float dyy = dog_pyr[layer_index].at<float>(cur_row + 1, cur_col) +
			dog_pyr[layer_index].at<float>(cur_row - 1, cur_col) -
			dog_pyr[layer_index].at<float>(cur_row, cur_col) * 2;
		float dss = dog_pyr[layer_index + 1].at<float>(cur_row, cur_col) +
			dog_pyr[layer_index - 1].at<float>(cur_row, cur_col) -
			dog_pyr[layer_index].at<float>(cur_row, cur_col) * 2;
		float dxy = (dog_pyr[layer_index].at<float>(cur_row + 1, cur_col + 1) -
			dog_pyr[layer_index].at<float>(cur_row + 1, cur_col - 1) -
			dog_pyr[layer_index].at<float>(cur_row - 1, cur_col + 1) -
			dog_pyr[layer_index].at<float>(cur_row - 1, cur_col - 1)) / 4;
		float dxs = (dog_pyr[layer_index + 1].at<float>(cur_row, cur_col + 1) -
			dog_pyr[layer_index + 1].at<float>(cur_row, cur_col - 1) -
			dog_pyr[layer_index - 1].at<float>(cur_row, cur_col + 1) -
			dog_pyr[layer_index - 1].at<float>(cur_row, cur_col - 1)) / 4;
		float dys = (dog_pyr[layer_index + 1].at<float>(cur_row + 1, cur_col) -
			dog_pyr[layer_index + 1].at<float>(cur_row - 1, cur_col) -
			dog_pyr[layer_index - 1].at<float>(cur_row + 1, cur_col) -
			dog_pyr[layer_index - 1].at<float>(cur_row - 1, cur_col)) / 4;
		Matx33f hessian(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		// the offset is the inverse of second derivative multiply by first derivative (Lowe's equation 3)
		Matx33f inv_hessian(hessian.inv(DECOMP_SVD));
		offset = -inv_hessian * dD;

		float offset_c = offset[0];
		float offset_r = offset[1];
		float offset_i = offset[2];

		// if the offset is smaller than 0.5 in any dimension, return the offset
		if (abs(offset_c) < 0.5 && abs(offset_r) < 0.5 && abs(offset_i) < 0.5)
			break;

		// if the offset is larger than 0.5 in any dimension, 
		// then it means that the extremum lies closer to a different sample point
		cur_col += cvRound(offset_c);
		cur_row += cvRound(offset_r);
		cur_interval += cvRound(offset_i);

		// if the point is over the border, discards it
		if (cur_interval < 0 || cur_interval > intervals ||
			cur_col < SIFT_IMG_BORDER || cur_row < SIFT_IMG_BORDER ||
			cur_col >= dog_pyr[octave * (intervals + 2) + cur_interval + 1].cols - SIFT_IMG_BORDER ||
			cur_row >= dog_pyr[octave * (intervals + 2) + cur_interval + 1].rows - SIFT_IMG_BORDER)
			return NULL;

		i++;
	}

	if (i >= SIFT_MAX_INTERP_STEPS)
		return NULL;

	// rejecting unstable extrema with low contrast
	// all extrema with a value of |D(x)| less than the threshold (0.03 in Lowe's paper) were discarded
	int layer_index = octave * (intervals + 2) + cur_interval + 1;
	float Dx = dog_pyr[layer_index].at<float>(cur_row, cur_col) + dD.dot(offset) / 2;
	if (abs(Dx) < SIFT_CONTR_THR)
		return NULL;

	if (!isEdge(octave, cur_interval, cur_row, cur_col)) {
		//TODO: return feature
	}
}
 
// eliminate edge responses
 
bool SIFT::isEdge(int octave, int interval, int row, int column) {
	int layer_index = octave * (intervals + 2) + interval + 1;

	// the principal curvatures can be computed from a 2 * 2 hessian matrix
	float dxx = dog_pyr[layer_index].at<float>(row, column + 1) +
		dog_pyr[layer_index].at<float>(row, column - 1) -
		dog_pyr[layer_index].at<float>(row, column) * 2;
	float dyy = dog_pyr[layer_index].at<float>(row + 1, column) +
		dog_pyr[layer_index].at<float>(row - 1, column) -
		dog_pyr[layer_index].at<float>(row, column) * 2;
	float dxy = (dog_pyr[layer_index].at<float>(row + 1, column + 1) -
		dog_pyr[layer_index].at<float>(row + 1, column - 1) -
		dog_pyr[layer_index].at<float>(row - 1, column + 1) -
		dog_pyr[layer_index].at<float>(row - 1, column - 1)) / 4;

	float trace = dxx + dyy;
	float det = dxx * dyy - dxy * dxy;

	//if the curvatures have different signs, the point is discarded as not being an extrema
	if (det <= 0)
		return true;

	// eliminates keypoints that have a ratio between the principal curvatures greater than the threshold (10 in Lowe's paper)
	if (trace * trace / det >= (SIFT_CURV_THR + 1) * (SIFT_CURV_THR + 1) / SIFT_CURV_THR)
		return true;

	return false;
}
 
