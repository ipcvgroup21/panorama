
/*************************************************
Copyright: SIFT::panorama
Author: Sally
Date:2016-07-26
Description: Extract SIFT features.
**************************************************/

#ifndef SIFT_H
#define SIFT_H

#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <math.h>
#include <cv.h>

using namespace cv;  
using namespace std;

/* Recommended sigma */
#define SIGMA 1.6f
/* Recommended scale spaces */
#define INTERVALS 3

#define SIFT_IMG_BORDER 5

#define SIFT_MAX_INTERP_STEPS 5

#define SIFT_CONTR_THR 0.04f

#define SIFT_CURV_THR 10.f

/* Data struct of key point */
typedef struct key_points{
	double x;
	double y;
	int oct_id;
	int scale_id;
	double orientation;
	std::vector<double> *v;
}key_point;


/* SITF features of an image.*/

class SIFT{
	private:
		Mat* DoGs; /* 1D array of DoG Pyramid */
		Mat* gausPyr;  /* 1D array of Gaussian Pyramid */
		int octave;  /* number of octaves */
		int scale; 
		//double sigma; 
		//vector<Keypoint> features;

	public:
		SIFT(Mat& img, int octave, int scale, double sigma);
		~SIFT(){delete[] DoGs; delete[] gausPyr;} 
		

	protected:
		void convertRGBToGray64F(const Mat& src, Mat& dst);
		void upSample(const Mat& src, Mat& dst); 
		void downSample(const Mat& src, Mat& dst);
		void convolve(const Mat& src, double filter[], Mat& dst, int a, int b);
		void gaussianSmoothing(const Mat& src, Mat& dst, double sigma);
		void substruction(const Mat& src1, const Mat& src2, Mat& dst);
		Mat* generateGaussianPyramid(Mat& src, int octaves, int scales, double sigma);
		Mat* generateDoGPyramid(Mat* gaussPyr, int octaves, int scales, double sigma);
		
		vector<key_point> findKeypoints();
		bool isExtremum(int octave, int interval, int row, int column);
		key_point* interpolateExtrema(int octave, int interval, int row, int column);
		bool isEdge(int octave, int interval, int row, int column);	
};

class GaussianMask{
	private:
		int maskSize;
		double* fullMask;
		double sigma;
	public:
		GaussianMask(double sigma);
		~GaussianMask(){delete [] fullMask; }
		double* getFullMask(){ return fullMask;}
		int getSize(){return maskSize;}
};
#endif

