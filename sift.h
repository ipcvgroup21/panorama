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

static const int SIFT_IMG_BORDER = 5;

static const int SIFT_MAX_INTERP_STEPS = 5;

static const float SIFT_CONTR_THR = 0.04f;

static const float SIFT_CURV_THR = 10.f;

/* Data struct of key point */
struct Keypoint{
	int octave;   /* the key point located octave */
	int interval; /* the key point located layer in the octave */
	int x; /* coordinate of the key point*/
	int y;
};

/* SITF features of an image.*/

class SIFT{
	private:
		vector<Mat> DoGs; /* 1D array of DoG */
		int octave;  /* number of octaves */
		int scale; 
		//double sigma; 
		//vector<Keypoint> features;

	public:
		SIFT(Mat& img, int octave, int scale, double sigma);
		

	protected:
		void convertRGBToGray64F(const Mat& src, Mat& dst);
		void upSample(const Mat& src, Mat& dst); 
		void downSample(const Mat& src, Mat& dst);
		void convolve(const Mat& src, double filter[], Mat& dst, int a, int b);
		void gaussianSmoothing(const Mat& src, Mat& dst, double sigma);
		void substruction(const Mat& src1, const Mat& src2, Mat& dst);
		vector<Mat> generateGaussianPyramid(Mat& src, int octaves, int scales, double sigma);
		vector<Mat> generateDoGPyramid(vector<Mat>& gaussPyr, int octaves, int scales, double sigma);

		void findKeypoints();
		bool isExtremum(int octave, int interval, int row, int column);
		Feature* interpolateExtrema(int octave, int interval, int row, int column);
		bool isEdge(int octave, int interval, int row, int column);
};

class GaussianMask{
	private:
		int maskSize;
		double* fullMask;
		double sigma;
	public:
		GaussianMask(double sigma);
		double* getFullMask(){ return fullMask;}
		int getSize(){return maskSize;}
};
#endif
