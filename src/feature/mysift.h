/*************************************************
Copyright: SIFT::panorama
Author: Sally
Date:2016-07-26
Description: Extract SIFT features.
**************************************************/
#ifndef MYSIFT_H
#define MYSIFT_H
#include <iostream> 
#include "lib/mat.h"
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv/cv.h>
#include <math.h>
#include <vector>
#include "lib/matrix.hh"
#include "feature/feature.hh"

//using namespace cv;  
using namespace std;
using namespace pano;

/* Recommended sigma */
#define SIGMA 1.6f
/* Recommended scale spaces */
#define INTERVALS 3

#define SIFT_IMG_BORDER 5

#define SIFT_MAX_INTERP_STEPS 5

#define SIFT_CONTR_THR 0.0005f

#define SIFT_CURV_THR 10.f

/* Data struct of key point */
typedef struct key_points{
	double x;
	double y;
	int oct_id;
	int scale_id;
	double orientation;
	vector<float> *v;
}key_point;


/* SITF features of an image.*/

class mySIFT{
	private:
		Mat<float>* DoGs; /* 1D array of DoG Pyramid */
		Mat<float>* gausPyr;  /* 1D array of Gaussian Pyramid */
		int octave;  /* number of octaves */
		int scale; 
		//double sigma; 
		vector<key_point> features;
		Mat<float>* mag_Pyramid;
		Mat<float>* ori_Pyramid;
		double* sigma;
		std::vector<Descriptor> Desc;

	public:
		mySIFT(Mat<float>& img, int octave, int scale, double sigma);
		~mySIFT(){delete[] sigma;} 
		std::vector<Descriptor> GetDescriptor() { return Desc; }
		vector<key_point> get_extrema() {return features; }

	protected:
		Mat<float> convertRGBToGray(const Mat<float>& src);
		//void upSample(const Mat<float>& src, Mat<float>& dst); 
		Mat<float> downSample(const Mat<float>& src);
		void convolve(const Mat<float>& src, double filter[], Mat<float>& dst, int a, int b);
		void gaussianSmoothing(const Mat<float>& src, Mat<float>& dst, double sigma);
		void subtraction(const Mat<float>& src1, const Mat<float>& src2, Mat<float>& dst);
		Mat<float>* generateGaussianPyramid(Mat<float>& src, int octaves, int scales, double sigma);
		Mat<float>* generateDoGPyramid(Mat<float>* gaussPyr, int octaves, int scales, double sigma);
		
		vector<key_point> findKeypoints();
		bool isExtremum(int octave, int interval, int row, int column);
		bool interpolateExtrema(int octave, int interval, int row, int column, key_point& kp);
		bool isEdge(int octave, int interval, int row, int column);	

		void GetOriAndMag(int oct_id, int scale_id);
		std::vector<double> OrientationAssignment(key_point p);
		void GetVector(key_point p);
		void TriInterpolation(double x, double y, double h, double w_mag, double hist[][8]);
		double sqr(double a);

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

