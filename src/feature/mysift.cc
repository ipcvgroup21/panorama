#include "mysift.h"
#include "lib/imgproc.hh"
#include "lodepng/lodepng.h"

mySIFT::mySIFT(Mat<float>& img, int octave, int scale, double sigma){
	// Initial the parameters
	this->octave = octave;
	this->scale = scale;
	
	// Construct the scale space
	this->gausPyr = generateGaussianPyramid(img, octave, scale, sigma);
	this->DoGs = generateDoGPyramid(this->gausPyr, octave, scale, sigma);
	//print_debug("start findkeypoints\n"); 
	this->features = findKeypoints();
	print_debug("find key points.\n");

	delete[] DoGs;

	this->mag_Pyramid = new Mat<float>[octave*(scale+3)];
	this->ori_Pyramid = new Mat<float>[octave*(scale+3)];
    	for(int i=0;i<octave;i++)
    		for(int j=0;j<scale+3;j++)
    			GetOriAndMag(i,j);
	print_debug("getoriandmag\n");
	int num_keypoint = features.size();
	print_debug("number = %d\n", num_keypoint);
    	for(int i=0;i<num_keypoint;i++)
    	{
    		key_point tmp = features[i];
    		std::vector<double> oris = OrientationAssignment(tmp);
		print_debug("num_keypoint = %d\n",i);
		if (oris.size() == 0) continue;
    		features[i].orientation=oris[0];
    		for(int j=1;j<oris.size();j++)
    		{
    			key_point p = tmp;
    			p.orientation = oris[j];
    			features.push_back(p);
    		}
    	}
	print_debug("number = %d\n", features.size());
     	for(int i=0;i<features.size();i++) GetVector(features[i]);
	
	print_debug("find features.\n");
	delete[] mag_Pyramid;
	delete[] ori_Pyramid;
	delete[] gausPyr;
}

Mat<float> mySIFT::convertRGBToGray(const Mat<float>& src){
	// Determine the type of input image
	if (src.channels() == 1){
		cout << "The input image is a gray image." << endl;
		return src;
	}

	// Initial the size of output image.
	Mat<float> dst(src.rows(), src.cols(), 1);
	//dst = cvCreateMat(src.rows, src.cols, CV_64F);

	// Convert the image 
	const float* srcdata = src.ptr();
	float* dstdata = dst.ptr();
	for(int i = 0; i < src.pixels(); i++){
		dstdata[i] = (srcdata[i * 3 + 0] + srcdata[i * 3 + 1] + srcdata[i * 3 + 2]) / 3.0f;
	}
	return dst;
}

 Mat<float> mySIFT::downSample(const Mat<float>& src){
	 // Determine the type of input image
	if (src.channels() != 1){
		cout << "The input image is not a gray image." << endl;
		return src;
	}
	
	Mat<float> dst(src.rows() / 2, src.cols() / 2, 1);
	

	for (int dstX = 0; dstX < dst.rows(); dstX++){
		for (int dstY = 0; dstY < dst.cols(); dstY++){
			float sum = (src.at(dstX * 2, dstY * 2) + src.at(dstX * 2 + 1, dstY * 2)
				+ src.at(dstX * 2, dstY * 2 + 1) + src.at(dstX * 2 + 1, dstY * 2 + 1));
			dst.at(dstX, dstY) = sum / 4.0f;
		}
	}
	
	return dst;
}

void mySIFT::convolve(const Mat<float>& src, double filter[], Mat<float>& dst, int a, int b){
	// Initialization
	dst = src.clone();
	
	// Padding
	 
	Mat<float> srcPadding(src.rows() + 2 * a, src.cols() + 2 * b, 1);
/*	for(int x = 0; x < srcPadding.rows(); x++){
		for(int y = 0; y < srcPadding.cols(); y++){
			if(y > (src.cols() + b - 1) || x > (src.rows() + a - 1) || x <= a || y <=b)
				srcPadding.at(x, y) = 0.0;
			else
				srcPadding.at(x, y) = src.at(x - a, y - b);
		}
	}
*/	 
	int filterSize = min(2 * a + 1, 2 * b + 1);

 /*
	for(int x = a; x < srcPadding.rows() - a; x++){
		for(int y = b; y < srcPadding.cols() - b; y++){
			double sum = 0.0;
			for(int s = -a; s <= a; s++){
				for(int t = -b; t <= b; t++){
					sum += filter[(s + a) * filterSize + (t + b)] * srcPadding.at(x + s, y + t);	
				}
			}
			dst.at(x - a, y - b) = sum;
		}
	} 
*/	 
	
	for(int x = a; x < src.rows() - a; x++){
 		for(int y = b; y < src.cols() - b; y++){
 			float sum = 0.0;
 			for(int s = -a; s <= a; s++){
 				for(int t = -b; t <= b; t++){
 					sum += filter[(s + a) * filterSize + (t + b)] * src.at(x + s, y + t);
 							
 				}
 			}
 			dst.at(x , y) = sum;
 		}
 	}
}

void mySIFT::gaussianSmoothing(const Mat<float>& src, Mat<float>& dst, double sigma){
	// Generate the gaussian mask
	GaussianMask gaussianMask(sigma);
	double* filter = gaussianMask.getFullMask();
	int filterSize = gaussianMask.getSize();
	Mat<float> dstCx;
	convolve(src, filter, dstCx, filterSize / 2, 0);
	convolve(dstCx, filter, dst, 0, filterSize / 2);
}

void mySIFT::subtraction(const Mat<float>& src1, const Mat<float>& src2, Mat<float>& dst){
	if (src1.cols() != src2.cols() || src1.rows() != src2.rows()){
		cout << "The sizes are not same." << endl;
		return;
	}
	dst = src1.clone();
	for (int x = 0; x < src1.rows(); x++){
		for (int y = 0; y < src1.cols(); y++){
//			dst.at(x,y) = ((src1.at(x,y) - src2.at(x,y)) > 0 ? (src1.at(x,y) - src2.at(x,y)) : (src2.at(x,y) - src1.at(x,y)));
			dst.at(x,y) = src1.at(x, y) - src2.at(x,y);
		}
	}
}

Mat<float>* mySIFT::generateGaussianPyramid(Mat<float>& src, int octaves, int scales, double sigma){
	double k = pow(2, (1 / (double)this->scale));
	int intervalGaus = scale + 3;

	Mat<float>* gaussPyr = new Mat<float>[octaves * intervalGaus];

	Mat<float> initGrayImg = convertRGBToGray(src);
	
	double *sigmas = new double[intervalGaus];
	sigmas[0] = sigma;
	for(int i = 1; i < intervalGaus; i++){
		sigmas[i] = sigmas[i - 1] * k;
	}

	this->sigma = new double[octave * intervalGaus];
	for (int i = 0; i < intervalGaus; i++)
		this->sigma[i] = sigmas[i];
	for(int i = 1 ; i < octave ; i++)
		for (int j = 0; j < intervalGaus; j++) {
			if (j == 0)
				this->sigma[i*intervalGaus] = this->sigma[i*intervalGaus - 3];
			else
				this->sigma[i*intervalGaus + j] = this->sigma[i*intervalGaus + j - 1] * k;
		}
	
	// Generate smoothing images in the first octave
	for (int i = 0; i < intervalGaus; i++){
		Mat<float> smoothing;
		if(i == 0)
			//gaussianSmoothing(initUpSampling, smoothing, sigmas[i]);
			gaussianSmoothing(initGrayImg, smoothing, sigmas[i]);
		else
			gaussianSmoothing(gaussPyr[i-1], smoothing, sigmas[i]);
		gaussPyr[i] = smoothing;
	}
	// Use down sample to generate every first layer of rest octave 
	// And then get smooth images base on it
	 
	for(int i = 1; i < octaves; i++){
		//Mat<float> downSampling;
		//downSample(gaussPyr[intervalGaus * (i - 1) + 3], downSampling);
		gaussPyr[i * intervalGaus] = downSample(gaussPyr[intervalGaus * (i - 1) + 3]);
		for(int j = 1; j < intervalGaus; j++){
			Mat<float> dsmoothing;
			gaussianSmoothing(gaussPyr[i * intervalGaus + j - 1], dsmoothing, sigmas[j]);
			gaussPyr[i * intervalGaus + j] = dsmoothing; 
		}

	} 
	//write_rgb("gaus16.jpg",gaussPyr[16]);
	int n = gaussPyr[15].pixels();
	vector<unsigned char> img(n*4);
	const float* p = gaussPyr[15].ptr();
	unsigned char* data = img.data();
	REP(i,n){
		data[0] = data[1] = data[2] = ((p[0] < 0) ? 1 : p[0]) * 255;
		data[3] = 255;
		data += 4;
		p += 1;
	}
	unsigned error = lodepng::encode("gauss15.png",img, gaussPyr[15].width(), gaussPyr[15].height());
	if(error)
		error_exit(ssprintf("png encode error %u: %s", error, lodepng_error_text(error)));

	return gaussPyr;
}

Mat<float>* mySIFT::generateDoGPyramid(Mat<float>* gaussPyr, int octaves, int scales, double sigma){
	int intervalDoGs = scale + 2;
	int intervalGaus = scale + 3;
	Mat<float>* dogPyr = new Mat<float>[this->octave * intervalDoGs];

	for(int i = 0; i < octaves; i++){
		for(int j = 0; j < intervalDoGs; j++){
			Mat<float> subImg;
			subtraction(gaussPyr[i * intervalGaus + j + 1], gaussPyr[i * intervalGaus + j], subImg);
			dogPyr[i * intervalDoGs + j] = subImg;
		}
	}
//	write_rgb("dog13.jpg",dogPyr[13]);
	int n = dogPyr[13].pixels();
	vector<unsigned char> img(n*4);
	const float* p = dogPyr[13].ptr();
	unsigned char* data = img.data();
	REP(i,n){
		data[0] = data[1] = data[2] =(unsigned char)(((p[0] < 0) ? (-p[0]) : p[0]) * 2000.0) ;
		data[3] = 255;
		data += 4;
		p += 1;
	}
	unsigned error = lodepng::encode("dog13.png",img, dogPyr[13].width(), dogPyr[13].height());
	if(error)
		error_exit(ssprintf("png encode error %u: %s", error, lodepng_error_text(error)));

	n = dogPyr[3].pixels();
	vector<unsigned char> img3(n*4);
	const float* p3 = dogPyr[3].ptr();
	unsigned char* data3 = img3.data();
	float min = 999.0;
	REP(i,n){
		if(min > p3[0]) min = p3[0];
		p3 += 1;
	}
	p3 = dogPyr[3].ptr();
	REP(i,n){
		data3[0] = data3[1] = data3[2] = (unsigned char)((p3[0] - min) * 1000.0);
				 //=(unsigned char)(((p3[0] < 0) ? (-p3[0]) : p3[0]) * 2000.0) ;
		data3[3] = 255;
		data3 += 4;
		p3 += 1;
	}
	unsigned error3 = lodepng::encode("dog3.png",img3, dogPyr[3].width(), dogPyr[3].height());
	if(error3)
		error_exit(ssprintf("png encode error %u: %s", error3, lodepng_error_text(error3)));
	
	return dogPyr;
	
}

GaussianMask::GaussianMask(double sigma){
	double gaussianDieOff = 0.001;
	vector<double> halfMask;
	for(int i = 0; true; i++){
	//for(int i = 0; i <= 3; i++){
		double d = exp(- i * i / (2 * sigma * sigma));
		if(d < gaussianDieOff) 
			break;
		halfMask.insert(halfMask.begin(), d);
	}

	this->maskSize = 2 * halfMask.size() - 1;
	this->fullMask = new double[this->maskSize];
	vector<double>::iterator it;
	int num = 0;
	for (it = halfMask.begin(); it != halfMask.end(); it++, num++){
		fullMask[num] = *it;
	}
	for (num -= 1; num < this->maskSize; num++){
		fullMask[num] = fullMask[this->maskSize - 1 - num];
	}

	double sum = 0.0;
	for (int i = 0; i < maskSize; i++){
		sum += fullMask[i];
	}
	for (int i = 0; i < maskSize; i++){
		fullMask[i] /= sum;
	}
}


// detect features by finding extrema in DoG scale space
vector<key_point> mySIFT::findKeypoints() {
	vector<key_point> keyPoints;
	for (int o = 0; o < octave; o++) {
		for (int i = 0; i < scale; i++) {
			for (int r = SIFT_IMG_BORDER; r < DoGs[o * (scale + 2) + i + 1].rows() - SIFT_IMG_BORDER; r++) {
				for (int c = SIFT_IMG_BORDER; c < DoGs[o * (scale + 2) + i + 1].cols() - SIFT_IMG_BORDER; c++) {
					if (isExtremum(o, i, r, c)) {
						key_point kp;
						if (interpolateExtrema(o, i, r, c, kp)) {
							keyPoints.push_back(kp);
						}
					}
				}
			}
		}
	
	}
	return keyPoints;
}

// check its 26 neighbors in 3x3 regions at the current and adjacent scales
// to find out if it's an extremum
bool mySIFT::isExtremum(int octave, int interval, int row, int column) {
	bool maximum = true, minimum = true;

	double cur_value = DoGs[octave * (this->scale + 2) + interval + 1].at(row, column);

	// (for convenience the loop check include current pixel itself)
	for (int i = interval - 1; i <= interval + 1; i++) {
		for (int r = row - 1; r <= row + 1; r++) {
			for (int c = column - 1; c <= column + 1; c++) {
				// current image = DoGs[octave * (scale + 2) + i + 1]
				if (cur_value < DoGs[octave * (scale + 2) + i + 1].at(r, c))
					maximum = false;
				if (cur_value > DoGs[octave * (scale + 2) + i + 1].at(r, c))
					minimum = false;
				if (!maximum && !minimum)
					return false;
			}
		}
	}
	return true;
}

// interpolate extrema to sub-pixel accuracy
bool mySIFT::interpolateExtrema(int octave, int interval, int row, int column, key_point& kp){
	int i = 0;
	int cur_interval = interval;
	int cur_row = row;
	int cur_col = column;
	int layer_index;
	Matrix dD(3, 1);
	Matrix offset(3, 1);
	double dx, dy, ds, dxx, dyy, dss, dxy, dxs, dys;
	double offset_c, offset_r, offset_i;
	while (i < SIFT_MAX_INTERP_STEPS) {
		layer_index = octave * (scale + 2) + cur_interval + 1;

		// first derivative
		dx = (DoGs[layer_index].at(cur_row, cur_col + 1) -
			DoGs[layer_index].at(cur_row, cur_col - 1)) / 2;
		dy = (DoGs[layer_index].at(cur_row + 1, cur_col) -
			DoGs[layer_index].at(cur_row - 1, cur_col)) / 2;
		ds = (DoGs[layer_index + 1].at(cur_row, cur_col) -
			DoGs[layer_index - 1].at(cur_row, cur_col)) / 2;
		//Vec3f dD(dx, dy, ds);
		dD.at(0, 0) = dx;
		dD.at(1, 0) = dy;
		dD.at(2, 0) = ds;

		// second partial derivative (3 * 3 hessian matrix)
		dxx = DoGs[layer_index].at(cur_row, cur_col + 1) +
			DoGs[layer_index].at(cur_row, cur_col - 1) -
			DoGs[layer_index].at(cur_row, cur_col) * 2;
		dyy = DoGs[layer_index].at(cur_row + 1, cur_col) +
			DoGs[layer_index].at(cur_row - 1, cur_col) -
			DoGs[layer_index].at(cur_row, cur_col) * 2;
		dss = DoGs[layer_index + 1].at(cur_row, cur_col) +
			DoGs[layer_index - 1].at(cur_row, cur_col) -
			DoGs[layer_index].at(cur_row, cur_col) * 2;
		dxy = (DoGs[layer_index].at(cur_row + 1, cur_col + 1) -
			DoGs[layer_index].at(cur_row + 1, cur_col - 1) -
			DoGs[layer_index].at(cur_row - 1, cur_col + 1) +
			DoGs[layer_index].at(cur_row - 1, cur_col - 1)) / 4;
		dxs = (DoGs[layer_index + 1].at(cur_row, cur_col + 1) -
			DoGs[layer_index + 1].at(cur_row, cur_col - 1) -
			DoGs[layer_index - 1].at(cur_row, cur_col + 1) +
			DoGs[layer_index - 1].at(cur_row, cur_col - 1)) / 4;
		dys = (DoGs[layer_index + 1].at(cur_row + 1, cur_col) -
			DoGs[layer_index + 1].at(cur_row - 1, cur_col) -
			DoGs[layer_index - 1].at(cur_row + 1, cur_col) +
			DoGs[layer_index - 1].at(cur_row - 1, cur_col)) / 4;
		
		Matrix hessian(3, 3);
		
		/* The Hessian Matrix:
		     /dxx dxy dxs\
		    | dyx dyy dys |
	        	 \dsx dsy dss/     */
		hessian.at(0, 0) = dxx;
		hessian.at(0, 1) = dxy;
		hessian.at(0, 2) = dxs;
		hessian.at(1, 0) = dxy;
		hessian.at(1, 1) = dyy;
		hessian.at(1, 2) = dys;
		hessian.at(2, 0) = dxs;
		hessian.at(2, 1) = dys;
		hessian.at(2, 2) = dss;

		// the offset is the inverse of second derivative multiply by first derivative (Lowe's equation 3)
		Matrix inv_hessian(3, 3);
		if (!hessian.inverse(inv_hessian)) {	  // pseudo inverse is slow
			inv_hessian = hessian.pseudo_inverse();
		}
		offset = inv_hessian * dD;

		offset_c = -offset.at(0, 0);
		offset_r = -offset.at(1, 0);
		offset_i = -offset.at(2, 0);

		// if the offset is smaller than 0.5 in any dimension, return the offset
		if (abs(offset_c) < 0.5 && abs(offset_r) < 0.5 && abs(offset_i) < 0.5)
			break;

		// if the offset is larger than 0.5 in any dimension, 
		// then it means that the extremum lies closer to a different sample point
		cur_col += round(offset_c);
		cur_row += round(offset_r);
		cur_interval += round(offset_i);

		// if the point is over the border, discards it
		if (cur_interval < 0 || cur_interval > scale - 1 ||
			cur_col < SIFT_IMG_BORDER || cur_row < SIFT_IMG_BORDER ||
			cur_col >= DoGs[octave * (scale + 2) + cur_interval + 1].cols() - SIFT_IMG_BORDER ||
			cur_row >= DoGs[octave * (scale + 2) + cur_interval + 1].rows() - SIFT_IMG_BORDER)
			return false;

		i++;
	}
	

	// can not find keypoint within the iteration
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;
	
	// rejecting unstable extrema with low contrast
	// all extrema with a value of |D(x)| less than the threshold (0.03 in Lowe's paper) were discarded
	layer_index = octave * (scale + 2) + cur_interval + 1;
        double dot_product = dD.at(0, 0)*offset.at(0, 0)+dD.at(1, 0)*offset.at(1, 0)+dD.at(2, 0)*offset.at(2, 0);
	double D_x = DoGs[layer_index].at(cur_row, cur_col) + dot_product / 2;
//	cout << "dot" << endl;
	if (abs(D_x) < SIFT_CONTR_THR)
		return false;
	
	if (!isEdge(octave, cur_interval, cur_row, cur_col)) {
		// the keypoint's actual coordinate in the image
		double x = (cur_col + offset_c);
		double y = (cur_row + offset_r);
		//print_debug("x=%lf,y=%lf\n",x,y);
		kp = { x, y, octave, cur_interval, 0.0, NULL };

		return true;
	}
	else
		return false;
}

// eliminate edge responses
bool mySIFT::isEdge(int octave, int interval, int row, int column) {
	int layer_index = octave * (scale + 2) + interval + 1;

	// the principal curvatures can be computed from a 2 * 2 hessian matrix
	double dxx = DoGs[layer_index].at(row, column + 1) +
		DoGs[layer_index].at(row, column - 1) -
		DoGs[layer_index].at(row, column) * 2;
	double dyy = DoGs[layer_index].at(row + 1, column) +
		DoGs[layer_index].at(row - 1, column) -
		DoGs[layer_index].at(row, column) * 2;
	double dxy = (DoGs[layer_index].at(row + 1, column + 1) -
		DoGs[layer_index].at(row + 1, column - 1) -
		DoGs[layer_index].at(row - 1, column + 1) +
		DoGs[layer_index].at(row - 1, column - 1)) / 4;

	double trace = dxx + dyy;
	double det = dxx * dyy - dxy * dxy;

	//if the curvatures have different signs, the point is discarded as not being an extremum
	if (det <= 0)
		return true;

	// eliminates keypoints that have a ratio between the principal curvatures greater than the threshold (10 in Lowe's paper)
	if (trace * trace / det >= (SIFT_CURV_THR + 1) * (SIFT_CURV_THR + 1) / SIFT_CURV_THR)
		return true;

	return false;
}


void mySIFT::GetOriAndMag(int oct_id, int scale_id)
{
	double PI=3.14159265358;
	Mat32f origin= gausPyr[oct_id*6+scale_id];
	int w = origin.cols(); 
	int h = origin.rows();
	mag_Pyramid[oct_id*6+scale_id] = Mat32f(h,w,1);
	ori_Pyramid[oct_id*6+scale_id] = Mat32f(h,w,1);

	Mat32f& mag = mag_Pyramid[oct_id*6+scale_id];
	Mat32f& ori = ori_Pyramid[oct_id*6+scale_id];

	for(int y=0;y<h;y++)
	{
		(mag.ptr(y))[0]=0;
		(ori.ptr(y))[0]=PI;

		for(int x=1;x<w-1;x++)
		{
			if(y>0 && y<h-1)
			{
				double dy=(ori.ptr(y+1))[x]-(ori.ptr(y-1))[x];
				double dx=(ori.ptr(y))[x+1]-(ori.ptr(y))[x-1];
				(mag.ptr(y))[x] = sqrt(dx*dx+dy*dy);
				(ori.ptr(y))[x] = (dy==0 && dx==0) ? 0:atan2(dy,dx);
				if((ori.ptr(y))[x]<0) (ori.ptr(y))[x] += 2*PI;
			}
			else
			{
				(mag.ptr(y))[x]=0;
				(ori.ptr(y))[x]=PI;
			}
		}

		(mag.ptr(y))[w-1]=0;
		(ori.ptr(y))[w-1]=PI;
	}
}


std::vector<double> mySIFT::OrientationAssignment(key_point p)
{
	Mat32f& ori_img=ori_Pyramid[p.oct_id*6+p.scale_id];
	Mat32f& mag_img=mag_Pyramid[p.oct_id*6+p.scale_id];
	double PI=3.14159265358;
	int r=round(this->sigma[p.oct_id*6+p.scale_id] *4.5);
	double hist[36];
	for(int i=0;i<36;i++) hist[i]=0;

	for(int xx=-r;xx<=r;xx++)
	{
		int x=p.x+xx;
		if(x<=0 || x>=ori_img.cols()-1) continue;
		for(int yy=-r;yy<=r;yy++)
		{
			int y=p.y+yy;
			if(y<=0 || y>=ori_img.rows()-1) continue;
			if(sqr(xx)+sqr(yy)>sqr(r)) continue;
			double ori=ori_img.at(y,x);
			int bin=round(ori/PI*180.0/10.0);
			if (bin==36) bin=0;
			hist[bin]+=mag_img.at(y,x)*exp( -(sqr(xx)+sqr(yy)) / (2*sqr(1.5*this->sigma[p.oct_id * 6 + p.scale_id])) );

		}
	}

	for(int i=0;i<2;i++)
	{
		double hist_tmp[36];
		for(int j=0;j<36;j++)
		{
			double prev=hist[j==0? 35:j-1];
			double next=hist[j==35? 0:j+1];
			hist_tmp[j]=hist[j]*0.5+(prev+next)*0.25;
		}
		for(int j=0;j<36;j++)
		{
			hist[j]=hist_tmp[j];
		}
	}

	double hist_max=0;
	for(int i=0;i<36;i++)
	{
		if(hist[i]>hist_max) hist_max=hist[i];
	}
	double threshold=0.8*hist_max;
	std::vector<double> result;

	for(int i=0;i<36;i++)
	{
		double prev=hist[i==0? 35:i-1];
		double next=hist[i==35? 0:i+1];

		if(hist[i]>threshold && hist[i]>prev && hist[i]>next)
		{
			double a=(next+prev-2*hist[i])*0.5;
			double b=(next-prev)*0.5;

			double bin_real = i*1.0 - 0.5*b/a;
			if(bin_real<0) bin_real+=36;
			double orientation = bin_real * 10.0 / 180.0 * PI;
			result.push_back(orientation);
		}
	}
	return result;
}


void mySIFT::GetVector(key_point p)
{
	double PI=3.14159265358;
	Mat32f& mag_img=mag_Pyramid[p.oct_id*6+p.scale_id];
	Mat32f& ori_img=ori_Pyramid[p.oct_id*6+p.scale_id];
	int w=ori_img.cols(), h=ori_img.rows();

	double ori=p.orientation;
	int r= round(sqrt(0.5)*3*this->sigma[p.oct_id * 6 + p.scale_id]*(4+1));
	double hist[16][8];
	for(int i=0;i<16;i++)
		for(int j=0;j<8;j++)
			hist[i][j]=0;

	double cos_part= cos(ori), sin_part= sin(ori);

	for(int xx=-r;xx<=r;xx++)
	{
		int x=p.x+xx;
		if(x<=0 || x>=w-1) continue;
		for(int yy=-r;yy<=r;yy++)
		{
			int y=p.y+yy;
			if(y<=0 || y>=h-1) continue;
			if(sqr(xx)+sqr(yy)>sqr(r)) continue;

			double bin_y = (-xx*sin_part+yy*cos_part)/(3.0*this->sigma[p.oct_id * 6 + p.scale_id])+1.5;
			double bin_x = (xx*cos_part +yy*sin_part)/(3.0*this->sigma[p.oct_id * 6 + p.scale_id])+1.5;

			if(bin_y<-1 || bin_y>=4 || bin_x<-1 || bin_x>=4) continue;

			double ori_loc = ori_img.at(y,x);
			double mag_loc = mag_img.at(y,x);
			double w_mag = mag_loc * exp( -(sqr(xx)+sqr(yy)) / 8 );
			double r_ori = ori_loc -  ori;

			if(r_ori<0) r_ori += 2*PI;

			double bin_hist = r_ori /PI * 180.0 /45.0;
			TriInterpolation(bin_x,bin_y,bin_hist,w_mag,hist);
		}
	}
	std::vector<float> result(128);
	for(int i=0;i<16;i++)
		for(int j=0;j<8;j++)
			result[8*i+j]=hist[i][j];

	double sum=0;
	for(int i=0;i<128;i++) sum+=sqr(result[i]);
	sum=sqrt(sum);
	for(int i=0;i<128;i++)
	{
		result[i]/=sum;
		if(result[i]>0.2) result[i]=0.2;
	}
	sum=0;
	for(int i=0;i<128;i++) sum+=sqr(result[i]);
	sum=sqrt(sum);
	for(int i=0;i<128;i++) result[i]/=sum;

	Descriptor d;
	d.descriptor=result;
	d.coor.x= p.x/w;
	d.coor.y= p.y/h;

	Desc.emplace_back(move(d));
}


void mySIFT::TriInterpolation(double x, double y, double h, double w_mag, double hist[][8])
{
	int xf=floor(x),
	    yf=floor(y),
	    hf=floor(h);

	double delta_x = x-xf,
	       delta_y = y-yf,
	       delta_h = h-hf;

	for(int i=0;i<2;i++)
	{
		if((yf+i)>=0 && (yf+i)<4)
		{
			double wy = w_mag * (i? delta_y : 1-delta_y);
			for(int j=0;j<2;j++)
			{
				if((xf+j)>=0 && (xf+j)<4)
				{
					double wx = wy * (j? delta_x : 1-delta_x);
					hist[4*(yf+i)+(xf+j)][hf % 8] += wx * (1-delta_h);
					hist[4*(yf+i)+(xf+j)][(hf+1) % 8] += wx * delta_h;
				}
			}
		}
	}
}

double mySIFT::sqr(double a)
{
	return a*a;
}

