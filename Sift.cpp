#include <cv.h>

using namespace cv;
using namespace std;


static const int SIFT_IMG_BORDER = 5;

static const int SIFT_MAX_INTERP_STEPS = 5;

static const float SIFT_CONTR_THR = 0.04f;

static const float SIFT_CURV_THR = 10.f;

typedef struct key_points
{
public:
	double x;
	double y;
	int oct_id;
	int scale_id;
	double orientation;
	std::vector<double> *v;
} key_point;

class Sift {
private:
	int octaves;
	int intervals;
	Mat* dog_pyr;
public:
	vector<key_point> findKeypoints();
	bool isExtremum(int octave, int interval, int row, int column);
	key_point* interpolateExtrema(int octave, int interval, int row, int column);
	bool isEdge(int octave, int interval, int row, int column);
};

// detect features by finding extrema in DoG scale space
vector<key_point> Sift::findKeypoints() {
	vector<key_point> keyPoints;
	for (int o = 0; o < octaves; o++)
	for (int i = 0; i < intervals; i++)
	for (int r = SIFT_IMG_BORDER; r < dog_pyr[o * (intervals + 2) + i + 1].rows - SIFT_IMG_BORDER; r++)
	for (int c = SIFT_IMG_BORDER; c < dog_pyr[o * (intervals + 2) + i + 1].cols - SIFT_IMG_BORDER; c++) {
		if (isExtremum(o, i, r, c)) {
			key_point* kp = interpolateExtrema(o, i, r, c);
			if (kp != NULL) {
				key_point new_kp;
				new_kp.x = kp->x;
				new_kp.y = kp->y;
				new_kp.oct_id = kp->oct_id;
				new_kp.scale_id = kp->scale_id;
				new_kp.orientation = kp->orientation;
				new_kp.v = kp->v;
				keyPoints.push_back(new_kp);
			}
		}
	}
	return keyPoints;
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
key_point* Sift::interpolateExtrema(int octave, int interval, int row, int column){
	int i = 0;
	int cur_interval = interval;
	int cur_row = row;
	int cur_col = column;
	Vec3f dD;
	Vec3f offset;
	double offset_c, offset_r, offset_i;
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

		offset_c = offset[0];
		offset_r = offset[1];
		offset_i = offset[2];

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
		// the keypoint's actual coordinate in the image
		double x = (cur_col + offset_c) * pow(2.0, octave);
		double y = (cur_row + offset_r) * pow(2.0, octave);
		key_point kp = { x, y, octave, cur_interval, 0.0, NULL };
		return &kp;
	}
}

// eliminate edge responses
bool Sift::isEdge(int octave, int interval, int row, int column) {
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

	//if the curvatures have different signs, the point is discarded as not being an extremum
	if (det <= 0)
		return true;

	// eliminates keypoints that have a ratio between the principal curvatures greater than the threshold (10 in Lowe's paper)
	if (trace * trace / det >= (SIFT_CURV_THR + 1) * (SIFT_CURV_THR + 1) / SIFT_CURV_THR)
		return true;

	return false;
}
