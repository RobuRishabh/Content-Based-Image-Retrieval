#ifndef PROJ2_DISTANCEMETRICS_H
#define PROJ2_DISTANCEMETRICS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Compute the sum of square difference of the two feature vectors.
 */
float sumOfSquareDifference(vector<float> &target, vector<float> &image);

/*
 * Compute the histogram intersection of the two feature vectors.
 */
float histogramIntersection(vector<float> &target, vector<float> &image);


#endif //PROJ2_DISTANCEMETRICS_H
