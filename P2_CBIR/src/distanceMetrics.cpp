#include <opencv2/opencv.hpp>
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/distanceMetrics.h"

using namespace cv;
using namespace std;

/*
Compute the sum of square Difference of the two feature vectors
*/
float sumOfSquareDifference(vector<float> &target, vector<float> &image){
	//It is an assertion to ensure that the sizes of the target and image vectors are equal. 
	//If the assertion fails, it will generate an error.
	CV_Assert(target.size() == image.size());
	float sum = 0;
	// iterates over the elements of the vectors.
	for(int i = 0; i < target.size(); i++){
		//calculates the squared difference between the corresponding elements of the target and image vectors. 
		//It squares the difference between target[i] and image[i] and adds it to the sum.
		sum += (target[i] - image[i]) * (target[i] - image[i]);
	}
	return sum;
}

/*
Compute the histogram intersection of the two feature vectors.
*/
float histogramIntersection(vector<float> &target, vector<float> &image){
	//is an assertion to ensure that the sizes of the target and image vectors are equal. 
	//If the assertion fails, it will generate an error. 
	//This is important because histogram intersection requires the two feature vectors to have the same size.
	CV_Assert(target.size() == image.size());
	//This variable will store the cumulative intersection value.
	float intersection = 0;
	//iterates over the elements of the vectors.
	for(int i = 0; i < target.size(); i++){
		//calculates the intersection value for the corresponding elements of the target and image vectors. 
		//It takes the minimum value between target[i] and image[i] and adds it to the intersection variable
		intersection += (min(target[i], image[i]));
	}
	return intersection;
}