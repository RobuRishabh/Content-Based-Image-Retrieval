//
// Rishabh Singh(NUID: 002767904     Email-id: singh.risha@northeastern.edu) 
// & 
// Aakash Singhal(NUID: 002761944    Email-id: singhal.aak@northeastern.edu)
//

#include <opencv2/opencv.hpp>
#include "distance_metric.h"


using namespace std;

/*
 * Compute the sum of square difference between two feature vectors.
 */
float sum_Of_Sq_Diff(vector<float>& target, vector<float>& image) {
    CV_Assert(target.size() == image.size()); // compare if the two features have the same size
    float sum_sd = 0;
    // Iterate through each element of the two feature vectors
    int i = 0;
    while (i < target.size()) {
        sum_sd += (target[i] - image[i]) * (target[i] - image[i]);
        i++;
    }
    return sum_sd;
}

/*
 * Compute the histogram intersection of the two feature vectors.
 */
float hist_Intersection(vector<float>& target, vector<float>& image) {
    CV_Assert(target.size() == image.size()); // compare if two features have the same size
    float intersection = 0;
    // Iterate through each element of the two feature vectors
    int i = 0;
    while (i < target.size()) {
        // Compute the minimum value between the two elements and add it to the intersection sum
        intersection += (min(target[i], image[i]));
        i++;
    }

    return intersection;
}