#pragma once
//
// Rishabh Singh(NUID: 002767904     Email-id: singh.risha@northeastern.edu) 
// & 
// Aakash Singhal(NUID: 002761944    Email-id: singhal.aak@northeastern.edu)
//

#ifndef DISTANCE_METRIC_H
#define DISTANCE_METRIC_H

#include <opencv2/opencv.hpp>

using namespace std;

float sum_Of_Sq_Diff(vector<float>& target, vector<float>& img);

float hist_Intersection(vector<float>& target, vector<float>& img);


#endif
