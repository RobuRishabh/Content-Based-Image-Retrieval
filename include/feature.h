#pragma once
//
// Rishabh Singh(NUID: 002767904     Email-id: singh.risha@northeastern.edu) 
// & 
// Aakash Singhal(NUID: 002761944    Email-id: singhal.aak@northeastern.edu)
//


#ifndef FEATURE_H
#define FEATURE_H
#include <opencv2/opencv.hpp>

using namespace std;
vector<float> baseline_matching(cv::Mat& img);

vector<float> histogram_matching(cv::Mat& img);

vector<float> mult_Hist(cv::Mat& img);

vector<float> texture_feature(cv::Mat& img);

vector<float> texture_Color(cv::Mat& img);

vector<float> gabor_texture(cv::Mat& img);

vector<float> gabor_texture_color(cv::Mat& img);

vector<float> multi_gabor_texture_color(cv::Mat& img);

cv::Mat sobel_X(cv::Mat& img);

cv::Mat sobel_Y(cv::Mat& img);

cv::Mat magnitude(cv::Mat& img);

cv::Mat orientation(cv::Mat& img);

cv::Mat middle(cv::Mat& img);

vector<float> feature_vector(cv::Mat& img);

#endif
