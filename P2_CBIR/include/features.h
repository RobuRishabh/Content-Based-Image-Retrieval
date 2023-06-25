#ifndef PROJ2_FEATURES_H
#define PROJ2_FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Given an image.
 * Use the 9x9 square in the middle of the image as a feature vector.
 * Return the feature vector.
 */
vector<float> baseline(Mat &image);

/*
 * Given an image.
 * Use the whole image RGB histogram with 8 bins for each of RGB as the feature vector.
 * Return the feature vector.
 */
vector<float> histogram(Mat &image);

/*
 * Given an image.
 * Split it into 2 x 2 grids
 * Calculate the histogram for each part, using RGB histogram with 8 bins for each of RGB
 * concatenate the result of each part into a single 1D vector and return the vector
 */
vector<float> multiHistogram(Mat &image);

/*
 * Given an image.
 * Convert it to grayscale and compute a 2D histogram of gradient magnitude and orientation
 * Using 8 bins for each dimension
 * the max value for magnitude is sqrt(2) * max(sx, sy), which is approximately 1.4 * 255 = 400
 * the max value for orientation is -PI to PI
 */
vector<float> texture(Mat &image);

/*
 * Given an image.
 * Calculate a 2D histogram of gradient orientation and magnitude and another 3D histogram of BGR color
 * concatenate the result of each part into a single 1D vector and return the vector
 */
vector<float> textureAndColor(Mat &image);

/*
 * Given an image, convert it to grayscale.
 * Apply 48 gabor filters on it (5 scales and 16 orientations)
 * For each result, calculate the mean and standard deviation of it
 * Concatenate the result into a 1D vector and return the vector
 */
vector<float> gaborTexture(Mat &image);

/*
 * Given an image.
 * Calculate a feature vector using gabor filters using `gaborTexture()`
 * Calculate another feature vector using color information using `histogram()`
 * Concatenate the two features and return the vector
 */
vector<float> gaborTextureAndColor(Mat &image);

/*
 * Given an image
 * Split it into 2 x 2 grids
 * Calculate a feature vector for each part using `gaborTextureAndColor()`
 * Concatenate the result into a 1D vector and return it
 */
vector<float> multiGaborTextureAndColor(Mat &image);

/*
 * Take a single-channel image
 * Compute sobelX, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal filter [-1, 0, 1], vertical filter [1, 2, 1]
 */
Mat sobelX(Mat &image);

/*
 * Take a single-channel image
 * Compute sobelY, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal [1, 2, 1], vertical [-1, 0, 1]
 */
Mat sobelY(Mat &image);

/*
 * Take a single-channel image,
 * calculate the gradient magnitude of it.
 */
Mat magnitude(Mat &image);

/*
 * Take a single-channel image,
 * calculate the gradient orientation of it.
 */
Mat orientation(Mat &image);

/*
 * Given an image.
 * Treat it as a 3 * 3 grid, and take the middle part
 */
Mat getMiddle(Mat &image);

/*
 * Convert a Mat to a 1D vector<float>
 */
vector<float> matToVector(Mat &m);

#endif //PROJ2_FEATURES_H
