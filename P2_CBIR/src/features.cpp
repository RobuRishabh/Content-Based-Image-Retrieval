#include <math.h>
#include <opencv2/opencv.hpp>
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/features.h"

using namespace cv;
using namespace std;
/*
Given an image
Use the 9x9 square in the middle of the image as a feature vector.
Return the feature vector.
*/

std::vector<float> baseline(Mat &image) {
	//declares a cv::Mat object named middle9X9 to store the 9x9 patch.
    cv::Mat middle9X9;
    //calculate the starting point (top-left corner) of the 9x9 patch. 
    //It takes half the width and height of the input image and 
    //subtracts 4 to ensure that the patch is centered.
    int x = image.cols / 2 - 4;
    int y = image.rows / 2 - 4;
    //extracts the 9x9 patch from the image using the cv::Rect constructor. 
    //The Rect specifies the starting point (x, y) and the size (9x9) of the region of interest (ROI). 
    //The clone() function is used to create a copy of the extracted patch and assign it to the middle9X9 matrix.
    middle9X9 = image(Rect(x, y, 9, 9)).clone();

    // convert the 9 x 9 mat to a 1D vector
    return matToVector(middle9X9);
}
/*
 * Given an image.
 * Use the whole image RGB histogram with 8 bins for each of RGB as the feature vector.
 * Return the feature vector.
 */
std::vector<float> histogram(Mat &image) {
    //Calculate the range of values that will fall into each bin of the histogram. 
    //In this case, the image's pixel values range from 0 to 255, and we divide it
    // into 8 equal-sized bins, so each bin represents a range of 32 values.
    int range = 256 / 8; 

    //initialize a 3D histogram with 8 bins for each of the RGB channels. 
    int histSize[] = {8, 8, 8};
    //The histSize array represents the number of bins for each channel.
    //We create a Mat object named feature to store the histogram, initialized with all values set to zero.
    Mat feature = Mat::zeros(3, histSize, CV_32F);

    //We loop over each pixel in the image. For each pixel, we access its RGB values using 
    //image.at<Vec3b>(i, j). Then, we divide each channel value by the range to determine 
    //which bin it belongs to. We increment the corresponding bin in the histogram feature by one.
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int b = image.at<Vec3b>(i, j)[0] / range;
            int g = image.at<Vec3b>(i, j)[1] / range;
            int r = image.at<Vec3b>(i, j)[2] / range;
            feature.at<float>(b, g, r)++;
        }
    }
    //After constructing the histogram, we normalize it using L2 normalization. 
    //This step ensures that the histogram values are scaled between 0 and 1, 
    //representing the relative frequencies of color occurrences.
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

    //convert the 3D histogram feature into a 1D vector using the matToVector function
    //and return it as the feature vector.
    return matToVector(feature);
}

/*
Given an image
Split it into 2x2 grid
Calculate the hstogram fro each part, using RGB histogram with 8 bins for each of RGB
Concatenate the result of each part into a single 1D vector and return the vector
*/
std::vector<float>multiHistogram(Mat &image){
    vector<float> feature;
    //calculate the center point of the image by dividing the number of columns (cols) and rows (rows) by 2 
    int x = image.cols/2, y = image.rows/2;
    //define two arrays topX and topY to store the starting coordinates of each grid. 
    //In this case, the image is split into a 2x2 grid, so each grid will have half the width (x) and half the height (y).
    int topX[] = {0, x};
    int topY[] = {0, y};
    //iterate over each grid in the 2x2 grid
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            //define a region of interest (ROI) m using the Rect function. 
            //The Rect takes the starting X and Y coordinates (topX[i], topY[j]) and the width (x) and height (y) of the region.
            //We then clone the ROI m to ensure that we have a separate copy.
            cv::Mat m =image(Rect(topX[i], topY[j], x , y)).clone();
            //calculate the feature vector v for the current ROI using the histogram function
            std::vector<float> v = histogram(m);
            //Finally, we concatenate the feature vector v to the feature vector using the insert function. 
            //This will append the elements of v at the end of feature.
            feature.insert(feature.end(), v.begin(), v.end());
        }
    }
    //return the concatenated feature vector.
    return feature;
}

/*
Given an image
Convert it to grayscale and compute a 2D histogram of gradient magnitude and orientation
Using 8 bins for each dimensions
The max value for magnitude is sqrt(2)*max(sx, sy), which is approximately 1.44 * 255 = 400
The max value for orientation is -PI to PI
*/
std::vector<float> texture(Mat &image){
    cv::Mat grayscale;
    //input image is converted to grayscale using the cvtColor function from OpenCV. 
    //This is necessary to calculate the gradient magnitude and orientation.
    cv::cvtColor(image, grayscale, COLOR_BGR2GRAY);
    //magnitude function is called with the grayscale image as input to calculate the gradient magnitude. 
    //This function calculates the magnitude of the gradients using the Sobel operator.
    cv::Mat imageMagnitude = magnitude(grayscale);
    //orientation function is called with the grayscale image as input to calculate the gradient orientation. 
    //This function calculates the orientation of the gradients using the Sobel operator.
    cv::Mat imageOrientation = orientation(grayscale);
    //An empty 2D histogram is created using Mat::zeros. 
    //The histogram has 8 bins along each dimension, representing the texture information.
    int histSize[] = {8, 8};
    cv::Mat feature = Mat::zeros(2, histSize, CV_32F);
    //Calculate the range in each bin
    float rangeMagnitude = 400 / 8.0;
    float rangeOrientation = 2* CV_PI / 8.0;

    //A loop is used to iterate over each pixel in the gradient magnitude and orientation images. 
    //The values are divided by their respective ranges and converted to integers to determine the corresponding bin indices. 
    //The histogram bins at those indices are incremented.
    for(int i=0; i<imageMagnitude.rows; i++){
        for(int j=0; j<imageMagnitude.cols; j++){
            int m = imageMagnitude.at<float>(i,j) / rangeMagnitude;
            int o = imageOrientation.at<float>(i,j) + CV_PI /rangeOrientation;
            feature.at<float>(m,o)++;
        }
    }
    //The calculated histogram is L2 normalized
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());
    //2D histogram is converted into a 1D vector using the matToVector function.
    //It flattens the histogram matrix row by row and returns it as a vector of floats.
    return matToVector(feature);

}

/* 
Given an image
Calculate a 2D histogram of gradient magnitude and orientation and another 3D histogram of BGR color
Concatenate the result of each part into a single 1D vectorand returns the vector
*/
std::vector<float>textureAndColor(Mat &image){
    //texture function is called with the input image as a parameter, and its returned feature vector is assigned to the variable feature. 
    //This calculates the texture feature vector of the image.
    std::vector<float>feature = texture(image);
    //histogram function is called with the input image as a parameter, and its returned feature vector is assigned to the variable color. 
    //This calculates the color feature vector of the image.
    std::vector<float>color = histogram(image);
    //insert function is used to concatenate the color vector at the end of the feature vector. This combines the texture and color feature vectors into a single vector.
    feature.insert(feature.end(), color.begin(), color.end());
    //concatenated feature vector is returned
    return feature;
}

/*
Given an image, convert it to grayscale
Apply 48 gabor filter on it (5 scales and 16 orientations)
For each results, calculate the mean and standard deviation of it
Concatenate the result into a 1D vector and return the vector
*/
std::vector<float>gaborTexture(Mat &image){
    vector<float> feature;
    cv::Mat grayscale;
    //convert image to grayscale
    cv::cvtColor(image, grayscale, COLOR_BGR2GRAY);
    //It declares an array sigmaValue that contains three different sigma values for the Gabor filters.
    float sigmaValue[] = {1.0, 2.0, 4.0};
    //loop over each sigma value in the sigmaValue array
    for(auto s : sigmaValue){
        //starts a loop that iterates 16 times for each orientation of the Gabor filter
        for(int k=0; k<16; k++){
            //calculates the orientation of the Gabor filter based on the loop variable k.
            float t = k * CV_PI / 8;
            //generates a Gabor kernel using the getGaborKernel function with specified parameters such as size, sigma, orientation, and other properties.
            cv::Mat gaborKernel = getGaborKernel(Size(31, 31), s, t, 10.0, 0.5, 0, CV_32F);
            //declares a variable filteredImage of type Mat to store the result of applying the Gabor filter.
            cv::Mat filteredImage;
            //cv::vector<float> hist(9,0);
            //It applies the Gabor filter represented by gaborKernel to the grayscale image grayscale using the filter2D function.
            filter2D(grayscale, filteredImage, CV_32F, gaborKernel);
            Scalar mean, stddev;
            //calculates the mean and standard deviation of the filtered image using the meanStdDev function.
            meanStdDev(filteredImage, mean, stddev);
            //adds the mean value of the filtered image to the feature vector.
            feature.push_back(mean[0]);
            //adds the standard deviation value of the filtered image to the feature vector.
            feature.push_back(stddev[0]);
        }
    }
    //performs L2 normalization on the feature vector, scaling it to have a unit norm.
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());
    return feature; 
}

/*
Given an image
Calculate a feature vector using Gabor filters using "gaborTexture()"
Calculate another feature vector using color information using histogram()
Concatenate the two feature and return the vector
*/
std::vector<float>gaborTextureAndColor(Mat &image){
    //calls the gaborTexture function to calculate a feature vector using Gabor filters on the input image. 
    //The calculated feature vector is stored in the feature vector.
    vector<float> feature = gaborTexture(image);
    //calls the histogram function to calculate a feature vector using color information from the input image. 
    //The calculated feature vector is stored in the color vector.
    vector<float> color =histogram(image);
    //concatenates the color vector at the end of the feature vector. 
    //This combines the two feature vectors into a single feature vector.
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;
 }

/*
Given an image
Split it into 2x2 grids
Calculate a feature vector for each part using "GaborTextureAndColor()"
Concatenate the result into a 1D vector and return it
*/
std::vector<float>multiGaborTextureAndColor(Mat &image){
    //initializes an empty vector feature to store the concatenated feature vectors.
    std::vector<float> feature;
    //calculates the dimensions of each grid. The image is split into 2 x 2 grids, 
    //x represents half the width of the image, and y represents half the height.
    int x = image.cols / 2, y = image.rows / 2;
    //two arrays topX and topY to specify the top-left corner coordinates of each grid. 
    //In this case, there are two grids: one starting from (0, 0) and the other starting from (x, y).
    int topX[] = {0, x};
    int topY[] = {0 ,y};
    //iterate over the two grids
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            //extracts the region of interest (ROI) from the image based on the current grid's coordinates. 
            //It creates a new Mat object m that represents the extracted region.
            cv::Mat m =image(Rect(topX[i], topY[j], x, y)).clone();
            //calls the gaborTextureAndColor function to calculate a feature vector for the extracted region.
            std::vector<float> v = gaborTextureAndColor(m);
            //concatenates the calculated feature vector v at the end of the feature vector. 
            //This combines the feature vectors of all the grid regions into a single feature vector.
            feature.insert(feature.end(), v.begin(), v.end());
        }
    }
    return feature;
}

/*
 * Take a single-channel image
 * Compute sobelX, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal filter [-1, 0, 1], vertical filter [1, 2, 1]
 */
Mat sobelX(Mat &image) {
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (j > 0 && j < image.cols - 1) {
                temp.at<float>(i, j) = -image.at<uchar>(i, j - 1) + image.at<uchar>(i, j + 1);
            }
        }
    }
    // apply vertical filter
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i == 0) {
                dst.at<float>(i, j) = (temp.at<float>(i + 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            } else if (i == temp.rows - 1) {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i - 1, j)) / 4;
            } else {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            }
        }
    }
    return dst;
}

/*
 * Take a single-channel image
 * Compute sobelY, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal [1, 2, 1], vertical [-1, 0, 1]
 */
Mat sobelY(Mat &image) {
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (j == 0) {
                temp.at<float>(i, j) = (image.at<uchar>(i, j + 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1)) / 4;
            } else if (j == image.cols - 1) {
                temp.at<float>(i, j) = (image.at<uchar>(i, j - 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j - 1)) / 4;
            } else {
                temp.at<float>(i, j) = (image.at<uchar>(i, j - 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1)) / 4;
            }
        }
    }
    // apply vertical filter
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i > 0 && i < temp.rows - 1) {
                dst.at<float>(i, j) = -temp.at<float>(i - 1, j) + temp.at<float>(i + 1, j);
            }
        }
    }
    return dst;
}

/*
 * Take a single-channel image,
 * calculate the gradient magnitude of it.
 */
Mat magnitude(Mat &image) {
    // calculate sobelX and sobelY
    Mat sx = sobelX(image);
    Mat sy = sobelY(image);

    // calculate gradient magnitude
    Mat dst;
    sqrt(sx.mul(sx) + sy.mul(sy), dst);

    return dst;
}

/*
 * Take a single-channel image,
 * calculate the gradient orientation of it.
 */
Mat orientation(Mat &image) {
    // calculate sobelX and sobelY
    Mat sx = sobelX(image);
    Mat sy = sobelY(image);

    // calculate orientation
    Mat dst(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            dst.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
        }
    }

    return dst;
}

/*
 * Given an image.
 * Treat it as a 3 * 3 grid, and take the middle part
 */
Mat getMiddle(Mat &image) {
    int x = image.cols / 2, y = image.rows / 2;
    Mat middle = image(Rect(x, y, x, y)).clone();
    return middle;
}

/*
 * Convert a Mat to a 1D vector<float>
 */
vector<float> matToVector(Mat &m) {
    Mat flat = m.reshape(1, m.total() * m.channels());
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}