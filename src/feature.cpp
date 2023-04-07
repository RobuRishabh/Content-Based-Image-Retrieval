//
// Rishabh Singh(NUID: 002767904     Email-id: singh.risha@northeastern.edu) 
// & 
// Aakash Singhal(NUID: 002761944    Email-id: singhal.aak@northeastern.edu)
//


#include <math.h>
#include <opencv2/opencv.hpp>
#include "feature.h"

using namespace std;

/*
 * after taking image.
 * Use the 9x9 square in the middle of the image as a gabor_feature vector.
 * Return the gabor_feature vector.
 */
vector<float> baseline_matching(cv::Mat& img) {
    cv::Mat square9X9;
    int x = img.cols / 2 - 4, y = img.rows / 2 - 4;
    square9X9 = img(cv::Rect(x, y, 9, 9)).clone();

    // returning the gabor_feature vector after converting the 9x9 matrix to a 1D vector
    return feature_vector(square9X9);
}

/*
 * after taking image.
 * Use a single normalized color histogram as the gabor_feature vector.
 * Return the gabor_feature vector.
 */
vector<float> histogram_matching(cv::Mat& img) {
    // calculate the range in each bin
    int range = 256 / 8; 

    // initialize histogram
    int histogram_size[] = { 8, 8, 8 };
    cv::Mat hist_feature = cv::Mat::zeros(3, histogram_size, CV_32F);

    // run a for-loop over the image and build a 3D histogram
    int i = 0, j = 0;
    while (i < img.rows) {
        j = 0;
        while (j < img.cols) {
            int b = img.at<cv::Vec3b>(i, j)[0] / range;
            int g = img.at<cv::Vec3b>(i, j)[1] / range;
            int r = img.at<cv::Vec3b>(i, j)[2] / range;
            hist_feature.at<float>(b, g, r)++;
            j++;
        }
        i++;
    }

    // L2 normalization of the histogram
    cv::normalize(hist_feature, hist_feature, 1, 0, cv::NORM_L2, -1, cv::Mat());

    // returning the gabor_feature vector after converting the 3D histogram into a 1D vector
    return feature_vector(hist_feature);
}

/*
 * after taking image.
 * split it into 2x2 grid
 * Use two or more color histograms of your choice as the gabor_feature vector.
 */
vector<float> mult_Hist(cv::Mat& img) {
    vector<float> mult_hist_feature;
    int x = img.cols / 2, y = img.rows / 2;
    int topX[] = { 0, x }; // Array containing the X-coordinates of the top left corner of each sub-image
    int topY[] = { 0, y }; // Array containing the Y-coordinates of the top left corner of each sub-image

    int i = 0, j = 0;
    while (i < 2) {
        while (j < 2) {
            cv::Mat ROI = img(cv::Rect(topX[i], topY[j], x, y)).clone();
            /* Get region of interest (ROI) as a separate image.
             Rect constructor takes in 4 arguments: x-coordinate of top left corner, y-coordinate of top left corner, width, height.
             The .clone() method is used to create a deep copy of the ROI,
             so that the original image is not modified. */
            vector<float> hist1 = histogram_matching(ROI);// Call histogram_matching function on the ROI and store the result in "hist1".
            mult_hist_feature.insert(mult_hist_feature.end(), hist1.begin(), hist1.end());// Concatenate the result of each sub-image into a single 1D vector.
            j++;
        }
        j = 0;
        i++;
    }

    return mult_hist_feature;
}

/*
 * after taking image.
 * Function texture_feature takes in an image as an input.
 * First the image is conerted to grayscale using cvtcolor then stored in grayscale.
 * 2D histogram with 8 bins for each dimension is initialized, with all elements zero and stored in histogram_feature.
 */
vector<float> texture_feature(cv::Mat& img) {
    cv::Mat grayscale;                      // convert image to grayscale
    cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);

    // compute gradient magnitude & orientation of grayscale image
    cv::Mat gradient_Magnitude = magnitude(grayscale);
    cv::Mat gradient_Orientation = orientation(grayscale);

    // initialize a 2D histogram
    int histogram_Size[] = { 8, 8 };
    cv::Mat hist_feature = cv::Mat::zeros(2, histogram_Size, CV_32F);

    // calculate the range of each bin in the histogram
    float magnitude_range = 400 / 8.0;
    float orientation_range = 2 * CV_PI / 8.0;

    // Build the 2D histogram by looping through the gradient magnitude and orientation
    int i = 0, j = 0;
    while (i < gradient_Magnitude.rows) {
        while (j < gradient_Magnitude.cols) {
            int mag_bin = (int)(gradient_Magnitude.at<float>(i, j) / magnitude_range);
            int orient_bin = (int)((gradient_Orientation.at<float>(i, j) + CV_PI) / orientation_range);
            hist_feature.at<float>(mag_bin, orient_bin)++;
            j++;
        }
        i++;
        j = 0;
    }


    // L2 normalize the histogram
    cv::normalize(hist_feature, hist_feature, 1, 0, cv::NORM_L2, -1, cv::Mat());

    // convert the 2D histogram into a 1D vector
    return feature_vector(hist_feature);
}

/*
 * after taking image.
 * Use a whole image color histogram and a whole image texture histogram as the gabor_feature vector.
 */
vector<float> texture_Color(cv::Mat& img) {
    // compute the texture gabor_feature of image
    vector<float> feature = texture_feature(img);
    // compute color gabor_feature of the image using previously made function histogram matching
    vector<float> color = histogram_matching(img);
    feature.insert(feature.end(), color.begin(), color.end());     //Concatenation of result 
    return feature;                                       // return gabor_feature
}

/*
 * after taking image.
 * Function gabor_texture takes in an image as an input.
 * First the image is conerted to grayscale then stored in grayscale.
 */
vector<float> gabor_texture(cv::Mat& img) {
    vector<float> gabor_feature;
    cv::Mat grayscale;
    cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
    float sigma[] = { 1.0, 2.0, 4.0 };   //different sigma values
    int i = 0;
    int size = sizeof(sigma) / sizeof(sigma[0]);

    while (i < size) {
        for (int temp = 0; temp < 10; temp++) {
            float orientation = temp * CV_PI / 8;
            // Gabor Kernel with sigma = value & orientation 
            cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(21, 21), sigma[i], orientation, 10.0, 0.5, 0, CV_32F);
            cv::Mat filteredImage;
            //vector<float> hist(9, 0);
            cv::filter2D(grayscale, filteredImage, CV_32F, gaborKernel);
            // calculate the mean and standard deviation of each filtered image
            cv::Scalar mean, stddev;
            cv::meanStdDev(filteredImage, mean, stddev);
            gabor_feature.push_back(mean[0]);
            gabor_feature.push_back(stddev[0]);
        }
        i++;
    }


    // L2 normalize the gabor_feature vector
    cv::normalize(gabor_feature, gabor_feature, 1, 0, cv::NORM_L2, -1, cv::Mat()); 

    return gabor_feature; // return gabor_feature 
}

/*
 * Extension ------ Gabor texture and color
 */
vector<float> gabor_texture_color(cv::Mat& img) {
    // compute the gabor texture of image using previously made function gabor_texture
    vector<float> gabor_texture_feature = gabor_texture(img);
    // compute color gabor_feature of the image using previously made function histogram matching
    vector<float> color = histogram_matching(img);
    gabor_texture_feature.insert(gabor_texture_feature.end(), color.begin(), color.end());  // Concatenation into 1D gabor_feature vector
    return gabor_texture_feature;   // return gabor_feature

}

/*
 * Extension -------- Multi-histogram Gabor texture and color
 * after taking image.
 * Use two or more color histograms of your choice as the gabor_feature vector.
 * Split it into 2 x 2 grids
 */
vector<float> multi_gabor_texture_color(cv::Mat& img) {
    vector<float> mult_gtc_feature;
    int x = img.cols / 2, y = img.rows / 2;
    int topX[] = { 0, x };
    int topY[] = { 0, y };
    int i = 0, j = 0;
    while (i < 2) {
        while (j < 2) {
            cv::Mat ROI = img(cv::Rect(topX[i], topY[j], x, y)).clone();
            /* Get region of interest (ROI) as a separate image.
             Rect constructor takes in 4 arguments: x-coordinate of top left corner, y-coordinate of top left corner, width, height.
             The .clone() method is used to create a deep copy of the ROI,
             so that the original image is not modified. */
            vector<float> hist1 = gabor_texture_color(ROI);// Call gabor_text_color function on the ROI and store the result in "hist1".
            mult_gtc_feature.insert(mult_gtc_feature.end(), hist1.begin(), hist1.end());// concatenate into a single 1D vector
            j++;
        }
        j = 0;
        i++;
    }

    return mult_gtc_feature; //return gabor_feature
}

/*
 * Compute sobelX filter
 * horizontal filter [-1, 0, 1], vertical filter [1, 2, 1]
 */
cv::Mat sobel_X(cv::Mat& img) {
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat node = cv::Mat::zeros(img.size(), CV_32F);
    //iteration through rows and cols - apply horizontal filter
    int i = 0;
    while (i < img.rows) {
        int j = 0;
        while (j < img.cols) {
            if (j > 0 && j < img.cols - 1) {
                node.at<float>(i, j) = -img.at<uchar>(i, j - 1) + img.at<uchar>(i, j + 1);
            }
            j++;
        }
        i++;
    }

    //iteration through rows and cols - apply vertical filter
     i = 0;
    while (i < node.rows) {
        int j = 0;
        while (j < node.cols) {
            if (i == 0) {
                dst.at<float>(i, j) = (node.at<float>(i + 1, j) + 2 * node.at<float>(i, j) + node.at<float>(i + 1, j)) / 4;
            }
            else if (i == node.rows - 1) {
                dst.at<float>(i, j) = (node.at<float>(i - 1, j) + 2 * node.at<float>(i, j) + node.at<float>(i - 1, j)) / 4;
            }
            else {
                dst.at<float>(i, j) = (node.at<float>(i - 1, j) + 2 * node.at<float>(i, j) + node.at<float>(i + 1, j)) / 4;
            }
            j++;
        }
        i++;
    }

    return dst;
}

/*
 * Compute sobelY filter
 * horizontal [1, 2, 1], vertical [-1, 0, 1]
 */
cv::Mat sobel_Y(cv::Mat& img) {
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat node = cv::Mat::zeros(img.size(), CV_32F);

    // iteration through rows and cols - apply horizontal filter
    int i = 0;
    int j = 0;

    while (i < img.rows) {
        j = 0;
        while (j < img.cols) {
            if (j == 0) {
                node.at<float>(i, j) = (img.at<uchar>(i, j + 1) + 2 * img.at<uchar>(i, j) + img.at<uchar>(i, j + 1)) / 4;
            }
            else if (j == img.cols - 1) {
                node.at<float>(i, j) = (img.at<uchar>(i, j - 1) + 2 * img.at<uchar>(i, j) + img.at<uchar>(i, j - 1)) / 4;
            }
            else {
                node.at<float>(i, j) = (img.at<uchar>(i, j - 1) + 2 * img.at<uchar>(i, j) + img.at<uchar>(i, j + 1)) / 4;
            }
            j++;
        }
        i++;
    }

    // iteration through rows and cols - apply vertical filter
     i = 0, j = 0;
    while (i < node.rows) {
        j = 0;
        while (j < node.cols) {
            if (i > 0 && i < node.rows - 1) {
                dst.at<float>(i, j) = -node.at<float>(i - 1, j) + node.at<float>(i + 1, j);
            }
            j++;
        }
        i++;
    }
    return dst;
}

/*
 * calculate the gradient magnitude
 */
cv::Mat magnitude(cv::Mat& img) {
    // calculate sobelX and sobelY previously defined
    cv::Mat s_x = sobel_X(img);
    cv::Mat s_y = sobel_Y(img);

    // calculate gradient magnitude
    cv::Mat dst;
    sqrt(s_x.mul(s_x) + s_y.mul(s_y), dst);

    return dst;
}

/*
 * calculate the gradient orientation
 */
cv::Mat orientation(cv::Mat& image) {
    // calculate sobelX and sobelY using previously defined function
    cv::Mat s_x = sobel_X(image);
    cv::Mat s_y = sobel_Y(image);

    // calculate orientation
    cv::Mat dst(image.size(), CV_32F);
    int i = 0, j = 0;
    while (i < image.rows) {
        while (j < image.cols) {
            dst.at<float>(i, j) = atan2(s_y.at<float>(i, j), s_x.at<float>(i, j));
            j++;
        }
        j = 0;
        i++;
    }


    return dst;
}

/*
 * after taking image.
 */
cv::Mat middle(cv::Mat& img) {
    // Computing the center cols and row
    int center_col = img.cols / 3;
    int center_row = img.rows / 3;
    cv::Mat middle = img(cv::Rect (center_col, center_row, center_col, center_row)).clone();
    return middle;  //Return extracted middle part of img
}

/*
 *  
 * Convert a Mat to a 1D vector<float>
 */
vector<float> feature_vector(cv::Mat& img) {
    //reshape matrix into 1D matrix
    cv::Mat flat = img.reshape(1, img.total() * img.channels());
    //convert matrix by flatenning to 32-bit floating point values 
    flat.convertTo(flat, CV_32F);
    if (img.isContinuous()) {
        return flat;
    }
    else {
        return flat.clone();
    }
}

