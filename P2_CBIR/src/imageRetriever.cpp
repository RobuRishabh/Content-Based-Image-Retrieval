#include <dirent.h>
#include <string.h>
#include <utility>
#include <opencv2/opencv.hpp>
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/csv_util.h"
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/features.h"
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/distanceMetrics.h"

using namespace cv;
using namespace std;

/*
 * Take five inputs.
 * The first is the path to the target image.
 * The second is the feature set.
 * The third is a path to the csv file where the feature vector for each image is written.
 * The forth is distance metrics.
 * The fifth is the desired number of matches N.
 *
 * Compute the features for the target image, reads the feature vector file,
 * and identifies the top N matches.
 */
int main(int argc, char *argv[]) {
    Mat target;
    vector<float> targetFeature;

    // check for sufficient arguments
        if (argc < 6) {
            cout << "Wrong input." << endl;
            exit(-1);
        }

    // get target image
    target = imread(argv[1]);
    if (target.empty()) {
        cout << "No target image." << endl;
        exit(-1);
    }

    // compute features for the target image
    if (!strcmp(argv[2], "b")) { // baseline
        targetFeature = baseline(target);
    } else if (!strcmp(argv[2], "c")) { // color
        targetFeature = histogram(target);
    } else if (!strcmp(argv[2], "mc")) { // multi histograms of color
        targetFeature = multiHistogram(target);
    } else if (!strcmp(argv[2], "t")) { // texture
        targetFeature = texture(target);
    } else if (!strcmp(argv[2], "tc")) { // texture and color
        targetFeature = textureAndColor(target);
    } else if (!strcmp(argv[2], "midc")) { // color on middle part
        Mat middle = getMiddle(target);
        targetFeature = histogram(middle);
    } else if (!strcmp(argv[2], "midt")) { // texture on middle part
        Mat middle = getMiddle(target);
        targetFeature = texture(middle);
    } else if (!strcmp(argv[2], "midtc")) { // texture and color on middle part
        Mat middle = getMiddle(target);
        targetFeature = textureAndColor(middle);
    } else if (!strcmp(argv[2], "gt")) { // Gabor texture
        targetFeature = gaborTexture(target);
    } else if (!strcmp(argv[2], "gtc")) { // Gabor texture and color
        targetFeature = gaborTextureAndColor(target);
    } else if (!strcmp(argv[2], "mgtc")) { // multi histograms of Gabor texture and color
        targetFeature = multiGaborTextureAndColor(target);
    } else if (!strcmp(argv[2], "midgtc")) { // Gabor texture and color on middle part
        Mat middle = getMiddle(target);
        targetFeature = gaborTextureAndColor(middle);
    } else {
        cout << "No such feature type." << endl;
        exit(-1);
    }

    // read features for the images
    vector<char *> imageNames;
    vector<vector<float>> imageFeatures;
    FILE *fp = fopen( argv[3], "r" );
    if(fp) {
        read_image_data_csv(argv[3], imageNames, imageFeatures);
    }

    // compute the distances between the target and each image
    vector<pair<string, float>> distances;
    float d;
    pair<string, float> imageAndDistance;
    for (int i = 0; i < imageNames.size(); i++) {
        if (!strcmp(argv[4], "sd")) {
            // sum of square difference
            d = sumOfSquareDifference(targetFeature, imageFeatures[i]);
            imageAndDistance = make_pair(imageNames[i], d);
            distances.push_back(imageAndDistance);
            // sort the vector of distances in ascending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right) {
            return left.second < right.second;
            });
        } else if (!strcmp(argv[4], "hi")) {
            // histogram intersection
            d = histogramIntersection(targetFeature, imageFeatures[i]);
            imageAndDistance = make_pair(imageNames[i], d);
            distances.push_back(imageAndDistance);
            // sort the vector of distances in descending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right) {
            return left.second > right.second;
            });
        } else {
            cout << "No such distance metrics." << endl;
            exit(-1);
        }
    }

    // get the first N matches, exclude the target itself
    int N = 0, i = 0;
    while (N < stoi(argv[5])) {
        Mat image = imread(distances[i].first);
        if (image.size() != target.size() || (sum(image != target) != Scalar(0,0,0,0))) {
            cout << distances[i].first << endl;
            i++;
            N++;
        } else {
            i++;
        }
    }
    return 0;
}
