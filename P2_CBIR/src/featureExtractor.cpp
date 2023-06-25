#include <string.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/features.h"
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/distanceMetrics.h"
#include "/home/roburishabh/PRCV_Projects/P2_CBIR/include/csv_util.h"

using namespace cv;
using namespace std;

/*
 * Take three inputs.
 * The first is the path to a directory of images.
 * The second is the feature set.
 * The third is a path to the csv file where the feature vector for each image is written.
 *
 * Create a file in the same directory of this program
 * Write the feature vector for each image to the file
 */
int main(int argc, char *argv[]) {
    char dirname[256];
    DIR *dirp;
    struct dirent *dp;

    // check for sufficient arguments
    if (argc < 4) {
        cout << "Wrong input." << endl;
        exit(-1);
    }

    // If the csv file already exists, no need to recompute the feature vectors
    FILE *fp = fopen( argv[3], "r" );
    if(fp) {
        cout << "csv file already exists." << endl;
        return 0;
    }

    // get the directory path
    strcpy(dirname, argv[1]);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        cout << "Cannot open directory " << dirname << endl;
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            char buffer[256];
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            Mat image = imread(buffer);
            vector<float> imageFeature;
            if (!strcmp(argv[2], "b")) { // baseline
                imageFeature = baseline(image);
            } else if (!strcmp(argv[2], "c")) { // color
                imageFeature = histogram(image);
            } else if (!strcmp(argv[2], "mc")) { // multi histograms of color
                imageFeature = multiHistogram(image);
            } else if (!strcmp(argv[2], "t")) { // texture
                imageFeature = texture(image);
            } else if (!strcmp(argv[2], "tc")) { // texture and color
                imageFeature = textureAndColor(image);
            } else if (!strcmp(argv[2], "midc")) { // color on middle part
                Mat middle = getMiddle(image);
                imageFeature = histogram(middle);
            } else if (!strcmp(argv[2], "midt")) { // texture on middle part
                Mat middle = getMiddle(image);
                imageFeature = texture(middle);
            } else if (!strcmp(argv[2], "midtc")) { // texture and color on middle part
                Mat middle = getMiddle(image);
                imageFeature = textureAndColor(middle);
            } else if (!strcmp(argv[2], "gt")) { // Gabor texture
                imageFeature = gaborTexture(image);
            } else if (!strcmp(argv[2], "gtc")) { // Gabor texture and color
                imageFeature = gaborTextureAndColor(image);
            } else if (!strcmp(argv[2], "mgtc")) { // multi histograms of Gabor texture and color
                imageFeature = multiGaborTextureAndColor(image);
            } else if (!strcmp(argv[2], "midgtc")) { // Gabor texture and color on middle part
                Mat middle = getMiddle(image);
                imageFeature = gaborTextureAndColor(middle);
            } else {
                cout << "No such feature type." << endl;
                exit(-1);
            }
            append_image_data_csv(argv[3], buffer, imageFeature);
        }
    }
    return 0;
}