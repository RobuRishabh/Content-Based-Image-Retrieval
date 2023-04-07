//
// Rishabh Singh(NUID: 002767904     Email-id: singh.risha@northeastern.edu) 
// & 
// Aakash Singhal(NUID: 002761944    Email-id: singhal.aak@northeastern.edu)
//

#include "dirent.h"
#include <string.h>
#include <utility>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "feature.h"
#include "distance_metric.h"

using namespace std;

#if 1

//int main(int argc, char* argv[]) {
int main() {
    cv::Mat target_img;
    vector<float> target_feature;

 /*
 * Here we are taking 5 inputs in argv[].
 * Input1: Path to the target image written as argv1
 * Input2: feature set written as argv2
 * Input3: Path where we want our output csv file to get generated as output.csv written as argv3
 * Input4: Distance metrics written as argv4
 * Input5: Our top N matches written as argv5
 */

    /*

    MODIFY COMMAND LINE ARGUMENTS HERE
    */
    std::string argv0 = "a.out";
    std::string argv1 = "E:\\PRCV Assignments\\Content Based Image Retrieval\\olympus\\olympus\\pic.0280.jpg";
    std::string argv2 = "mult_hist_tc_mid";
    std::string argv3 = "E:\\PRCV Assignments\\Content Based Image Retrieval\\Content Based Image Retrieval\\output.csv";
    std::string argv4 = "hist_int";
    std::string argv5 = "8";

    int argc = 6;
    char* argv[6];
    argv[0] = _strdup(argv0.c_str());
    argv[1] = _strdup(argv1.c_str());
    argv[2] = _strdup(argv2.c_str());
    argv[3] = _strdup(argv3.c_str());
    argv[4] = _strdup(argv4.c_str());
    argv[5] = _strdup(argv5.c_str());


    // check for sufficient arguments
    if (argc < 6) {
        cout << "Enter Valid Input" << endl;
        exit(-1);
    }

    // Loads the target image from file
    target_img = cv::imread(argv[1]);
    cout << target_img.size() << endl;
    if (target_img.empty()) {
        cout << "Enter Valid target image" << endl;
        exit(-1);
    }

    // Compare the features of target image
    if (!strcmp(argv[2], "base_match")) { // baseline_matching
        target_feature = baseline_matching(target_img);
    }
    else if (!strcmp(argv[2], "hist_match")) { // histogram_matching
        target_feature = histogram_matching(target_img);
    }
    else if (!strcmp(argv[2], "mult_hist_match")) { // multi_histograms_matching
        target_feature = mult_Hist(target_img);
    }
    else if (!strcmp(argv[2], "text_col")) {                // Texture and color
        target_feature = texture_Color(target_img);
    }
    else if (!strcmp(argv[2], "tc_mid")) {    // Custom Design: texture and color on middle part of the image
        cv::Mat mid = middle(target_img);
        target_feature = texture_Color(mid);
    }
    else if (!strcmp(argv[2], "mult_hist_tc")) {            // Extension1: multi_histograms of Gabor texture and color
        target_feature = multi_gabor_texture_color(target_img);
    }
    else if (!strcmp(argv[2], "mult_hist_tc_mid")) {        // Extension2: Gabor texture and color on middle part
        cv::Mat mid = middle(target_img);
        target_feature = gabor_texture_color(mid);
    }
    else {
        cout << "Enter Valid feature type" << endl;
        exit(-1);
    }

    cout << "Task implemented successfully" << endl;

    // reading features and names of the images
    vector<char*> img_Name;
    vector<vector<float>> img_feature;
    FILE* fp = fopen(argv[3], "r");
    if (fp) {
        read_image_data_csv(argv[3], img_Name, img_feature);
    }

    // Finding the distances between target image and feature vector image
    vector<pair<string, float>> dist;
    float d;
    pair<string, float> img_dist;
    cout << "image size: " << img_Name.size() << endl;
    for (int i = 0; i < img_Name.size(); i++) {
        if (!strcmp(argv[4], "sum_sq_dist")) {
            // sum of squared difference as distance metric
            cout << "target: " << target_img.size() << "\t imgFeature: " << img_feature[i].size() << img_Name[i] << endl;
            d = sum_Of_Sq_Diff(target_feature, img_feature[i]);
            img_dist = make_pair(img_Name[i], d);
            dist.push_back(img_dist);
            // sort the vector of distances in ascending order
            sort(dist.begin(), dist.end(), [](auto& left, auto& right) {
                return left.second < right.second;
                });
        }
        else if (!strcmp(argv[4], "hist_int")) {
            // histogram intersection as distance metric
            d = hist_Intersection(target_feature, img_feature[i]);
            img_dist = make_pair(img_Name[i], d);
            dist.push_back(img_dist);
            // sort the vector of distances in descending order
            sort(dist.begin(), dist.end(), [](auto& left, auto& right) {
                return left.second > right.second;
                });
        }
        else {
            cout << "No such distance metrics." << endl;
            exit(-1);
        }
    }

    // get the top 4 or N image matches, excluding the target image itself

    int N = stoi(argv[5]);
    for (int i = 0; i < dist.size() && N > 0; i++) {
        cv::Mat image = cv::imread(dist[i].first);
        if (image.size() != target_img.size() || (sum(image != target_img) != cv::Scalar(0, 0, 0, 0))) {
            cout << dist[i].first << endl;
            N--;
        }
    }

    return 0;
}
#endif