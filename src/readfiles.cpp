/*
  Bruce A. Maxwell
  S21
  
  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "dirent.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include "csv_util.h"
#include "feature.h"
#include "distance_metric.h"



using namespace std;

/*
  Given a directory on the command line, scans through the directory for image files.
  /*
  
  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
#if 0

// int main(int argc, char *argv[]) 
int main() {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  /*
  
   MODIFY COMMAND LINE ARGUMENTS HERE
  */
  std::string argv0 = "a.out";
  std::string argv1 = "E:\\PRCV Assignments\\Content Based Image Retrieval\\olympus\\olympus";
  std::string argv2 = "mult_hist_tc_mid";
  std::string argv3 = "E:\\PRCV Assignments\\Content Based Image Retrieval\\Content Based Image Retrieval\\output.csv";

  int argc = 4;
  char* argv[4];
  argv[0] = _strdup(argv0.c_str());
  argv[1] = _strdup(argv1.c_str());
  argv[2] = _strdup(argv2.c_str());
  argv[3] = _strdup(argv3.c_str());

  // Here we are giving 3 arguments: directory location, feature type, output file location.
  // check for sufficient arguments
  if(argc < 4) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "\\");
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);



      std::string str(buffer);

      cv::Mat img =  cv::imread(str);
      cout << "size: " << img.size() << endl;
      vector<float> img_feature;
      if (!strcmp(argv[2], "base_match")) { // baseline_matching
          img_feature = baseline_matching(img);
      }
      else if (!strcmp(argv[2], "hist_match")) { // histogram_matching
          img_feature = histogram_matching(img);
      }
      else if (!strcmp(argv[2], "mult_hist_match")) { // multi_histograms_matching
          img_feature = mult_Hist(img);
      }
      else if (!strcmp(argv[2], "text_col")) {     // Texture and color 
          img_feature = texture_Color(img);
      }
      else if (!strcmp(argv[2], "tc_mid")) { // Custom Design: texture and color on middle part of the image
          cv::Mat mid = middle(img);
          img_feature = texture_Color(mid);
      }
      else if (!strcmp(argv[2], "mult_hist_tc")) { // Extension1: multi_histograms of Gabor texture and color
          img_feature = multi_gabor_texture_color(img);
         
      }
      else if (!strcmp(argv[2], "mult_hist_tc_mid")) { //Extension2: Gabor texture and color on middle part
          cv::Mat mid = middle(img);
          img_feature = gabor_texture_color(mid);
      }
      else {
          cout << "No such feature type." << endl;
          exit(-1);
      }
      append_image_data_csv(argv[3], buffer, img_feature);
    }
  }
  
  printf("Terminating\n");

  return(0);
}

#endif