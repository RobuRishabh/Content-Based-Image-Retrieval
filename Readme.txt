1. I performed this project in windows-11 using Visual Studio IDE.


2. Instructions for running the executables: There are two executable file-
    (a).'readfiles.cpp' is used to calculate the feature vectors of each image in the database and write the info to a csv file. 
    (b).'extract_feature.cpp' is used to calculate the feature vector for the target image and then get the top N matches.

    - 'readfiles.cpp'
        - It will takes 3-input.
		* Input1: Directory location written as argv1
 		* Input2: feature type written as argv2
 		* Input3: Path where we want our output csv file to get generated as output.csv written as argv3
        - the csv file will get saved in location provided as "output.csv"

    - 'extract_features.cpp'
	  - Here I have taken 5 inputs in argv[].
 		* Input1: Path to the target image written as argv1
 		* Input2: feature set written as argv2
 		* Input3: Path where we want our output csv file to get generated as output.csv written as argv3
 		* Input4: Distance metrics written as argv4
 		* Input5: Our top N matches written as argv5

    (c). Step0 - Always run the "readfiles.cpp" first then "extract_features.cpp", for every Task.
	   Step1 - Go to "readfiles.cpp" and check the pre-processor directive i.e., just before the main() there is a statement
			written "#if 0" then make it "#if 1", keeping pre-processor directive of "extract_feature.cpp" "#if 0". So, this will
			enable the "readfiles.cpp" file and disable the "extract_feature.cpp" file. Now "readfiles.cpp" will run into main()
         Step2 - Now, as you are finished calculating feature vectors, now make the pre-processor directive of "readfiles.cpp" 
			"#if 0" to stop the "readfile.cpp"
	   Step3 - then make the pre-processor directive just above the main() of "extract_feature.cpp" "#if 1". keeping pre-processor 
			directive of "readfiles.cpp" "#if 0". So, this will enable the "extract_feature.cpp" file and disable the 
			"readfiles.cpp" file. Now "extract_feature.cpp" will run into main()
	   Step4 - You will get the names of figures in output screen after step3.  
      
    (d). After executing every task successfully, please delete the "output.csv" file from the location. Otherwise, when you will run the program 
		again, it will append the previous result read files into the csv and you will not get result. It will throw an exception error
		of "Size of target image and other images are not same"



- Enter the following arguments in feature type in both "readfiles.cpp" and "extract_feature.cpp" to run the executables 
	Example:   std::string argv2 = "mult_hist_tc_mid";
    - features
        - base_match: baseline matching (task 1)
        - hist_match: histogram matching(task 2)
        - mult_hist_match: Multi-histogram Matching (task 3)
        - text_col: Texture and Color (task 4)
        - tc_mid: Custom Design (Sobel color and texture on middle part of the image) (task 5)
        - mult_hist_tc: applying multiple histograms of Gabor texture and color (extension)
        - mult_hist_tc_mid: applying Gabor texture and color on middle part of the image (extension)


- Enter the following arguments in distance metric in "extract_feature.cpp" to run the executables 
	Example:  std::string argv4 = "hist_int";
    - distance metric
        - sum_sq_dist: sum of square difference
        - hist_int: histogram intersection

- I have created distance_metric.cpp file for the calculation of distance metric.
    
