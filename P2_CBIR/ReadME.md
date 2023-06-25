## Content Based Image Retrieval
The purpose of this project is to continue the process of learning how to manipulate and analyze images at a pixel level. In addition, this is the project where we will be doing matching, or pattern recognition.

The overall task for this project is, given a database of images and a target image, find images in the data with similar content. For this project we will not be using neural networks or object recognition methods. Instead, we will focus on more generic characteristics of the images such as color, texture, and their spatial layout. This will give you practice with working with different color spaces, histograms, spatial features, and texture features.

# Setup 
You can find a large database of images from anywhere on the internet e.g., Kaggle. Try images which has a variety of sizes, ages, and quality. You may want to set aside a subset of the images (e.g. 10-20 of them) for basic testing as you develop your code. Pick a set that have some similar images and some very different images.

# Task
For this task the inputs to the system will be a target image T, an image database B, a method of computing features for an image F, a distance metric for comparing the image features from two images D(Ft,Fi), and the desired number of output images N. The output will be an ordered list of the N images in B that are most similar to T, according to F and D, in ascending order of distance. Remember, the smaller a distance, the more similar two images will be.

The process can be generally described as the following four steps.

* **Compute the features Ft on the target image T**
* **Compute the features {Fi} on all of the images in B**
* **Compute the distance of T from all of the images in B using the distance metric D(Ft,Fi)**
* **Sort the images in B according to their distance from T and return the best N matches**

The entire process can be implemented as a command line program that takes in a target filename for T, a directory of images as the database B, the feature type, the matching method, and the number of images N to return. The program should print out the filenames of the top N matching images. If you're feeling fancy, you could create an OpenCV GUI that lets the user open an image and then displays the target and the top N matches.

# Implementation
Match a target image feature with features from a database of images. There are 2 different ways from which we can do this process :
* **Monolithic**
1. Read the target image
2. Compute feature vector from the target image F_target
3. For each image i in the database :
	Read image i
	Compute feature vector from image i
	Compute the distance from the image i feature vector to F_target
	Store distance from image i feature vector to F_target
	Store distance and image i ID
4. Sort the distances
5. Print out the top N image IDs

* **Split Process into two parts**
# Phase-1
1. Open a database file
2. For each image i in the database:
	Read image i
	Compute feature vector from image i
	Store feature vector from image in the database
3. Close the database file

# Phase-2
1. Read the target image
2. Compute feature vector for target image F_target
3. Read the database file of the image feature vectors
4. For each entry i in the databse:
	Compute a distance between F_target and the feature vector i in the database
	Store the distances and entry ID
5. Sort the distances
6. Print Out the top N entry IDs.

## Distance Metrics
Distance metrics are used to quantify the dissimilarity or similarity between two feature vectors. Let's discuss each distance metric and their motives:

1. sumOfSquareDifference: This distance metric calculates the sum of square differences between corresponding elements of two feature vectors. It measures the overall difference between the vectors. The smaller the value, the more similar the vectors are. This metric is useful when comparing continuous or numerical features.

2. histogramIntersection: This distance metric computes the histogram intersection between two feature vectors. It calculates the sum of the minimum values between corresponding elements of the vectors. This metric measures the overlapping or intersection between the histograms represented by the vectors. A higher value indicates a higher degree of similarity between the vectors. This metric is commonly used in image processing and computer vision tasks to compare histograms or distribution-based features.

In various applications, such as image recognition, content-based retrieval, clustering, and classification, it is essential to measure the distance or similarity between feature representations of objects or patterns. These metrics enable us to compare and rank feature vectors based on their similarity or dissimilarity, helping in various tasks like matching, classification, and retrieval.


## Task - 1 : Baseline Matching
Baseline matching refers to a simple and straightforward approach to image matching or similarity comparison. In the context of the given task, the baseline matching method involves using a specific region of the image, specifically the 9x9 square in the middle, as a feature vector. This feature vector represents the essential characteristics or attributes of the image.

The baseline matching process can be summarized as follows:

1. Target Image: The first step is to select a target image for comparison. 
2. Feature Extraction: The 9x9 square region in the middle of the target image is extracted, creating a smaller image patch. This patch serves as the feature vector for the target image.
3. Feature Comparison: The program then loops over a directory of images containing other images. For each image in the directory, the same feature 	extraction process is applied, resulting in a feature vector for each image.
4. Distance Metric: The similarity or dissimilarity between the target image's feature vector and each image's feature vector is determined using a distance metric. In this case, the sum-of-squared-difference (SSD) distance metric is used.
5. Matching and Ranking: The computed distances are stored in an array or vector, allowing for the sorting of matches based on the similarity scores. The top N matches, where N is predefined, are identified. 

## Task - 2 : Histogram Matching
Histogram matching is a technique used to compare images based on their color distributions. In this task, we will use a single normalized color histogram as the feature vector for each image. The histogram captures the frequency of color values in the image.

To perform histogram matching, we follow these steps:

1. Target Image: We select a target image for comparison. 
2. Color Histogram: We calculate a color histogram for the target image. The histogram represents the distribution of color values in the image. For this task, we use a normalized color histogram that considers the red-green (rg) chromaticity values. We divide the range of color values into bins and count the number of pixels that fall into each bin. 
3. Feature Comparison: Next, we compute the color histograms for all other images in the dataset using the same bin configuration.
4. Distance Metric: To measure the similarity between the target image's histogram and each image's histogram, we use the histogram intersection as the distance metric. The histogram intersection calculates the overlapping area between two histograms, indicating their similarity. A higher intersection value indicates a closer match.
5. Matching and Ranking: We store the calculated distances in an array or vector and sort them in ascending order. The top N matches with the smallest distances are considered the closest matches to the target image.

## Task - 3 : Multi-Histogram Matching
In this task, we will use two or more color histograms to represent different spatial parts of an image. Each histogram will capture the color distribution of a specific region in the image. These regions can be overlapping or disjoint.

To compare the similarity between images, we will use a distance metric called histogram intersection. This metric calculates the similarity between corresponding bins in two histograms. The higher the intersection value, the more similar the color distributions are.

We will also compute the histograms for other images in the dataset. Then, we will compare the histograms using histogram intersection and rank the images based on their similarity scores.

The top N matches will be the images with the highest similarity scores. 
These matches were obtained by using two RGB histograms: one representing the top half and the other representing the bottom half of the image. Each histogram has 8 bins for each RGB channel. 
The distances between the histograms were combined using weighted averaging to determine the final similarity scores.

## Task - 4 : Texture and Color
In this task, we are required to use both color and texture information to create a feature vector for image matching. The feature vector will consist of a whole image color histogram and a whole image texture histogram. The goal is to find the top N matching images for a given target image using a distance metric that weights the color and texture histograms equally.

For the texture histogram, we can use the Sobel magnitude image. The Sobel filters can be applied to the image to calculate the gradients, and the magnitudes of these gradients can be used to create a histogram. Alternatively, we can also consider using additional texture analysis methods such as Gabor filter responses to create texture histograms.

1. Once we have calculated both the color and texture histograms for the target image, we can compare them with the histograms of other images in our dataset using a distance metric. 
2. The distance metric should give equal weightage to both the color and texture histograms. This comparison will help identify the top N images that are most similar to the target image in terms of color and texture.

3. To evaluate the results, we should compare the top N matches obtained in this task (using color and texture histograms) with the top N matches from tasks 2 and 3. This will allow us to observe and understand how the inclusion of texture information affects the matching results in comparison to using only color histograms or a combination of color and spatial information.

## Task - 5 : Gabor Texture, Gabor Texture and color, Multi-Gabor texture and color
In this task, we perform texture and color feature extraction from an input image using Gabor filters and color histograms. They provide different levels of feature granularity, ranging from the entire image to smaller grid regions. These feature vectors can be used for tasks such as image matching, classification, or retrieval.

1. gaborTexture(Mat &image):
        The function takes an input image and converts it to grayscale.
        It then applies a set of Gabor filters to the grayscale image.
        For each filtered image, it calculates the mean and standard deviation.
        The mean and standard deviation values are stored in a 1D vector, forming the texture feature vector.
        The feature vector is L2 normalized and returned.

2. gaborTextureAndColor(Mat &image):
        The function takes an input image.
        It calculates the texture feature vector using the gaborTexture function.
        It also calculates the color feature vector using the histogram function.
        The texture and color feature vectors are concatenated into a single feature vector.
        The feature vector is returned.

3. multiGaborTextureAndColor(Mat &image):
        The function takes an input image.
        It splits the image into a 2x2 grid.
        For each grid cell, it extracts the corresponding region of interest (ROI).
        It calculates the feature vector for each ROI using the gaborTextureAndColor function.
        The feature vectors for all the grid cells are concatenated into a single feature vector.
        The feature vector is returned.
        
# Comparision between Distance Metrics
The choice between the "Sum of Square Difference" (SSD) and "Histogram Intersection" (HI) distance metrics depends on the specific requirements and characteristics of your image retrieval or matching task. Let's discuss each metric:

    Sum of Square Difference (SSD):
        The SSD metric computes the sum of the squared differences between corresponding elements of the feature vectors.
        It measures the pixel-level similarity between images by comparing their feature values.
        SSD is often used in cases where the feature vectors represent pixel intensities or local descriptors.
        This metric emphasizes the differences between images and is sensitive to changes in pixel values.
        It is useful for tasks such as template matching or comparing images based on pixel-level similarities.
        However, SSD may not be effective when dealing with image transformations, lighting variations, or global structural differences.

    Histogram Intersection (HI):
        The HI metric computes the intersection value between corresponding elements of the feature vectors.
        It measures the overlap between histograms or probability distributions.
        HI is commonly used when the feature vectors represent image histograms or visual bag-of-words representations.
        This metric is robust to global changes in image intensity, such as lighting variations, and focuses on the distribution of features.
        HI is particularly effective for tasks like content-based image retrieval, where the overall distribution and frequency of features are important.
        However, HI may not capture fine-grained pixel-level differences between images.        
























