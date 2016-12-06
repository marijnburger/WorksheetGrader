// #include "EndToEndWrapper.h"
// #include "opencv2/text.hpp"
// #include "opencv2/core/utility.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// #include <iostream>
// #include <string>
// #include <math.h>
//
// using namespace std;
// using namespace cv;
// using namespace cv::text;
//
// // **** Class Constants ********************************************************
// const int NUMBER_OF_QUESTIONS = 3;
//
// // **** Function Declarations **************************************************
// void assignGrade(String theFilename, float score);
//
// float compareAnswers(String* answers, String* solutions);
//
// void greenScreen(String foregroundFilename);
//
// void moreContrast(String filename);
//
// // **** Main *******************************************************************
// int main()
// {
// 	cout << "Loading image..." << endl;
// 	String testFilename = "test5.JPG";
//
//
// 	cout << "Normalizing image background..." << endl;
// 	greenScreen2(testFilename);
//
// 	//moreContrast(testFilename);
//
//
//
// 	Mat gradedPaper = imread("afterGreenScreen.JPG");
// 	//Mat gradedPaper = imread(testFilename);
// 	//Mat gradedPaper2 = imread("afterGreenScreen.JPG");
// 	Mat gradedPaper2 = imread(testFilename);
//
// 	Size kSize(7, 7);
// 	cout << "Blurring image..." << endl;
// 	GaussianBlur(gradedPaper, gradedPaper, kSize, 2.0, 2.0);
// 	//GaussianBlur(gradedPaper, gradedPaper, kSize, 2.0, 2.0);
// 	//cout << "Inverting colors..." << endl;
// 	//bitwise_not(gradedPaper, gradedPaper2);  // Color inversion
// 	//imwrite("inverseImage.JPG", gradedPaper2);
// 	cout << "Thresholding image..." << endl;
// 	Mat thr(gradedPaper.rows, gradedPaper.cols, CV_8UC1);
// 	cvtColor(gradedPaper, thr, CV_BGR2GRAY); //Convert to gray
// 	threshold(thr, thr, 150, 255, THRESH_BINARY); //Threshold the gray
// 	imwrite("thresholdedImage.JPG", thr);
// 	vector<string> outputwords;
// 	EndToEndWrapper e2e = EndToEndWrapper();
// 	cout << "Running Tesseract OCR..." << endl;
// 	outputwords = e2e.run("thresholdedImage.JPG");
// 	/*Mat output = imread("recognition.JPG");
// 	namedWindow("recognition", WINDOW_NORMAL);
// 	imshow("recognition", output);*/
// 	waitKey(0);
//
// 	//int numberOfWordsFound = outputwords.size();
//
//
// 	String someAnswers[NUMBER_OF_QUESTIONS] = { "Hotel", "China", "Hello" };
// 	String theSolution[NUMBER_OF_QUESTIONS] = { outputwords[0], outputwords[1], outputwords[2] };
//
// 	float theScore = compareAnswers(someAnswers, theSolution);
//
// //	Mat gradedPaper = imread("test5.JPG");
// 	imwrite("gradedPaper.JPG", gradedPaper);
// 	assignGrade("gradedPaper.JPG", theScore);
//
// 	Mat aGradedPaper = imread("gradedPaper.JPG");
// 	namedWindow("Graded Paper", WINDOW_NORMAL);
// 	imshow("Graded Paper", aGradedPaper);
// 	waitKey(0);
//
// }
//
// // **** Function Definitions ***************************************************
//
// // Place a grade on an assignment.
// // Score = points earned
// // Possible = points possible
// void assignGrade(String theFilename, float score)
// {
// 	float possible = static_cast<float>(NUMBER_OF_QUESTIONS);
// 	int percent = static_cast<int>(floor((score / possible) * 100));
//
// 	String scoreString = to_string(percent) + "%";
// 	Mat theImage = imread(theFilename);
//
// 	putText(theImage, scoreString, cvPoint(200, 200), FONT_HERSHEY_DUPLEX, 6.0,
// 		CV_RGB(255, 0, 0), 8, CV_AA);
//
// 	imwrite("gradedPaper.JPG", theImage);
// }
//
// // Compare answer to solutions and output
// // number of matches as an int
// float compareAnswers(String* answers, String* solutions)
// {
// 	float count = 0.0;
// 	for (int i = 0; i < NUMBER_OF_QUESTIONS; i++)
// 	{
// 		if (answers[i].compare(solutions[i]) == 0)
// 		{
// 			count = count + 1;
// 		}
// 	}
// 	return count;
// }
//
// void greenScreen(String foregroundFilename)
// {
// 	Mat imageForeground = imread(foregroundFilename);
// 	Mat imageBackground = imread("whiteSquare.JPG");
//
// 	int size = 4;
//
// 	// size is a constant - the number of buckets in each dimension
// 	int dims[] = { size, size, size };
//
// 	//3D histogram of integers initialized to zero
// 	Mat hist(3, dims, CV_32S, Scalar::all(0));
//
// 	int bucketSize = 256 / size;   // if size = 4 -> bucketSize = 64
//
// 								   // Loops to put pixels in buckets
// 	for (int r = 0; r < imageForeground.rows; r++)
// 	{
// 		for (int c = 0; c < imageForeground.cols; c++)
// 		{
// 			int blue = imageForeground.at<Vec3b>(r, c)[0];
// 			int green = imageForeground.at<Vec3b>(r, c)[1];
// 			int red = imageForeground.at<Vec3b>(r, c)[2];
//
// 			int b = blue / bucketSize;
// 			int g = green / bucketSize;
// 			int r = red / bucketSize;
//
// 			hist.at<float>(b, g, r) += 1;
// 		}
// 	}
//
// 	int mostVotes = 0;
// 	int commonBlue = 0;
// 	int commonGreen = 0;
// 	int commonRed = 0;
//
// 	// loops to count which bucket is fullest and find
// 	// corresponding most common color
// 	for (int bl = 0; bl < size; bl++)
// 	{
// 		for (int gr = 0; gr < size; gr++)
// 		{
// 			for (int rd = 0; rd < size; rd++)
// 			{
// 				if (hist.at<float>(bl, gr, rd) > mostVotes)
// 				{
// 					mostVotes = static_cast<int>(hist.at<float>(bl, gr, rd));
// 					commonBlue = bl * bucketSize + bucketSize / 2;
// 					commonGreen = gr * bucketSize + bucketSize / 2;
// 					commonRed = rd * bucketSize + bucketSize / 2;
// 				}
// 			}
// 		}
// 	}
//
// 	// loops to do green screen effect
// 	for (int row = 0; row < imageForeground.rows; row++)
// 	{
// 		for (int col = 0; col < imageForeground.cols; col++)
// 		{
// 			int blueB = imageForeground.at<Vec3b>(row, col)[0];
// 			int greenG = imageForeground.at<Vec3b>(row, col)[1];
// 			int redR = imageForeground.at<Vec3b>(row, col)[2];
//
// 			int differenceBlue = abs(blueB - commonBlue);
// 			int differenceGreen = abs(greenG - commonGreen);
// 			int differenceRed = abs(redR - commonRed);
//
// 			if (differenceBlue < bucketSize &&
// 				differenceGreen < bucketSize && differenceRed < bucketSize)
// 			{
// 				int imageBackgroundRows = imageBackground.rows;
// 				int imageBackgroundCols = imageBackground.cols;
//
// 				int blueBackground = imageBackground.at<Vec3b>
// 					(row % imageBackgroundRows, col % imageBackgroundCols)[0];
// 				int greenBackground = imageBackground.at<Vec3b>
// 					(row % imageBackgroundRows, col % imageBackgroundCols)[1];
// 				int redBackground = imageBackground.at<Vec3b>
// 					(row % imageBackgroundRows, col % imageBackgroundCols)[2];
//
// 				imageForeground.at<Vec3b>(row, col)[0] = blueBackground;
// 				imageForeground.at<Vec3b>(row, col)[1] = greenBackground;
// 				imageForeground.at<Vec3b>(row, col)[2] = redBackground;
// 			}
// 		}
// 	}
//
// 	// create output file for green-screen effect
// 	imwrite("afterGreenScreen.jpg", imageForeground);
// }
//
// void moreContrast(String filename)
// {
// 	const float contrastFactor = 1.5;
//
// 	Mat inputImage = imread(filename);
// 	Mat sharpImage = imread(filename);
//
// 	float blueTotal = 0;
// 	float greenTotal = 0;
// 	float redTotal = 0;
//
// 	float totalPixels = static_cast<float>(inputImage.cols * inputImage.rows);
//
// 	float averageBlue;
// 	float averageGreen;
// 	float averageRed;
//
// 	float blueB;
// 	float greenG;
// 	float redR;
//
// 	float blueDifference;
// 	float greenDifference;
// 	float redDifference;
//
// 	float setColorBlue;
// 	float setColorGreen;
// 	float setColorRed;
//
// 	// Loops to get average color
// 	for (int row = 0; row < inputImage.rows; row++)
// 	{
// 		for (int col = 0; col < inputImage.cols; col++)
// 		{
// 			blueB = inputImage.at<Vec3b>(row, col)[0];
// 			greenG = inputImage.at<Vec3b>(row, col)[1];
// 			redR = inputImage.at<Vec3b>(row, col)[2];
//
// 			blueTotal = blueTotal + blueB;
// 			greenTotal = greenTotal + greenG;
// 			redTotal = redTotal + redR;
// 		}
// 	}
//
// 	averageBlue = blueTotal / totalPixels;
// 	averageGreen = greenTotal / totalPixels;
// 	averageRed = redTotal / totalPixels;
//
// 	// Loop to get difference from average
// 	for (int row = 0; row < inputImage.rows; row++)
// 	{
// 		for (int col = 0; col < inputImage.cols; col++)
// 		{
// 			blueB = inputImage.at<Vec3b>(row, col)[0];
// 			greenG = inputImage.at<Vec3b>(row, col)[1];
// 			redR = inputImage.at<Vec3b>(row, col)[2];
//
// 			// the difference * the contrastFactor
// 			blueDifference = (averageBlue - blueB) * contrastFactor * -1;
// 			greenDifference = (averageGreen - greenG) * contrastFactor * -1;
// 			redDifference = (averageRed - redR) * contrastFactor * -1;
//
// 			setColorBlue = floor(averageBlue + blueDifference);
// 			setColorGreen = floor(averageGreen + greenDifference);
// 			setColorRed = floor(averageRed + redDifference);
//
// 			if (setColorBlue > 255)
// 			{
// 				setColorBlue = 255;
// 			}
// 			if (setColorGreen > 255)
// 			{
// 				setColorGreen = 255;
// 			}
// 			if (setColorRed > 255)
// 			{
// 				setColorRed = 255;
// 			}
//
//
// 			if (setColorBlue < 0)
// 			{
// 				setColorBlue = 0;
// 			}
// 			if (setColorGreen < 0)
// 			{
// 				setColorGreen = 0;
// 			}
// 			if (setColorRed < 0)
// 			{
// 				setColorRed = 0;
// 			}
//
// 			sharpImage.at<Vec3b>(row, col)[0] = static_cast<int>(setColorBlue);
// 			sharpImage.at<Vec3b>(row, col)[1] = static_cast<int>(setColorGreen);
// 			sharpImage.at<Vec3b>(row, col)[2] = static_cast<int>(setColorRed);
// 		}
// 	}
//
// 	// create output file
// 	imwrite("moreContrast.jpg", sharpImage);
// }
