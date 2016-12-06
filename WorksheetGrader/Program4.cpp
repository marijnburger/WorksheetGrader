#include "EndToEndWrapper.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::text;

// **** Function Declarations **************************************************

// Place a grade on an assignment. 
// Score = points earned
// Possible = points possible
void assignGrade(String theFilename, float score, float possible)
{
	int percent = static_cast<int>(floor((score / possible) * 100));

	String scoreString = to_string(percent) + "%";
	Mat theImage = imread(theFilename);

	putText(theImage, scoreString, cvPoint(200,200), FONT_HERSHEY_DUPLEX, 6.0,
		CV_RGB(255,0,0), 8, CV_AA);

	imwrite("gradedPaper.JPG", theImage);

}

// **** Main *******************************************************************
int main()
{
	
	EndToEndWrapper e2e = EndToEndWrapper();
	e2e.run("test5.JPG");
	Mat output = imread("recognition.JPG");
	namedWindow("recognition", WINDOW_NORMAL);
	imshow("recognition", output);
	waitKey(0);
	

	Mat gradedPaper = imread("test5.JPG");
	imwrite("gradedPaper.JPG", gradedPaper);
	assignGrade("gradedPaper.JPG", 75, 100);

	
	Mat aGradedPaper = imread("gradedPaper.JPG");
	namedWindow("Graded Paper", WINDOW_NORMAL);
	imshow("Graded Paper", aGradedPaper);
	waitKey(0);
	
}

// **** Function Definitions ***************************************************