#include "EndToEndWrapper.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

// **** Function Declarations **************************************************


// **** Main *******************************************************************
int main()
{
	EndToEndWrapper e2e = EndToEndWrapper();
	e2e.run("test5.JPG");
	Mat output = imread("recognition.JPG");
	namedWindow("recognition", WINDOW_NORMAL);
	imshow("recognition", output);
	waitKey(0);
}

// **** Function Definitions ***************************************************