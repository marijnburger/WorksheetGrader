#include "EndToEndWrapper.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::text;

int main()
{
	EndToEndWrapper e2e = EndToEndWrapper();
	e2e.run("test5.JPG");
	Mat output = imread("recognition.JPG");
	imshow("recognition", output);
	waitKey(0);
}