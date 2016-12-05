#include "EndToEndWrapper.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

int main(int argc, char* argv[])
{
	EndToEndWrapper e2e = EndToEndWrapper();
	char* nothing[1];
	e2e.run(0,nothing);
	Mat output = imread("recognition.JPG");
	imshow("recognition", output);
	waitKey(0);
}