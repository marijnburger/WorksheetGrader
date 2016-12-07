#pragma once
//TODO cite:
/*
* end_to_end_recognition.cpp
*
* A demo program of End-to-end Scene Text Detection and Recognition:
* Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:
* Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
*
* Created on: Jul 31, 2014
*     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
*/

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::text;

class EndToEndWrapper
{
private:
	int thresh = 50, N = 11;
public:
	EndToEndWrapper();
	~EndToEndWrapper();

	//wraps 'end_to_end_recognition.cpp'
	vector<string> run(String filename);
	void normalizeBG(String filename);
	void thresholdImage(String filename);
	void EndToEndWrapper::findRectangles(const Mat& image, vector<vector<Point> >& rectangles);
	void EndToEndWrapper::drawRectangles(Mat& image, const vector<vector<Point> >& squares);
	vector<string> runOCR(String filename);
};