// Worksheet Grader
// CSS 487 Project

// Marijn Burger
// Jack Eldridge

// Uses end_to_end_recognition.cpp from OpenCV
// Sample code

// Papers will only be graded if at least 3 answers are found
// It will crash if it does not find at least 3 answers

#include "EndToEndWrapper.h"
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::text;

// **** Class Constants *******************************************************
const int NUMBER_OF_QUESTIONS = 3;

// **** Function Declarations *************************************************
void assignGrade(String theFilename, float score);
float compareAnswers(String* answers, String* solutions);

// **** Main ******************************************************************
int main()
{
	String testFilename = "InputWorksheet.JPG";
	String theSolution[NUMBER_OF_QUESTIONS] = { "HOTEL", "CHINA", "Hello" };

	Mat gradedPaper = imread(testFilename);
	Mat gradedPaper2 = imread(testFilename);

	vector<string> outputwords;

	EndToEndWrapper e2e = EndToEndWrapper();
	outputwords = e2e.runTargeted("InputWorksheet.JPG");
	Mat output = imread("input0.JPG");
	namedWindow("Answer 1", WINDOW_NORMAL);
	imshow("Answer 1", output);
	waitKey(0);
	output = imread("input1.JPG");
	namedWindow("Answer 2", WINDOW_NORMAL);
	imshow("Answer 2", output);
	waitKey(0);
	output = imread("input2.JPG");
	namedWindow("Answer 3", WINDOW_NORMAL);
	imshow("Answer 3", output);
	waitKey(0);

	// Will crash if at least 3 words were not found
	if (outputwords.size() < 3)
	{
		outputwords[0] = "UNCLEAR";
		outputwords[1] = "UNCLEAR";
		outputwords[2] = "UNCLEAR";
	}

	cout << outputwords[0] << endl;
	cout << outputwords[1] << endl;
	cout << outputwords[2] << endl;

	String someAnswers[NUMBER_OF_QUESTIONS] =
	{ outputwords[0], outputwords[1], outputwords[2] };
	//String someAnswers[NUMBER_OF_QUESTIONS] = { "CHINA", "mars", "carbon" };

	// Make all letters lowercase
	char someChar;
	String lowerWord;
	for (int i = 0; i < NUMBER_OF_QUESTIONS; i++)
	{
		lowerWord = "";
		for (int j = 0; j < someAnswers[i].length(); j++)
		{
			someChar = tolower(someAnswers[i][j]);
			lowerWord += someChar;
		}
		someAnswers[i] = lowerWord;
	}

	for (int i = 0; i < NUMBER_OF_QUESTIONS; i++)
	{
		lowerWord = "";
		for (int j = 0; j < theSolution[i].length(); j++)
		{
			someChar = tolower(theSolution[i][j]);
			lowerWord += someChar;
		}
		theSolution[i] = lowerWord;
	}

	float theScore = compareAnswers(someAnswers, theSolution);


	imwrite("gradedPaper.JPG", gradedPaper);
	assignGrade("gradedPaper.JPG", theScore);

	//Display Graded Paper
	Mat aGradedPaper = imread("gradedPaper.JPG");
	namedWindow("Graded Paper", WINDOW_NORMAL);
	imshow("Graded Paper", aGradedPaper);
	waitKey(0);

}

// **** Function Definitions **************************************************

// Place a grade on an assignment. 
// Score = points earned
// Possible = points possible
void assignGrade(String theFilename, float score)
{
	float possible = static_cast<float>(NUMBER_OF_QUESTIONS);
	int percent = static_cast<int>(floor((score / possible) * 100));

	String scoreString = to_string(percent) + "%";
	Mat theImage = imread(theFilename);

	putText(theImage, scoreString, cvPoint(200, 200), FONT_HERSHEY_DUPLEX, 6.0,
		CV_RGB(255, 0, 0), 8, CV_AA);

	imwrite("gradedPaper.JPG", theImage);
}

// Compare answer to solutions and output
// number of matches as an int
float compareAnswers(String* answers, String* solutions)
{
	float count = 0.0;
	for (int i = 0; i < NUMBER_OF_QUESTIONS; i++)
	{
		if (answers[i].compare(solutions[i]) == 0)
		{
			count = count + 1;
		}
	}
	return count;
}

