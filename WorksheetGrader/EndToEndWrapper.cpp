#include "EndToEndWrapper.h"

EndToEndWrapper::EndToEndWrapper()
{
}

EndToEndWrapper::~EndToEndWrapper()
{
}

vector<string> EndToEndWrapper::run(String filename) {
	struct EndToEndFuncs
	{
		static size_t minimum(size_t x, size_t y, size_t z)
		{
			return x < y ? min(x, z) : min(y, z);
		}
		static size_t edit_distance(const string& A, const string& B)
		{
			size_t NA = A.size();
			size_t NB = B.size();

			vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

			for (size_t a = 0; a <= NA; ++a)
				M[a][0] = a;

			for (size_t b = 0; b <= NB; ++b)
				M[0][b] = b;

			for (size_t a = 1; a <= NA; ++a)
				for (size_t b = 1; b <= NB; ++b)
				{
					size_t x = M[a - 1][b] + 1;
					size_t y = M[a][b - 1] + 1;
					size_t z = M[a - 1][b - 1] + (A[a - 1] == B[b - 1] ? 0 : 1);
					M[a][b] = minimum(x, y, z);
				}

			return M[A.size()][B.size()];
		}
		static bool isRepetitive(const string& s)
		{
			int count = 0;
			for (int i = 0; i<(int)s.size(); i++)
			{
				if ((s[i] == 'i') ||
					(s[i] == 'l') ||
					(s[i] == 'I'))
					count++;
			}
			if (count >((int)s.size() + 1) / 2)
			{
				return true;
			}
			return false;
		}
		static void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
		{
			for (int r = 0; r<(int)group.size(); r++)
			{
				ERStat er = regions[group[r][0]][group[r][1]];
				if (er.parent != NULL) // deprecate the root region
				{
					int newMaskVal = 255;
					int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
					floodFill(channels[group[r][0]], segmentation, Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
						Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
				}
			}
		}
		static bool   sort_by_lenght(const string &a, const string &b) { return (a.size()>b.size()); }
		//TODO:
		// - get words out
		// - slim down
		static vector<string> run_main(int argc, const char* argv[])
		{
			//cout << endl << argv[0] << endl << endl;
			cout << "A demo program of End-to-end Scene Text Detection and Recognition: " << endl;
			cout << "Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:" << endl;
			cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;

			Mat image;

			if (argc>1)
			image = imread(argv[1]);
			else
			{
    			cout << "    Usage: " << argv[0] << " <input_image> [<gt_word1> ... <gt_wordN>]" << endl;
    			return vector<string>();
			}

			cout << "IMG_W=" << image.cols << endl;
			cout << "IMG_H=" << image.rows << endl;

			/*Text Detection*/

			// Extract channels to be processed individually
			vector<Mat> channels;

			Mat grey;
			cvtColor(image, grey, COLOR_RGB2GRAY);

			// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
			channels.push_back(grey);
			channels.push_back(255 - grey);

			double t_d = (double)getTickCount();
			// Create ERFilter objects with the 1st and 2nd stage default classifiers
			Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 8, 0.00015f, 0.13f, 0.2f, true, 0.1f);
			Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);

			vector<vector<ERStat> > regions(channels.size());
			// Apply the default cascade classifier to each independent channel (could be done in parallel)
			for (int c = 0; c<(int)channels.size(); c++)
			{
				er_filter1->run(channels[c], regions[c]);
				er_filter2->run(channels[c], regions[c]);
			}
			cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d) * 1000 / getTickFrequency() << endl;

			Mat out_img_decomposition = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
			vector<Vec2i> tmp_group;
			for (int i = 0; i<(int)regions.size(); i++)
			{
				for (int j = 0; j<(int)regions[i].size(); j++)
				{
					tmp_group.push_back(Vec2i(i, j));
				}
				Mat tmp = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
				er_draw(channels, regions, tmp_group, tmp);
				if (i > 0)
					tmp = tmp / 2;
				out_img_decomposition = out_img_decomposition | tmp;
				tmp_group.clear();
			}

			double t_g = (double)getTickCount();
			// Detect character groups
			vector< vector<Vec2i> > nm_region_groups;
			vector<Rect> nm_boxes;
			erGrouping(image, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_ANY, "trained_classifier_erGrouping.xml");
			cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g) * 1000 / getTickFrequency() << endl;



			/*Text Recognition (OCR)*/

			double t_r = (double)getTickCount();
			Ptr<OCRTesseract> ocr = OCRTesseract::create();
			cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r) * 1000 / getTickFrequency() << endl;
			string output;

			Mat out_img;
			Mat out_img_detection;
			Mat out_img_segmentation = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
			image.copyTo(out_img);
			image.copyTo(out_img_detection);
			float scale_img = 600.f / image.rows;
			float scale_font = (float)(2 - scale_img) / 1.4f;
			vector<string> words_detection;

			t_r = (double)getTickCount();
			for (int i = 0; i<(int)nm_boxes.size(); i++)
			{

				rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(0, 255, 255), 3);

				Mat group_img = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
				er_draw(channels, regions, nm_region_groups[i], group_img);
				Mat group_segmentation;
				group_img.copyTo(group_segmentation);
				//image(nm_boxes[i]).copyTo(group_img);
				group_img(nm_boxes[i]).copyTo(group_img);
				copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

				vector<Rect>   boxes;
				vector<string> words;
				vector<float>  confidences;
				ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

				output.erase(remove(output.begin(), output.end(), '\n'), output.end());
				//cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
				if (output.size() < 3)
					continue;

				for (int j = 0; j<(int)boxes.size(); j++)
				{
					boxes[j].x += nm_boxes[i].x - 15;
					boxes[j].y += nm_boxes[i].y - 15;

					//cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
					if ((words[j].size() < 2) || (confidences[j] < 51) ||
						((words[j].size() == 2) && (words[j][0] == words[j][1])) ||
						((words[j].size()< 4) && (confidences[j] < 60)) ||
						isRepetitive(words[j]))
						continue;
					words_detection.push_back(words[j]);

					rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255, 0, 255), 3);
					Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
					rectangle(out_img, boxes[j].tl() - Point(3, word_size.height + 3), boxes[j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
					putText(out_img, words[j], boxes[j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255, 255, 255), (int)(3 * scale_font));
					out_img_segmentation = out_img_segmentation | group_segmentation;
				}

			}

			cout << "TIME_OCR = " << ((double)getTickCount() - t_r) * 1000 / getTickFrequency() << endl;

/* CHANGED CODE HERE **********************************************************/
/* CHANGES: commented out, unused by our implementation
			/* Recognition evaluation with (approximate) hungarian matching and edit distances */
			/*
			if (argc>2)
			{
				int num_gt_characters = 0;
				vector<string> words_gt;
				for (int i = 2; i<argc; i++)
				{
					string s = string(argv[i]);
					if (s.size() > 0)
					{
						words_gt.push_back(string(argv[i]));
						//cout << " GT word " << words_gt[words_gt.size()-1] << endl;
						num_gt_characters += (int)(words_gt[words_gt.size() - 1].size());
					}
				}

				if (words_detection.empty())
				{
					//cout << endl << "number of characters in gt = " << num_gt_characters << endl;
					cout << "TOTAL_EDIT_DISTANCE = " << num_gt_characters << endl;
					cout << "EDIT_DISTANCE_RATIO = 1" << endl;
				}
				else
				{

					sort(words_gt.begin(), words_gt.end(), sort_by_lenght);

					int max_dist = 0;
					vector< vector<int> > assignment_mat;
					for (int i = 0; i<(int)words_gt.size(); i++)
					{
						vector<int> assignment_row(words_detection.size(), 0);
						assignment_mat.push_back(assignment_row);
						for (int j = 0; j<(int)words_detection.size(); j++)
						{
							assignment_mat[i][j] = (int)(edit_distance(words_gt[i], words_detection[j]));
							max_dist = max(max_dist, assignment_mat[i][j]);
						}
					}

					vector<int> words_detection_matched;

					int total_edit_distance = 0;
					int tp = 0, fp = 0, fn = 0;
					for (int search_dist = 0; search_dist <= max_dist; search_dist++)
					{
						for (int i = 0; i<(int)assignment_mat.size(); i++)
						{
							int min_dist_idx = (int)distance(assignment_mat[i].begin(),
								min_element(assignment_mat[i].begin(), assignment_mat[i].end()));
							if (assignment_mat[i][min_dist_idx] == search_dist)
							{
								//cout << " GT word \"" << words_gt[i] << "\" best match \"" << words_detection[min_dist_idx] << "\" with dist " << assignment_mat[i][min_dist_idx] << endl;
								if (search_dist == 0)
									tp++;
								else { fp++; fn++; }

								total_edit_distance += assignment_mat[i][min_dist_idx];
								words_detection_matched.push_back(min_dist_idx);
								words_gt.erase(words_gt.begin() + i);
								assignment_mat.erase(assignment_mat.begin() + i);
								for (int j = 0; j<(int)assignment_mat.size(); j++)
								{
									assignment_mat[j][min_dist_idx] = INT_MAX;
								}
								i--;
							}
						}
					}

					for (int j = 0; j<(int)words_gt.size(); j++)
					{
						//cout << " GT word \"" << words_gt[j] << "\" no match found" << endl;
						fn++;
						total_edit_distance += (int)words_gt[j].size();
					}
					for (int j = 0; j<(int)words_detection.size(); j++)
					{
						if (find(words_detection_matched.begin(), words_detection_matched.end(), j) == words_detection_matched.end())
						{
							//cout << " Detection word \"" << words_detection[j] << "\" no match found" << endl;
							fp++;
							total_edit_distance += (int)words_detection[j].size();
						}
					}


					//cout << endl << "number of characters in gt = " << num_gt_characters << endl;
					cout << "TOTAL_EDIT_DISTANCE = " << total_edit_distance << endl;
					cout << "EDIT_DISTANCE_RATIO = " << (float)total_edit_distance / num_gt_characters << endl;
					cout << "TP = " << tp << endl;
					cout << "FP = " << fp << endl;
					cout << "FN = " << fn << endl;
				}
			}

            */
/* END OF CHANGED CODE ********************************************************/


			//resize(out_img_detection,out_img_detection,Size(image.cols*scale_img,image.rows*scale_img));
			//imshow("detection", out_img_detection);
			//imwrite("detection.jpg", out_img_detection);
			//resize(out_img,out_img,Size(image.cols*scale_img,image.rows*scale_img));
			//namedWindow("recognition", WINDOW_NORMAL);
			imwrite("recognition.JPG", out_img);
			//imshow("recognition", out_img);
			//waitKey(0);
			//imwrite("recognition.jpg", out_img);
			//imwrite("segmentation.jpg", out_img_segmentation);
			//imwrite("decomposition.jpg", out_img_decomposition);
/* CHANGED CODE HERE **********************************************************/
			return words_detection;
/* END OF CHANGED CODE ********************************************************/
		}
	};
	const char *arg1 = filename.c_str();
	const char* argv[2] = { String("function call").c_str(), arg1 };
	return EndToEndFuncs::run_main(2, argv);
}

vector<string> EndToEndWrapper::runOCR(String filename) {
	Mat image = imread(filename);
	cout << "Parsing " << filename << "...";
	vector<vector<Point>> squares;
	findSquares(image, squares);
	drawSquares(image, squares);
	namedWindow("rectangles1.JPG", WINDOW_NORMAL);
	imshow("rectangles1.JPG", image);
	imwrite("rectangles.JPG", image);
	waitKey(0);
	findSquares(image, squares);
	drawSquares(image, squares);
	namedWindow("rectangles2.JPG", WINDOW_NORMAL);
	imshow("rectangles2.JPG", image);
	imwrite("rectangles.JPG", image);
	waitKey(0);
	findSquares(image, squares);
	drawSquares(image, squares);
	namedWindow("rectangles3.JPG", WINDOW_NORMAL);
	imshow("rectangles3.JPG", image);
	imwrite("rectangles.JPG", image);
	waitKey(0);
	findSquares(image, squares);
	drawSquares(image, squares);
	namedWindow("rectangles4.JPG", WINDOW_NORMAL);
	imshow("rectangles4.JPG", image);
	imwrite("rectangles.JPG", image);
	waitKey(0);
	findSquares(image, squares);
	drawSquares(image, squares);
	namedWindow("rectangles5.JPG", WINDOW_NORMAL);
	imshow("rectangles5.JPG", image);
	imwrite("rectangles.JPG", image);
	waitKey(0);
	Ptr<OCRTesseract> ocr = OCRTesseract::create();
	string output;
	Mat out_img;
	Mat out_img_detection;
	Mat out_img_segmentation = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
	image.copyTo(out_img);
	image.copyTo(out_img_detection);
	float scale_img = 600.f / image.rows;
	float scale_font = (float)(2 - scale_img) / 1.4f;
	vector<string> words_detection;
	vector<Rect> answers;
	vector< vector<Vec2i> > answerlocations;
	
	for (int i = 0; i < (int)answers.size(); i++)
	{
		rectangle(out_img_detection, answers[i].tl(), answers[i].br(), Scalar(0, 255, 255), 3);

		Mat group_img = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
		Mat group_segmentation;
		group_img.copyTo(group_segmentation);
		//image(answers[i]).copyTo(group_img);
		group_img(answers[i]).copyTo(group_img);
		copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

		vector<Rect>   boxes;
		vector<string> words;
		vector<float>  confidences;
		ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

		output.erase(remove(output.begin(), output.end(), '\n'), output.end());
		//cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;


		for (int j = 0; j < (int)boxes.size(); j++)
		{
			boxes[j].x += answers[i].x - 15;
			boxes[j].y += answers[i].y - 15;

			//cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
			if ((words[j].size() < 2) || (confidences[j] < 51) ||
				((words[j].size() == 2) && (words[j][0] == words[j][1])) ||
				((words[j].size() < 4) && (confidences[j] < 60)))
				continue;
			words_detection.push_back(words[j]);
			rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255, 0, 255), 3);
			Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
			rectangle(out_img, boxes[j].tl() - Point(3, word_size.height + 3), boxes[j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
			putText(out_img, words[j], boxes[j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255, 255, 255), (int)(3 * scale_font));
			out_img_segmentation = out_img_segmentation | group_segmentation;
		}
	}
	return words_detection;
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double EndToEndWrapper::angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void EndToEndWrapper::findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 0, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}

// the function draws all the squares in the image
 void EndToEndWrapper::drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
}
