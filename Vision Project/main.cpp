#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <atlstr.h>  

using namespace std;
using namespace cv;
using namespace cv::ml;

//computer에 깔린 open cv 경로 대로 고쳐줘야함
String face_cascade = "C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
String eye_cascade = "C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml";
String mouth_cascade = "C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_mouth.xml";
CascadeClassifier face;
CascadeClassifier eye;
CascadeClassifier mouth;


Mat facedetect(String imgname);

bool R1(int R, int G, int B);
bool R2(float Y, float Cr, float Cb);
bool R3(float H, float S, float V);
Mat GetSkin(Mat face);
Mat GetAvgColor(Mat skin);
Mat flatSkin(Mat skin, unsigned int num);
void train(SVM* svm, int num);

int main()
{
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(3);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));

	//Training Four Class
	train(svm, 10);
	Mat avgColor, face, skin;
	
	for (int i = 1; i < 5; i++) {
		String imagename = "Dataset\\Fall\\" + to_string(i) + ".jpg";
		face = facedetect(imagename);
		skin = GetSkin(face);
		avgColor = GetAvgColor(skin);
		float res = svm->predict(avgColor);
		cout << res << endl;
	}
	for (int i = 1; i < 5; i++) {
		String imagename = "Dataset\\Spring\\" + to_string(i) + ".jpg";
		face = facedetect(imagename);
		skin = GetSkin(face);
		avgColor = GetAvgColor(skin);
		float res = svm->predict(avgColor);
		cout << res << endl;
	}
	for (int i = 1; i < 5; i++) {
		String imagename = "Dataset\\Summer\\" + to_string(i) + ".jpg";
		face = facedetect(imagename);
		skin = GetSkin(face);
		avgColor = GetAvgColor(skin);
		float res = svm->predict(avgColor);
		cout << res << endl;
	}
	for (int i = 1; i < 5; i++) {
		String imagename = "Dataset\\Winter\\" + to_string(i) + ".jpg";
		face = facedetect(imagename);
		skin = GetSkin(face);
		avgColor = GetAvgColor(skin);
		float res = svm->predict(avgColor);
		cout << res << endl;
	}
	
	return 0;
}
void train(SVM* svm, int num) {
	float labels0[4] = { 1.0, -1.0, -1.0, -1.0 };
	float labels1[4] = { -1.0, 1.0, -1.0, -1.0 };
	float labels2[4] = { -1.0, -1.0, 1.0, -1.0 };
	float labels3[4] = { -1.0, -1.0, -1.0, 1.0 };
	int labels[4] = { 1,2,3,4 };
	Mat res(0, 0, CV_32FC3);
	Mat labelsMat;
	String season = "";

	season = "Spring";
	for (int i = 1; i < num + 1; i++)
	{
		String imagename = "Dataset\\" + season + "\\" + to_string(i) + ".jpg";
		Mat face = facedetect(imagename);
		Mat skin = GetSkin(face);
		Mat avgColor = GetAvgColor(skin);
		res.push_back(avgColor);
		//Mat flat = flatSkin(skin, 5000);
		//res.push_back(flat);
		Mat label(1, 1, CV_32SC1, &labels[0]);
		labelsMat.push_back(label);
	}
	season = "Summer";
	for (int i = 1; i < num + 1; i++)
	{
		String imagename = "Dataset\\" + season + "\\" + to_string(i) + ".jpg";
		Mat face = facedetect(imagename);
		Mat skin = GetSkin(face);
		Mat avgColor = GetAvgColor(skin);
		res.push_back(avgColor);
		//Mat flat = flatSkin(skin, 5000);
		//res.push_back(flat);
		Mat label(1, 1, CV_32SC1, &labels[1]);
		labelsMat.push_back(label);
	}
	season = "Fall";
	for (int i = 1; i < num + 1; i++)
	{
		String imagename = "Dataset\\" + season + "\\" + to_string(i) + ".jpg";
		Mat face = facedetect(imagename);
		Mat skin = GetSkin(face);
		Mat avgColor = GetAvgColor(skin);
		res.push_back(avgColor);
		//Mat flat = flatSkin(skin, 5000);
		//res.push_back(flat); 
		Mat label(1, 1, CV_32SC1, &labels[2]);
		labelsMat.push_back(label);
	}
	season = "Winter";
	for (int i = 1; i < num + 1; i++)
	{
		String imagename = "Dataset\\" + season + "\\" + to_string(i) + ".jpg";
		Mat face = facedetect(imagename);
		Mat skin = GetSkin(face);
		Mat avgColor = GetAvgColor(skin);
		res.push_back(avgColor);
		//Mat flat = flatSkin(skin, 5000);
		//res.push_back(flat);
		Mat label(1, 1, CV_32SC1, &labels[3]);
		labelsMat.push_back(label);
	}
	
	Ptr<TrainData> td = TrainData::create(res, ROW_SAMPLE, labelsMat);
	svm->train(td);
}

Mat facedetect(String imgname)
{
	Mat img = imread(imgname); // 본래의 색으로 출력한다 
	
	if (img.data == NULL){
		cout << "이미지 열기 실패" << endl;
	}
	if (!face.load(face_cascade)){
		cout << "Cascade 파일 열기 실패" << endl;
	}
	if (!eye.load(eye_cascade)) {
		cout << "Cascade 파일 열기 실패" << endl;
	}
	
	if (!mouth.load(mouth_cascade)) {
		cout << "Cascade 파일 열기 실패" << endl;
	}
	//Face detect
	Mat gray;

	//gray scale로 변환
	cvtColor(img, gray, CV_BGR2GRAY);

	//histogram 얻기
	equalizeHist(gray, gray);

	//이미지 표시용 변수
	vector<Rect> faces;
	vector<Rect> eyes;
	vector<Rect> mouths;

	//얼굴의 위치와 영역 탐색
	face.detectMultiScale(gray, faces, 1.1, 5, 0);

	for (int k = 0; k<faces.size(); k++) {
		Mat face = gray(faces[k]);
		//해당 얼굴에서 눈 영역 검출
		eye.detectMultiScale(face, eyes, 1.3, 10, 0, Size(20, 20));
	
		//해당 얼굴에서 입 영역 검출
		mouth.detectMultiScale(face, mouths, 2, 20, 0, Size(10, 10));
	}

	//얼굴 영역 제외 삭제
	Mat dst = img.clone();
	Vec3b cblack = Vec3b::all(0);
	
	bool black = true;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			for (int k = 0; k<faces.size(); k++) {
				Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
				Point tr(faces[k].x, faces[k].y);
				if (j < lb.x && tr.x < j && i < lb.y && tr.y < i)
					black = false;

				if (!black)
				{
					Mat face = gray(faces[k]);
					//해당 얼굴에서 눈 영역 검출 및 삭제
					//eye.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
					for (int m = 0; m < eyes.size(); m++) {
						Point lb(faces[k].x + eyes[m].x + eyes[m].width, faces[k].y + eyes[m].y + eyes[m].height);
						Point tr(faces[k].x + eyes[m].x, faces[k].y + eyes[m].y);
						if (j < lb.x && tr.x < j && i < lb.y && tr.y < i)
							black = true;
					}
					//해당 얼굴에서 눈 영역 검출 및 삭제
					//mouth.detectMultiScale(face, mouths, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
					for (int n = 0; n < mouths.size(); n++) {
						Point lb(faces[k].x + mouths[n].x + mouths[n].width, faces[k].y + mouths[n].y + mouths[n].height);
						Point tr(faces[k].x + mouths[n].x, faces[k].y + mouths[n].y);
						if (j < lb.x && tr.x < j && i < lb.y && tr.y < i)
							black = true;
					}
				}
			}

			if (black)
				dst.ptr<Vec3b>(i)[j] = cblack;

			black = true;
		}
	}
	
	

	for (int k = 0; k<faces.size(); k++) {
		Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
		Point tr(faces[k].x, faces[k].y);

			Mat face = gray(faces[k]);
			//해당 얼굴에서 눈 영역 검출 및 삭제

			for (int m = 0; m < eyes.size(); m++) {
				Point lb(faces[k].x + eyes[m].x + eyes[m].width, faces[k].y + eyes[m].y + eyes[m].height);
				Point tr(faces[k].x + eyes[m].x, faces[k].y + eyes[m].y);
				rectangle(img, lb, tr, Scalar(255, 255, 0), 3, 4, 0);
			}
			//해당 얼굴에서 눈 영역 검출 및 삭제
			for (int n = 0; n < mouths.size(); n++) {
				Point lb(faces[k].x + mouths[n].x + mouths[n].width, faces[k].y + mouths[n].y + mouths[n].height);
				Point tr(faces[k].x + mouths[n].x, faces[k].y + mouths[n].y);
				rectangle(img, lb, tr, Scalar(0, 255, 255), 3, 4, 0);
			}

			rectangle(img, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
		}

	//imshow("skin", img);
	waitKey();

	return dst;
}

bool R1(int R, int G, int B) {
	bool e1 = (R>95) && (G>40) && (B>20) && ((max(R, max(G, B)) - min(R, min(G, B)))>15) && (abs(R - G)>15) && (R>G) && (R>B);
	bool e2 = (R>220) && (G>210) && (B>170) && (abs(R - G) <= 15) && (R>B) && (G>B);
	return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
	return (H<25) || (H > 230);
}

Mat GetSkin(Mat image) {
	// allocate the result matrix
	Mat dst = image.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	Mat src_ycrcb, src_hsv;
	// OpenCV scales the YCrCb components, so that they
	// cover the whole value range of [0,255], so there's
	// no need to scale the values:
	cvtColor(image, src_ycrcb, CV_BGR2YCrCb);
	// OpenCV scales the Hue Channel to [0,180] for
	// 8bit images, so make sure we are operating on
	// the full spectrum from [0,360] by using floating
	// point precision:
	image.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
	// Now scale the values between [0,255]:
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			Vec3b pix_bgr = image.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);

			if (!(a&&b&&c))
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}

	//imshow("skin", dst);
	waitKey();
	return dst;
}

Mat GetAvgColor(Mat image) {
	unsigned int R, G, B, count;
	Mat converted;
	count = R = G = B = 0;
	cvtColor(image, converted, CV_RGB2HSV);
	Vec3b cblack = Vec3b::all(0);
	
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.ptr<Vec3b>(i)[j] != cblack) {
				R += converted.ptr<Vec3b>(i)[j][0];
				G += converted.ptr<Vec3b>(i)[j][1];
				B += converted.ptr<Vec3b>(i)[j][2];
//				R += image.ptr<Vec3b>(i)[j][0];
//				G += image.ptr<Vec3b>(i)[j][1];
//				B += image.ptr<Vec3b>(i)[j][2];
				count++;
			}
		}
	}
	float avg[3] = {(float)R / count, (float)G / count, (float)B / count };
	//return Vec3b((uchar)(R / count), (uchar)(G / count), (uchar)(B / count));
	Mat avgRGB(1, 3, CV_32FC1, avg);
	cout << avgRGB.ptr<float>(0)[0] << " " << avgRGB.ptr<float>(0)[1] << " " << avgRGB.ptr<float>(0)[2] << endl;
	return avgRGB;
}

Mat flatSkin(Mat image, unsigned int num) {
	Mat res;
	int count = 0;
	Vec3b cblack = Vec3b::all(0);
	Mat flat(0, 0, CV_32FC3);
/*
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.ptr<Vec3b>(i)[j] != cblack) {
				res.push_back(image.ptr<Vec3b>(i)[j]);
				count++;
			}
			if (count > num) {
				resize(res, flat, cvSize(1, num * 3));
				return flat;
			}
		}
	}
	*/
	resize(image, flat, Size(num*3, 1), 0, 0, INTER_CUBIC);
	return flat;

}