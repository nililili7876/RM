#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include "RMTools.hpp"

using namespace std;
using namespace cv;
using namespace RMTools;

class SolveAngle
{
public:
	SolveAngle();
    float pitchOffset = 21;
    float yawOffset = -20;

    float scale = 0.99f;
	float f_ = 1500;

    float yaw,pitch,dist;
	vector<Point2f> rectPoint2D;

	bool shoot;

	void Generate2DPoints(Rect rect);
	void GetPose(const Rect& rect, float ballet_speed, bool small);
    void GetPoseV(const vector<Point2f>& pts, float ballet_speed,bool small);
	void Generate3DPoints(bool mode);
private:
	Mat_<double> cameraMatrix;
	Mat_<double> distortionCoefficients;
	Mat rvecs;
	Mat tvecs;
	vector<Point3f> targetPoints3D;
	float targetWidth3D{};
	float targetHeight3D{};

	int shootPriority = 0;
	float averageX;
	float averageY;

	int value;
    Mat waveBG = Mat(480,640,CV_8UC3,Scalar(0,0,0));
    DisPlayWaveCLASS *wave = new DisPlayWaveCLASS(waveBG, &value,"YAW-Wave",100,Scalar(200,100,0),Scalar(0,255,0));
};
