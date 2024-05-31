//
//  markerArray.cpp
//  CamIO4iOS
//
//  Created by Huiying Shen on 9/11/18.
//  Copyright © 2018 Huiying Shen. All rights reserved.
//


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco.hpp>

#include "util.h"
#include "markerArray.h"

using namespace std;
using namespace cv;

namespace {
    bool _DEBUG = false;
}


void Marker3dGroup::fillterMarkers(const MarkerArray &markers, vector<Point2f> &imagePoints, vector<Point3f> &objectPoints){
    for (int i=0; i<vMarker3d.size(); i++){
        const vector<Point2f> *corners = markers.getCorner(vMarker3d[i].id_);
        if (corners==0) continue;
        imagePoints.insert(imagePoints.end(), corners->begin(),corners->end());
        objectPoints.insert(objectPoints.end(), vMarker3d[i].vP3f.begin(), vMarker3d[i].vP3f.end());
    }
}

bool Marker3dGroup::getConvexHullMarkers(const MarkerArray &markers, vector<Marker2d3d> &vMarker){
    convexHull.clear();
    detectedIds.clear();
    for (int i=0; i<vMarker3d.size(); i++) {
        const vector<Point2f> *corners = markers.getCorner(vMarker3d[i].id_);
        if (corners !=0 ){
            convexHull.tryAddData(vMarker3d[i].id_, (*corners)[0]);
            detectedIds.push_back(vMarker3d[i].id_);
        }
    }
    
    if (!convexHull.findLeft()) return false;
    while(convexHull.findNext())
        ;
    convexHull.removePointsClose2Line();
    
    vMarker.clear();
    for (int i=0; i<vMarker3d.size(); i++){
    if (!convexHull.hasId(vMarker3d[i].id_)) continue;
//        cout<<"1: vMarker3d[i].id_ = "<<vMarker3d[i].id_<<endl;
        const vector<Point2f> *corners = markers.getCorner(vMarker3d[i].id_);
        if (corners!=0)
            vMarker.push_back(Marker2d3d(vMarker3d[i].id_,*corners,vMarker3d[i].vP3f));
    }
    return true;
}

bool Marker3dGroup::detect(const vector<Marker2d3d> &vMarker,const Mat &camMatrix, const Mat &distCoeffs){
    
    imagePoints.clear();
    objectPoints.clear();
    for (int i=0; i<vMarker.size(); i++){
        const vector< Point2f > &c2d = vMarker[i].corners2d;
        const vector< Point3f > &c3d = vMarker[i].corners3d;
        imagePoints.insert(imagePoints.end(), c2d.begin(),c2d.end());
        objectPoints.insert(objectPoints.end(), c3d.begin(),c3d.end());
    }
    nIdDetected = imagePoints.size()/4;
    if (imagePoints.size()<nIdMin*4) return false;  //at least there are 2 markers
    if (_DEBUG) cout<<"imagePoints.size() = "<<imagePoints.size()<<endl;
        
    medianDist = Marker3dGroup::getMedianDist(imagePoints);
//    solvePnP(objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec);  // iterative for accuracy
//
//    cv::projectPoints(objectPoints, rvec, tvec, camMatrix, distCoeffs, projectedPoints);
//    pair<float,float> err = getProjErr(imagePoints,projectedPoints);
//    errMean = err.first;
//    errMax = err.second;
    solve(camMatrix, distCoeffs);
    if (_DEBUG) cout<<"errMean, errMax = "<<errMean<<", "<<errMax<<endl;
    if (errMean>errMeanTol || errMax>errMaxTol) {
        Matx31f r(rvec),t(tvec);
        if (_DEBUG){
            cout<<"r = "<<r<<endl;
            cout<<"t = "<<t<<endl;
            for (int i=0; i<imagePoints.size(); i++)
                cout<<i<<", "<<imagePoints[i]<<", "<<projectedPoints[i]<<", "<<objectPoints[i]<<endl;
        }
        return false;
    }
    return true;
}
 
Mat Marker3dGroup::rvecDiff0(const Vec3d &rvec0, const Vec3d &rvec1) {
	Mat rMat0, rMat1, rMat, rot_vec(1, 3, CV_32F);
	Rodrigues(rvec0, rMat0);
	Rodrigues(rvec1, rMat1);
	rMat = rMat0 * rMat1.t();
	Rodrigues(rMat, rot_vec);
	return rot_vec;
}

int MarkerArray::removeMarkerClose2ImageCorners(const cv::Size &imageSize, vector< vector< Point2f > > &corners,vector< int > &ids){
    int sz0 = corners.size();
    Point2f center = Point2f(imageSize.width/2,imageSize.height/2);
    float dMax = sqrt(center.dot(center));
    for (int i = 0; i<corners.size(); ){
        bool close2corner = false;
        vector< Point2f > &cn = corners[i];
        for (int j=0; j<4; j++){
            Point2f diff = center - cn[j];
            if (sqrt(diff.dot(diff)) > 0.9*dMax)
                close2corner = true;
        }
        if (close2corner){
            corners.erase(corners.begin() + i);
            ids.erase(ids.begin() + i);
        }
        else
            i++;
    }
    return sz0 - corners.size();
}

int MarkerArray::copyValidMarkers(const MarkerArray &src, const vector< int > &ids0){
    clear();
    for (int j=0; j<ids0.size(); j++){
        for (int i=0; i<src.ids.size(); i++)
            if (src.ids[i]==ids0[j]) {
                ids.push_back(src.ids[i]);
                corners.push_back(src.corners[i]);
            }
    }
    if (_DEBUG) cout<<"copyValidMarkers(), size() = "<<size()<<endl;
    return (int)ids.size();
}

void MarkerArray::draw(const  vector< int > &ids, const vector<vector<Point2f> > &corners, Mat &bgr, float mult, Scalar color){
    if (corners.size()==0) return;
    vector<vector<Point2f> > vc = corners;
    for (int i=0; i<vc.size(); i++){
        for (int j=0; j<vc[i].size(); j++)
            vc[i][j] *= mult;
    }
    
    aruco::drawDetectedMarkers(bgr, vc,noArray(),color);
    for (int i = 0; i < ids.size(); i++){
        std::stringstream s;
        s << ids[i];
        int x = vc[i][0].x, y = vc[i][0].y;
        cv::putText(bgr, s.str(), cv::Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
}

void MarkerArray::drawXYZ(const Point3f &p3, int precision, const Point &xy, Mat &bgr, int fontFace, float fontScale, const Scalar &color){
    std::stringstream s0, s1, s2;
    s0 << "x = " << std::setprecision(precision) << p3.x;
    putText(bgr, s0.str(), cv::Point(xy.x, xy.y), fontFace, fontScale, color);
    s1 << "y = " << std::setprecision(precision) << p3.y;
    putText(bgr, s1.str(), cv::Point(xy.x+100, xy.y), fontFace, fontScale, color);
    s2 << "z = " << std::setprecision(precision) << p3.z;
    putText(bgr, s2.str(), cv::Point(xy.x+200, xy.y), fontFace, fontScale, color);
}

//void MarkerArray::drawAxis(Mat &bgr,const Mat &camMatrix, const Mat &distCoeffs, int idMin, int idMax){
//    for (int i=0; i<ids.size(); i++){
//        if (ids[i]<idMin || ids[i]>idMax) continue;
//        std::stringstream s;
//        s << ids[i];
//        int x = corners[i][0].x, y = corners[i][0].y;
//        cv::putText(bgr, s.str(), cv::Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
//        aruco::drawAxis(bgr, camMatrix, distCoeffs, rvecs[i], tvecs[i], 0.05);
//    }
//}

void MarkerArray::test0(){
    
}
