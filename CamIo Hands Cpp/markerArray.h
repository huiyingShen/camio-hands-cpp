//
//  markerArray.hpp
//  CamIO4iOS
//
//  Created by Huiying Shen on 9/11/18.
//  Copyright © 2018 Huiying Shen. All rights reserved.
//

#ifndef markerArray_hpp
#define markerArray_hpp



#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco.hpp>
#include "util.h"

using namespace std;
using namespace cv;
  

struct MarkerBuf: SimpleRingBuf<std::vector<Point2f> >{
    MarkerBuf(int cap = 30 ):SimpleRingBuf<std::vector<Point2f> >(cap){
        for (int i=0; i<vDat.size(); i++){
            vDat[i].resize(4);
            for (int j=0; j<4; j++){
                float x = ((double) rand() / (RAND_MAX)) + 1, y = ((double) rand() / (RAND_MAX)) + 1;
                vDat[i][j] = Point2f(x,y);
            }
        }
        time_point = chrono::system_clock::now() - chrono::duration<int,std::ratio<3600> >();  //3600s before now;
    }
    
    vector<vector<Point2f> > inliner;
    
    vector<Point2f> bestMarker;
    chrono::time_point<chrono::system_clock> time_point;
    
    bool isStale(int sec = 5*60){
        return (chrono::system_clock::now() - time_point).count() > sec;
    }
    bool tryAdd(const vector<Point2f> &vp){
        if (vp.size() !=4 ) return false;
        add(vp);
        return true;
    }
    static Point2f getCentroid(const vector<vector<Point2f> > &vDat){
        Point2f p(0,0);
        for (int i=0; i<vDat.size(); i++){
            const vector<Point2f> &vp = vDat[i];
            Point2f p1 = vp[0] +vp[1] + vp[2] + vp[3];;
            p1 *= 0.25;
            p += p1;
        }
        if (vDat.size() > 0)
            p *= 1.0/vDat.size();
        return p;
    }
    
    static int removeOutliner(vector<vector<Point2f> > &vDat, float tol = 2.0){
        int cnt = vDat.size();
        Point2f p0 = getCentroid(vDat);
//        cout<<"removeOutliner(), p0: "<<p0<<endl;
        for (int i=0; i<vDat.size(); ){
            vector<Point2f> &vp = vDat[i];
            Point2f p = vp[0] +vp[1] + vp[2] + vp[3];
            p *= 0.25;
//            cout<<"removeOutliner(), p: "<<p<<endl;
            if (dist2(p,p0) > tol*tol)
                vDat.erase(vDat.begin() + i);
            else
                i++;
        }
        return cnt - vDat.size();
    }
    
    int findBestMarker(const vector<vector<Point2f> > &inliner){
        Point2f p0 = getCentroid(inliner);
        float d2Min = 9999.0;
        int indx = -1;
        for (int i=0; i<inliner.size(); i++){
            const vector<Point2f> &vp = inliner[i];
            Point2f p = vp[0] +vp[1] + vp[2] + vp[3];
            p *= 0.25;
            float d2 = dist2(p,p0);
            if (d2 < d2Min){
                d2Min = d2;
                indx = i;
            }
        }
        return indx;
    }
    
    bool haveDetection(float tol, int minMarkers){
        inliner = vDat;
        while (removeOutliner(inliner,tol) > 0 ){
            continue;
        }
        if (inliner.size() < minMarkers) return false;
        int indx = findBestMarker(inliner);
        if (indx == -1) return false;
        bestMarker = inliner[indx];
        time_point = chrono::system_clock::now();
        cout<<"bestMarker = "<<bestMarker<<endl;
        return true;
    }
    
};

struct Marker3d{
    int id_;
    float length;
    vector<Point3f> vP3f;
    Marker3d(int id_=0, float length=2.0){
        init(id_,length);
    }
    
    void init(int id_, float length){
        this->id_= id_;
        this->length = length;
        vP3f.resize(4);
        float sz = length/2.0;
        vP3f[0] = Point3f(-sz,sz,0);
        vP3f[1] = Point3f(sz,sz,0);
        vP3f[2] = Point3f(sz,-sz,0);
        vP3f[3] = Point3f(-sz,-sz,0);
    }
    void translate(const Point3f &p){
        for (int i=0; i<4; i++)
            vP3f[i] += p;
    }
    void rotateX(float angle){
        Vec3f rvec(angle/180.0*3.1415926,0,0);
        ::rotate(vP3f,rvec);
    }
    void rotateY(float angle){
        Vec3f rvec(0,angle/180.0*3.1415926,0);
        ::rotate(vP3f,rvec);
    }
    void rotateZ90(){
        Vec3f rvec(0,0,3.1415926/2);
        ::rotate(vP3f,rvec);
    }
    friend ostream& operator<<(ostream& os, const Marker3d& m);
};

struct MarkerPair{
    Marker3d marker3;
    MarkerBuf marker2;
    
    void init(int id_, float length, int rotz, const Point3f &center){
        marker3.init(id_,length);
        for (int i=0; i<rotz; i++)
            marker3.rotateZ90();
        marker3.translate(center);
    }
};

struct MarkerBase4{
    MarkerPair marker00,marker01,marker11, marker10;
    vector<MarkerPair *> vPair;
    int id_;
    vector<int> locOrder;

    void init(int id_ = 249){
        float w = 29.75*0.0254, h = 23.25*0.0254;
        this->id_ = id_;
        float sz0 = 3*0.0254;
        float length = sz0*0.8;
        marker00.init(id_,length,0,Point3f(-sz0*0.5,     sz0*0.5,    0));
        marker01.init(id_,length,0,Point3f( sz0*0.5,    -sz0*0.5 + w,0));
        marker11.init(id_,length,0,Point3f( sz0*0.5 + h,-sz0*0.5 + w,0));
        marker10.init(id_,length,0,Point3f(-sz0*0.5 + h, sz0*0.5,    0));
        vPair.clear();
        vPair.push_back(&marker00);
        vPair.push_back(&marker01);
        vPair.push_back(&marker11);
        vPair.push_back(&marker10);
        
        locOrder.push_back(00);
        locOrder.push_back(01);
        locOrder.push_back(11);
        locOrder.push_back(10);
    }

    
    bool isStale(int sec = 5*60){
        for (int i=0; i<4; i++)
            if (vPair[i]->marker2.isStale(sec))
                return true;
        return false;
    }
    
    bool tryAddFrame(const vector< int > &ids, const vector< vector< Point2f > > &corners, int loc){
        if (ids.size() != corners.size()) return false;
        int indx = -1;
        for (int i=0; i<ids.size(); i++)
            if (ids[i] == id_){
                indx = i;
                break;
            }
        if (indx == -1) return false;
        switch (loc) {
            case 00:
                marker00.marker2.tryAdd(corners[indx]);
                break;
            case 01:
                marker01.marker2.tryAdd(corners[indx]);
                break;
            case 11:
                marker11.marker2.tryAdd(corners[indx]);
                break;
            case 10:
                marker10.marker2.tryAdd(corners[indx]);
                break;
            default:
                return false;
        }
        return true;
    }
    
    void test2(const vector< int > &ids, const vector< vector< Point2f > > &corners){
        
        
    }
    void test1(const vector< int > &ids, const vector< vector< Point2f > > &corners){
        if (!tryAddFrame(ids,corners,00)) return;
        float tol = 5;
        int minMarkers = corners.size()/2;
        bool b = marker00.marker2.haveDetection(tol,minMarkers);
        if (b){
            cout<<"MarkerBase4, test1(), detected!"<<endl;
        }
        
    }
};

struct MarkerArray;  // forward declaration

struct Marker2d3d {
    int id_ = -1;
    vector< Point2f > corners2d;
    vector< Point3f > corners3d;
    Marker2d3d(int id_,const vector< Point2f > &corners2d, const vector< Point3f > &corners3d):id_(id_),corners2d(corners2d),corners3d(corners3d){}
};

struct MarkerDetection{
    vector<Marker2d3d> vMarker;
    Mat rvec, tvec;
};

struct SolvePnp{
    vector<cv::Point3f> objectPoints;
    vector<cv::Point2f> imagePoints;
    vector<cv::Point2f> projectedPoints;

    cv::Mat rvec, tvec;
    float errMean, errMax;
    
    void solve(const cv::Mat &camMatrix, const cv::Mat &distCoeffs){
        cv::solvePnP(objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec);  // iterative for accuracy

        cv::projectPoints(objectPoints, rvec, tvec, camMatrix, distCoeffs, projectedPoints);
        pair<float,float> err = ::getProjErr(imagePoints,projectedPoints);
        errMean = err.first;
        errMax = err.second;
    }
};
struct Marker3dGroup: SolvePnp{
    vector<Marker3d> vMarker3d;
    vector<Point2f> projectedPoints;

    ConvexHullGiftWrapping convexHull;

    vector<int> detectedIds;
    
    chrono::time_point<chrono::system_clock> valid_time_point;

    int nIdMin;
    int nIdDetected;
    float medianDist;
    float errMeanTol,errMaxTol;
    
    
    Marker3dGroup(){
        // make sure valid_time_point is too old
        nIdMin = 2;
        std::tm start{};
        std::time_t t = std::mktime(&start);
        valid_time_point = chrono::system_clock::from_time_t(t);
    }
	void clear() { vMarker3d.clear(); }
	void push_back(const Marker3d &m3d) { vMarker3d.push_back(m3d); }
    bool detect(MarkerArray &markers, Mat &bgr,const Mat &camMatrix, const Mat &distCoeffs);
    bool getConvexHullMarkers(const MarkerArray &markers, vector<Marker2d3d> &vMarker);
    void fillterMarkers(const MarkerArray &markers, vector<Point2f> &imagePoints, vector<Point3f> &objectPoints);
    
    bool detect(const MarkerArray &markers, const Mat &camMatrix, const Mat &distCoeffs){
        vector<Marker2d3d> vMarker;
        getConvexHullMarkers(markers,vMarker);
        return Marker3dGroup::detect(vMarker,camMatrix, distCoeffs);
    }
    bool detect(const vector<Marker2d3d> &vMarker, const Mat &camMatrix, const Mat &distCoeffs);

	Mat tvecDiff(const Marker3dGroup &other) { return other.tvec - tvec; }
	static Mat rvecDiff0(const Vec3d &rvec0, const Vec3d &rvec1);
	Mat rvecDiff(const Marker3dGroup &other) { return rvecDiff0(rvec, other.rvec); }
    //FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255)
    void drawDetectErrors(Mat &bgr, const cv::Point &p, int fontFace=FONT_HERSHEY_SIMPLEX, float fontScale=0.33, Scalar color=Scalar(0,255,0)){
        std::stringstream s;
        s << "reproj. err = " << std::setprecision(3) << errMean<<", "<<std::setprecision(3) << errMax;//<<", "<<std::setprecision(3) << medianDist;
        cv::putText(bgr, s.str(), p, fontFace, fontScale, color);
    }
    void draw_tvec(Mat &bgr, const cv::Point &p, int fontFace=FONT_HERSHEY_SIMPLEX, float fontScale=0.33, Scalar color=Scalar(0,255,0)){
        Matx31f tmp(tvec);
        std::stringstream s;
        s<<"t = ";
        for (int i=0; i<3; i++)
            s<<std::setprecision(3) << tmp(i,0)<<",";
//        <<std::setprecision(3) << tmp(1,0)<<","<<std::setprecision(3) << tmp(2,0);
        cv::putText(bgr, s.str(), p, fontFace, fontScale, color);
    }
    void drawProjectedPoints(Mat &bgr, Scalar color=Scalar(0,255,0)){
        for (int i=0; i<projectedPoints.size(); i++){
            cv::circle(bgr, projectedPoints[i],5,color,3);
        }
    }
    
    void drawAxis(Mat &bgr, const Mat &camMatrix, const Mat &distCoeffs, float l = 0.01){
        aruco::drawAxis(bgr, camMatrix, distCoeffs, rvec, tvec, l);
    }
    
    static float getMedianDist(const vector<Point2f> &v2f){
        Point2f centroid = Point2f(0,0);
        for (int i=0; i<v2f.size(); i++)
            centroid += v2f[i];
        centroid *= 1.0/v2f.size();
        
        vector<float> vd2;
        for (int i=0; i<v2f.size(); i++){
            Point2f tmp = centroid - v2f[i];
            vd2.push_back(tmp.dot(tmp));
        }
        std::sort(vd2.begin(), vd2.end());
        
        return sqrt(vd2[vd2.size()/2]);
    }
};

ostream& operator<<(ostream& os, const Marker3d& m);

struct ArucoObj{
    cv::Ptr<aruco::Dictionary> dictionary;
    cv::Ptr<aruco::DetectorParameters> detectionParams;
    
    float squareLength, markerLength;
    ArucoObj(aruco::PREDEFINED_DICTIONARY_NAME dictName = aruco::DICT_5X5_250):dictionary(aruco::getPredefinedDictionary(dictName)){
        detectionParams = aruco::DetectorParameters::create();
        detectionParams->cornerRefinementMethod = 2; //corner refinement, 2 -> contour
        detectionParams->adaptiveThreshWinSizeMin = 33;
        detectionParams->adaptiveThreshWinSizeMax = 53;
    }
    
    ArucoObj(cv::Ptr<aruco::Dictionary> dictionary):dictionary(dictionary){
        detectionParams = aruco::DetectorParameters::create();
        detectionParams->cornerRefinementMethod = 2; //corner refinement, 2 -> contour
        detectionParams->adaptiveThreshWinSizeMin = 33;
        detectionParams->adaptiveThreshWinSizeMax = 53;
    }
    void setParam(float squareLength, float markerLength){
        this->squareLength = squareLength;
        this->markerLength = markerLength;
    }
};

struct SingleMarker {
	int id_ = -1;
	vector< Point2f > corners;
	Vec3d rvec, tvec;
	SingleMarker() {}
	void reset() { id_ = -1; }
    SingleMarker(int id_, const vector< Point2f > &corners):id_(id_),corners(corners) {}
	SingleMarker(int id_, const vector< Point2f > &corners, const Vec3d &rvec, const Vec3d  &tvec) { 
		init(id_, corners, rvec, tvec); 
	}
	void init(int id_, const vector< Point2f > &corners, const Vec3d &rvec, const Vec3d  &tvec){
		this->id_ = id_;
		this->corners = corners;
		this->rvec = rvec;
		this->tvec = tvec;
	}
};


struct MarkerArray: public ArucoObj{
    vector< int > ids;
    vector< vector< Point2f > > corners;
    vector<Vec3d> rvecs, tvecs;
    
    
    MarkerArray(aruco::PREDEFINED_DICTIONARY_NAME dictName=aruco::DICT_4X4_250):ArucoObj(dictName){}
    MarkerArray(cv::Ptr<aruco::Dictionary> dictionary):ArucoObj(dictionary){}

    void clear(){ids.clear(); corners.clear();}
    static bool isIn(int i, const vector<int> &vi){
        for (int k: vi)
            if (i==k)
                return true;
        return false;
    }
    const vector<Point2f> *getCorner(int id_) const {
        int indx = id2Index(id_);
        if (indx !=-1 )  return &(corners[indx]);
        return 0;
    }
    int id2Index(int id_) const {
        for (int i=0; i<ids.size(); i++)
            if (id_==ids[i])
                return i;
        return -1;
    }
    int copyValidMarkers(const MarkerArray &src, const vector< int > &ids0);
    void detect(const Mat &gray){
        aruco::detectMarkers(gray, dictionary, corners, ids, detectionParams);
        removeMarkerClose2ImageCorners(gray.size(),corners, ids);
    }
    
    int removeMarkerClose2ImageCorners(const cv::Size &imageSize, vector< vector< Point2f > > &corners, vector< int > &ids);

	bool getMarkerById(int id_, SingleMarker &marker) {
		for (int i = 0; i < ids.size(); i++)
			if (id_ == ids[i]) {
				marker.init(id_, corners[i], rvecs[i], tvecs[i]);
				return true;
			}

		return false;
	}
    size_t size() const { return ids.size();}
    
    static void draw(const  vector< int > &ids, const vector<vector<Point2f> > &corners, Mat &bgr, float mult=1.0, Scalar color= Scalar(0, 255, 0));
    
    void draw(Mat &bgr, float mult=1.0, Scalar color= Scalar(127, 127, 0)){draw(ids,corners,bgr,mult,color);}
    
    void estimatePose(float markerLength,const Mat &camMatrix, const Mat &distCoeffs){
        aruco::estimatePoseSingleMarkers(corners,markerLength,camMatrix,distCoeffs,rvecs,tvecs);
    }
    static void drawXYZ(const Point3f &p3, int precision, const cv::Point &xy, Mat &bgr, int fontFace, float fontScale, const Scalar &color);
    //void drawAxis(Mat &bgr,const Mat &camMatrix, const Mat &distCoeffs, int idMin=0, int idMax=250);
    
    static void test0();
};


#endif /* markerArray_hpp */
