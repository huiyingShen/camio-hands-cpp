#ifndef __UTIL_H__
#define __UTIL_H__


#include <chrono>
#include <thread>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <iomanip>      // std::setw


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include <string>
#include <sstream>
#include <vector>
#include <iterator>

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}
std::vector<std::string> split(const std::string &s, char delim);


string getTestString(string fn);

std::string & ltrim(std::string & str);
std::string & rtrim(std::string & str);

void rotate(vector<Point3f> &vP3f, const Vec3f &rvec);

template<typename TPoint>
float dist2(TPoint p1, TPoint p2){
	p1 -= p2;
	return p1.dot(p1);
}

bool setCameraCalib(std::string calibStr, cv::Mat &camMatrix,cv::Mat &distCoeffs);
bool replace(string &input, const string &to_be_replaced, const string &replacing );
void replaceAll(string &input, const string &to_be_replaced, const string &replacing );
void getOneGroup(stringstream &ss, string &out, char cStart = '[', char cEnd=']');
vector<int> getAllPos(string s, string tok);
vector<string> getAllGroup(string s, string tokStart, string tokEnd);

template<class T> void getVector(const string &s, vector<T> &out, int size){
    stringstream ss0;
    ss0<<s;
    ss0<<'\n';
    out.resize(size);
    for (int i=0; i<out.size(); i++)
        ss0>>out[i];
}

float dist2(const Point2f &p1, const Point2f &p2);
float dist2(const Point3f &p1, const Point3f &p2);
void testDist2();

string p3fToString(const Point3f &p3f);

template<typename T>
void vec2mat(MatIterator_<T> it, const vector<double> &vec){
    for (int i=0; i<vec.size(); i++,it++)
        *it = vec[i];
}

void pointToVec3f(const Point3f &p3f, Vec3f &v3f);
void toCamera(const Matx33f &rMat, const Vec3f &tvec, Vec3f &v3f);

template<typename T>
struct SimpleCluster{
    float tol;
    vector<T> data;
    T centroid;
    int weight() const { return data.size();}
    SimpleCluster(const T &t, float tol = 0.05):tol(tol),centroid(t){data.push_back(t);}
    
    bool tryAdd(const T &t){
        float d2 = dist2(centroid,t);
        if (d2 < tol*tol){
            float w = weight();
            centroid = w/(w+ 1.0f) * centroid + 1.0/(w + 1.0f)* t;
            data.push_back(t);
            return true;
        }
        return false;
    }
    static void add(vector<SimpleCluster> &vCluster, const T &t){
        for (int i=0; i<vCluster.size(); i++){
            if (vCluster[i].tryAdd(t))
                break;
        }
        vCluster.push_back(SimpleCluster(t));
    }
    
    static int getMaxWeight(const vector<SimpleCluster> &vCluster){
        int indx = -1, w = 0;
        for (int i=0; i<vCluster.size(); i++){
            if (vCluster[i].weight()>w){
                indx = i;
                w = vCluster[i].weight();
            }
        }
        return indx;
    }
};


template<typename T>
struct TimedData{
    T data;
    chrono::time_point<chrono::system_clock> time_point;
    long timeDiffMilli(const TimedData<T> &other) const {
        return chrono::duration_cast<chrono::milliseconds>(time_point - other.time_point).count();
    }
//    TimedData():time_point (chrono::system_clock::now()){}
    TimedData(const T &data = T()):data(data),time_point (chrono::system_clock::now()){}
};


template<typename T>
class RingBuf{
public:
    int capacity;
    vector<TimedData<T> > vDat;
    int kStart;
    
    RingBuf(int capacity=999):kStart(-1),capacity(capacity){
        vDat.resize(capacity);
        for (int i=0; i<capacity; i++){
            add(T());
        }
    }
    long milli_sec_2_newest_update(){
        return (chrono::system_clock::now() - vDat[kStart].time_point).count()/1000;
    }
    long milli_sec_2_oldest_update(){
        int k = (kStart+1)%capacity;
        return (chrono::system_clock::now() - vDat[k].time_point).count()/1000;
    }
//    chrono::time_point<chrono::system_clock> newest_time_point(){
//        return vDat[kStart].time_point;
//    }
    long getDtMax(){
        auto now = chrono::system_clock::now();
        long dtMax = 0;
        for (int i=0; i<capacity; i++){
            long dt = (now - vDat[i].time_point).count()/1000; // milli sec
//            cout<<"dt = "<<dt<<endl;;
            if (dtMax < dt)
                dtMax = dt;
        }
        return dtMax;
    }
    void add(const T &dat){
//        auto now = chrono::system_clock::now();
        kStart = (kStart+1)%capacity;
        vDat[kStart] = TimedData<T>(dat);
//        long dt = (now - vDat[kStart].time_point).count();
//        cout<<"kStart = "<<kStart<<endl;
    }
};



struct RingBufP2f: RingBuf<Point2f>{
    Point2f centroid;
    RingBufP2f(int capacity=30):RingBuf<Point2f>(capacity){}
    void setCentroid();
    float getMaxDist();
    bool isFresh(long tol = 5000){
        long dtMax = getDtMax();
        cout<<"dtMax = "<<dtMax<<endl;
        return dtMax < tol;
    }
    bool isTight(float tol = 5.0){
        setCentroid();
        float md = getMaxDist();
        return md < tol;
    }
};


struct RingBufInt: RingBuf<int>{
    int nDat;
    int iMin, iMax;
    vector<float> histo;
    RingBufInt(int iMin=0, int iMax=5, int capacity=500):iMin(iMin),iMax(iMax),RingBuf<int>(capacity){
        histo.resize(iMax-iMin+1);
    }
    
    void getValidRange(long dtTolMilli);
    bool tryAdd(int dat){
        if (dat<iMin || dat>iMax) return false;
        add(dat);
        return true;
    }
    
    bool setHisto();
    
};

template<class T>
struct SimpleRingBuf{
    int cur;
    vector<T> vDat;
    chrono::time_point<chrono::system_clock> lastUpdate;
    SimpleRingBuf(int cap=30){reset(cap);}
    void reset(int cap=30){
        cur = -1;
        vDat.resize(cap);
        lastUpdate = chrono::system_clock::now();
    }
    void add(const T &d){
        cur = (cur+1)%vDat.size();
        vDat[cur] = d;
        lastUpdate = chrono::system_clock::now();
    }
};

template<typename PointT>
struct RingBufPointT: SimpleRingBuf<PointT>{
    PointT centroid;
    int nValid;
    virtual bool isValid(const PointT &p)=0;
    void setCentroid(){
        centroid *= 0;
        nValid = 0;
        for (int k=0; k<this->vDat.size(); k++)
            if (isValid(this->vDat[k])){
                nValid ++;
                centroid += this->vDat[k];
            }
        
        if (nValid > 0)
            centroid *= 1.0/nValid;
    }
    float getMaxDist(){
        setCentroid();
        float d2Max = 0;
        for (int k=0; k<this->vDat.size(); k++){
            if (isValid(this->vDat[k])){
                float d2 = dist2<PointT>(centroid,this->vDat[k]);
                d2Max = max(d2Max,d2);
            }
        }
        return sqrt(d2Max);
    }
    std::pair<float,PointT> clustering(){
        setCentroid();
        return {1.0*nValid/this->vDat.size(), centroid};
    }
};
struct RingBufPoint2f:RingBufPointT<Point2f>{
    bool isValid(const Point2f &p){
        return p.x > 0;
    }
};

//struct RingBufMatx31f: SimpleRingBuf<cv::Matx31f>{
//    RingBufMatx31f(int cap=30):SimpleRingBuf<cv::Matx31f>(cap){}
//    Matx31f getSmoothed(){
//        Matx31f out(0,0,0);
//        for (int i=0; i<vDat.size(); i++)
//            out += vDat[i];
//        return out*(1.0/vDat.size());
//    }
//};

float dist2(const Point3f &p1, const Point3f &p2);
float dist2(const Point2f &p1, const Point2f &p2);
float dist(const Point3f &p1, const Point3f &p2);
float dist(const Point2f &p1, const Point2f &p2);


struct RingBufP3f: SimpleRingBuf<cv::Point3f>{
    Point3f centroid;
    RingBufP3f(int cap=30, Point3f p3f = Point3f(-99,0,0)):SimpleRingBuf<cv::Point3f>(cap){
        for(int i=0; i<vDat.size(); i++)
            vDat[i] = p3f;
    }
    Point3f getCentroid();
};

struct ConvexHullGiftWrapping{
    vector<std::pair<int,Point2f> > vPair;
    vector<std::pair<int,Point2f> > convexHull;

    void clear() {vPair.clear(); convexHull.clear();}
    bool tryAddData(int id_, const Point2f &p);
    bool findLeft();
    bool findNext();
    void removePointsClose2Line(float tol = -0.98);
    
    bool hasId(int id_){
        for (int i=0; i<convexHull.size(); i++)
            if (convexHull[i].first == id_)
                return true;
        return false;
    }
    
    static float d2(const Point2f &p1,const Point2f &p2){
        return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
    }
    
    static float cosine(const Point2f &prev,const Point2f &cur,const Point2f &next){
        float d2b = d2(prev,cur);
        float d2c = d2(next,cur);
        return -(d2(prev,next) - d2b - d2c)/2.0/sqrt(d2b*d2c);
    }
    
    static void test0();
};

template<class T>
struct HistoT{
    vector<T> vVal;
    vector<int> vCnt;
    int indxMax;
    float fMostComm;
    void reset(){vVal.clear(); vCnt.clear();}
    void add(const T &val){
        for (unsigned int i=0; i<vVal.size(); i++)
            if (val==vVal[i]){
                vCnt[i]++;
                return;
            }
        vVal.push_back(val);
        vCnt.push_back(1);
    }
    
    int getMostCommon(){
        if (vVal.size() == 0) {
            fMostComm = 0.1;
            return -1;
        }
        int cntMax=0;
        float sum = 0;
        for (unsigned int i=0; i<vCnt.size(); i++){
            sum += vCnt[i];
            if (cntMax<vCnt[i]){
                cntMax = vCnt[i];
                indxMax = i;
            }
        }
        fMostComm = vCnt[indxMax]/sum;
        //cout<<"getMostCommon(), sum = "<<sum<<endl;
        return vVal[indxMax];
    }
    int getCntMax(){return vCnt[indxMax];}
    float get_fMostComm(){return fMostComm;}
};

pair<float,float> getProjErr(const vector<Point2f> &imagePoints, const vector<Point2f> &projectedPoints);
void contour(Mat src_gray, Mat drawing, int thresh=100);
void getCorner(Mat src_gray, Mat drawing, int thresh=200);
#endif
