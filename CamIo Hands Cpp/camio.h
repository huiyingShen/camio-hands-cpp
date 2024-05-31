//
//  camio.hpp
//  CamIO4iOS
//
//  Created by Huiying Shen on 1/29/18.
//  Copyright © 2018 Huiying Shen. All rights reserved.
//

#ifndef camio_hpp
#define camio_hpp

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>


#include "util.h"
#include "markerArray.h"

class CamIO {
    enum Action { NewRegion, Add2Region, SelectRegion, DeleteRegion, Exploring };

    cv::Ptr<aruco::Dictionary> dictionary;
    cv::Ptr<aruco::DetectorParameters> detectionParams;
    MarkerArray markers;



    chrono::time_point<chrono::system_clock> time_point;
    chrono::time_point<chrono::system_clock> lastStylusSeen;



public:
    static Mat_<double> camMatrix, distCoeffs;
    static Mat_<double> camRotMat;
public:

	CamIO(aruco::PREDEFINED_DICTIONARY_NAME dictName = aruco::DICT_4X4_250) :markers(dictName),
          time_point(chrono::system_clock::now()),lastStylusSeen(chrono::system_clock::now()){
        
        camMatrix = Mat_<double>(3,3);
        distCoeffs = Mat_<double>(1,5);
        distCoeffs = 0;
	}
    
   
   
    void clear() {}
    
};

#endif /* camio_hpp */
