//
//  camio.cpp
//  CamIO4iOS
//
//  Created by Huiying Shen on 1/29/18.
//  Copyright © 2018 Huiying Shen. All rights reserved.
//

#include <math.h>
#include <chrono>
#include <thread>

#include "util.h"
#include "camio.h"

Mat_<double> CamIO::camMatrix;
Mat_<double> CamIO::distCoeffs;
Mat_<double> CamIO::camRotMat;


string getTimeString(){
    auto t0 = std::time(nullptr);
    auto tm = *std::localtime(&t0);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%m/%d/%Y %H:%M:%S");
    return oss.str();
}

