//
//  OpenCVWrapper.m
//  CamIo Hands Cpp
//
//  Created by Huiying Shen on 5/30/24.
//


#import <Foundation/Foundation.h>
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>

#import "markerArray.h"
#import "camio.h"
#import "OpenCVWrapper.h"

@interface CamIoWrapper()
@property CamIO *camio;
@end

@implementation CamIoWrapper
- (id) init {
    if (self = [super init]) {
        aruco::PREDEFINED_DICTIONARY_NAME dictName = aruco::DICT_4X4_250;
        self.camio = new CamIO(dictName);
    } return self;
}

- (void) dealloc {
    delete self.camio;
}

- (void) clear {
    self.camio->clear();
}
@end
