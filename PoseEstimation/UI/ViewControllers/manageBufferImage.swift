//
//  manageBufferImage.swift
//  PoseEstimation
//
//  Created by Ishwar Shelke on 23/08/23.
//  Copyright © 2023 tensorflow. All rights reserved.
//

import Foundation
import UIKit
import CoreMedia


extension CVPixelBuffer {
    

    

    
    func getCmSampleBuffer() -> CMSampleBuffer? {
        
        var sampleBuffer: CMSampleBuffer?
        
        
        
        let scale = CMTimeScale(1_000_000_000)
        
        let time = CMTime(value: CMTimeValue(2 * Double(scale)), timescale: scale)
        
        
        
        var timingInfo = CMSampleTimingInfo(duration: CMTime.invalid, presentationTimeStamp: time, decodeTimeStamp: CMTime.invalid)
        
        
        
        var _videoInfo: CMVideoFormatDescription?
        
        
        
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: self, formatDescriptionOut: &_videoInfo)
        
        
        
        guard let videoInfo = _videoInfo else {
            
#if DEBUG
            
            print("videoInfo is nil")
            
#endif
            
            return nil
            
        }
        
        
        
        CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: self, dataReady: true, makeDataReadyCallback: nil, refcon: nil, formatDescription: videoInfo, sampleTiming: &timingInfo, sampleBufferOut: &sampleBuffer)
        
        
        
        return sampleBuffer
        
    }
    
}



extension CMSampleBuffer {
    
    func getCiImage() -> CIImage? {
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(self) else {
            
            return nil
            
        }
        
        
        
        // var tempPixelBuffer: CVPixelBuffer?
        
        // let error = CVPixelBufferCreate(kCFAllocatorDefault, CVPixelBufferGetHeight(pixelBuffer), CVPixelBufferGetWidth(pixelBuffer), kCVPixelFormatType_420YpCbCr8PlanarFullRange, nil, &tempPixelBuffer)
        
        
        
        // guard error == kCVReturnSuccess, let newPixelBuffer = tempPixelBuffer else {
        
        //    return
        
        // }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        
        
        return ciImage
        
    }
    
}


extension UIImage {
    
    
    
    
    
//    func rotate(radians: Float) -> UIImage? {
//        
//        var newSize = CGRect(origin: CGPoint.zero, size: size).applying(CGAffineTransform(rotationAngle: CGFloat(radians))).size
//        
//        // Trim off the extremely small float value to prevent core graphics from rounding it up
//        
//        newSize.width = floor(newSize.width)
//        
//        newSize.height = floor(newSize.height)
//        
//        
//        
//        UIGraphicsBeginImageContextWithOptions(newSize, false, scale)
//        
//        let context = UIGraphicsGetCurrentContext()!
//        
//        
//        
//        // Move origin to middle
//        
//        context.translateBy(x: newSize.width / 2, y: newSize.height / 2)
//        
//        // Rotate around middle
//        
//        context.rotate(by: CGFloat(radians))
//        
//        // Draw the image at its center
//        
//        draw(in: CGRect(x: -size.width / 2, y: -size.height / 2, width: size.width, height: size.height))
//        
//        
//        
//        let newImage = UIGraphicsGetImageFromCurrentImageContext()
//        
//        UIGraphicsEndImageContext()
//        
//        
//        
//        return newImage
//        
//    }
    
    
    
    
    func pixelBuffer() -> CVPixelBuffer? {
        
        let width = size.width
        
        let height = size.height
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     
             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         
                                         Int(width),
                                         
                                         Int(height),
                                         
                                         kCVPixelFormatType_32ARGB,
                                         
                                         attrs,
                                         
                                         &pixelBuffer)
        
        
        
        guard status == kCVReturnSuccess else {
            
            return nil
            
        }
        
        
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        let context = CGContext(data: pixelData,
                                
                                width: Int(width),
                                
                                height: Int(height),
                                
                                bitsPerComponent: 8,
                                
                                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                                
                                space: rgbColorSpace,
                                
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        
        
        context?.translateBy(x: 0, y: height)
        
        context?.scaleBy(x: 1.0, y: -1.0)
        
        
        
        UIGraphicsPushContext(context!)
        
        draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        
        
        return pixelBuffer
        
    }
    
    
    
}
