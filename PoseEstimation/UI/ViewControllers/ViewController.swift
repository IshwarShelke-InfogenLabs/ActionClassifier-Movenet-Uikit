// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import AVFoundation
import UIKit
import ARKit
import RealityKit
import os

final class ViewController: UIViewController, ARSessionDelegate {
    
    // MARK: Storyboards Connections
    @IBOutlet private weak var overlayView: OverlayView!
    @IBOutlet private weak var threadStepperLabel: UILabel!
    @IBOutlet private weak var threadStepper: UIStepper!
    @IBOutlet private weak var totalTimeLabel: UILabel!
    @IBOutlet private weak var scoreLabel: UILabel!
    @IBOutlet private weak var delegatesSegmentedControl: UISegmentedControl!
    @IBOutlet private weak var modelSegmentedControl: UISegmentedControl!
    
    // MARK: Pose estimation model configs
    private var modelType: ModelType = Constants.defaultModelType
    private var threadCount: Int = Constants.defaultThreadCount
    private var delegate: Delegates = Constants.defaultDelegate
    private let minimumScore = Constants.minimumScore
    
    var res : Person?
    @IBOutlet var arView: ARView!
    let session = ARSession()
    let arAnchor = AnchorEntity()
    
    let sphereEntityLS = ModelEntity(mesh: .generateSphere(radius: 0.02))
    let sphereEntityRS = ModelEntity(mesh: .generateSphere(radius: 0.02))
    let sphereEntityNose = ModelEntity(mesh: .generateSphere(radius: 0.02))
    let sphereEntityLeftWrist = ModelEntity(mesh: .generateSphere(radius: 0.02))
    let sphereEntityRightWrist = ModelEntity(mesh: .generateSphere(radius: 0.02))
    
    var frameRate: Int = 0
    
    // MARK: Visualization
    // Relative location of `overlayView` to `previewView`.
    private var imageViewFrame: CGRect?
    // Input image overlaid with the detected keypoints.
    var overlayImage: OverlayView?
    
    // MARK: Controllers that manage functionality
    // Handles all data preprocessing and makes calls to run inference.
    private var poseEstimator: PoseEstimator?
//    private var cameraFeedManager: CameraFeedManager!
    
    // Serial queue to control all tasks related to the TFLite model.
    let queue = DispatchQueue(label: "serial_queue")
    
    // Flag to make sure there's only one frame processed at each moment.
    var isRunning = false
    
    var currframe: ARFrame?
    
    // MARK: View Handling Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        configSegmentedControl()
        configStepper()
        updateModel()
//        configCameraCapture()
    }
    
    // for ar session
    override func viewDidAppear(_ animated: Bool){
        arView.session.delegate = self
        super.viewDidAppear(animated)
        
        // Run a world tracking configration - you cannot get scenedepth in body tracking
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth
        
        
        arAnchor.addChild(sphereEntityLS)
        arAnchor.addChild(sphereEntityRS)
        arAnchor.addChild(sphereEntityNose)
        arAnchor.addChild(sphereEntityLeftWrist)
        arAnchor.addChild(sphereEntityRightWrist)
        
        arView.scene.addAnchor(arAnchor)

        // Run the view's session
        arView.session.run(configuration)
    }
    


    private func configStepper() {
        threadStepper.value = Double(threadCount)
        threadStepper.setDecrementImage(threadStepper.decrementImage(for: .normal), for: .normal)
        threadStepper.setIncrementImage(threadStepper.incrementImage(for: .normal), for: .normal)
    }
    
    private func configSegmentedControl() {
        // Set title for device control
        delegatesSegmentedControl.setTitleTextAttributes(
            [NSAttributedString.Key.foregroundColor: UIColor.lightGray],
            for: .normal)
        delegatesSegmentedControl.setTitleTextAttributes(
            [NSAttributedString.Key.foregroundColor: UIColor.black],
            for: .selected)
        // Remove existing segments to initialize it with `Delegates` entries.
        delegatesSegmentedControl.removeAllSegments()
        var defaultDelegateIndex = 0
        Delegates.allCases.enumerated().forEach { (index, eachDelegate) in
            if eachDelegate == delegate {
                defaultDelegateIndex = index
            }
            delegatesSegmentedControl.insertSegment(
                withTitle: eachDelegate.rawValue,
                at: index,
                animated: false)
        }
        delegatesSegmentedControl.selectedSegmentIndex = defaultDelegateIndex
        
        // Config model segment attributed
        modelSegmentedControl.setTitleTextAttributes(
            [NSAttributedString.Key.foregroundColor: UIColor.lightGray],
            for: .normal)
        modelSegmentedControl.setTitleTextAttributes(
            [NSAttributedString.Key.foregroundColor: UIColor.black],
            for: .selected)
        // Remove existing segments to initialize it with `Delegates` entries.
        modelSegmentedControl.removeAllSegments()
        var defaultModelTypeIndex = 0
        ModelType.allCases.enumerated().forEach { (index, eachModelType) in
            if eachModelType == modelType {
                defaultModelTypeIndex = index
            }
            modelSegmentedControl.insertSegment(
                withTitle: eachModelType.rawValue,
                at: index,
                animated: false)
        }
        modelSegmentedControl.selectedSegmentIndex = defaultModelTypeIndex
    }
    

    /// Call this method when there's change in pose estimation model config, including changing model
    /// or updating runtime config.
    private func updateModel() {
        // Update the model in the same serial queue with the inference logic to avoid race condition
        queue.async {
            do {
                switch self.modelType {
                case .posenet: break
                    //          self.poseEstimator = try PoseNet(
                    //            threadCount: self.threadCount,
                    //            delegate: self.delegate)
                case .movenetLighting, .movenetThunder:
                    self.poseEstimator = try MoveNet(
                        threadCount: self.threadCount,
                        delegate: self.delegate,
                        modelType: self.modelType)
                }
            } catch let error {
                os_log("Error: %@", log: .default, type: .error, String(describing: error))
            }
        }
    }
    
    @IBAction private func threadStepperValueChanged(_ sender: UIStepper) {
        threadCount = Int(sender.value)
        threadStepperLabel.text = "\(threadCount)"
        updateModel()
    }
    
    @IBAction private func delegatesValueChanged(_ sender: UISegmentedControl) {
        delegate = Delegates.allCases[sender.selectedSegmentIndex]
        updateModel()
    }
    
    @IBAction private func modelTypeValueChanged(_ sender: UISegmentedControl) {
        modelType = ModelType.allCases[sender.selectedSegmentIndex]
        updateModel()
    }
}

// MARK: - CameraFeedManagerDelegate Methods
extension ViewController {
    // create cgimage from cvpixelbuffer
    func createCGImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
       let ciContext = CIContext()
       let ciImage = CIImage(cvImageBuffer: pixelBuffer)
       return ciContext.createCGImage(ciImage, from: ciImage.extent)
    }
    
    // newly added
    func pixelBufferFromCGImage(image:CGImage) -> CVPixelBuffer? {
        
        
        let options = [
            
            
            
            kCVPixelBufferCGImageCompatibilityKey as String: NSNumber(value: true),
            
            
            
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: NSNumber(value: true),
            
            
            
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            
            
            
        ] as CFDictionary
        
        
        
        
        
        
        
        let size:CGSize = .init(width: image.width, height: image.height)
        
        
        
        var pxbuffer: CVPixelBuffer? = nil
        
        
        
        let status = CVPixelBufferCreate(
            
            
            
            kCFAllocatorDefault,
            
            
            
            Int(size.width),
            
            
            
            Int(size.height),
            
            
            
            kCVPixelFormatType_32BGRA,
            
            
            
            options,
            
            
            
            &pxbuffer)
        
        
        
        guard let pxbuffer = pxbuffer else { return nil }
        
        
        
        
        
        
        
        CVPixelBufferLockBaseAddress(pxbuffer, [])
        
        
        
        guard let pxdata = CVPixelBufferGetBaseAddress(pxbuffer) else {return nil}
        
        
        
        
        
        
        
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)
        
        
        
        
        
        
        
        guard let context = CGContext(data: pxdata, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pxbuffer), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo:bitmapInfo.rawValue) else {
            
            
            
            return nil
            
            
            
        }
        
        
        
        context.concatenate(CGAffineTransformIdentity)
        
        
        
        context.draw(image, in: .init(x: 0, y: 0, width: size.width, height: size.height))
        
        
        
        
        
        
        
        ///error: CGContextRelease' is unavailable: Core Foundation objects are automatically memory managed
        
        
        
        ///maybe CGContextRelease should not use it
        
        
        
        //            CGContextRelease(context)
        
        
        
        CVPixelBufferUnlockBaseAddress(pxbuffer, [])
        
        
        
        return pxbuffer
        
        
        
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
//        print("AR Frame: ",session.currentFrame!)
        // assumption 60 fps
        
//        session.currentFrame?.capturedImage.si
//        if (frame.sceneDepth != nil){
            var depth = session.currentFrame?.sceneDepth?.depthMap
            frameRate += 1
            //        
            guard let depth = depth else {return}
            print("depthMap: ",depth)
            
            //        let frame = session.currentFrame?.capturedDepthData
            //        print("depth data: ",frame.capturedDepthData)
            
        if frameRate%4 == 0{
            // Get the captured image from the ARFrame.
            if let capturedImage = session.currentFrame?.capturedImage
            {
                if let cmSampleBuffer = capturedImage.getCmSampleBuffer(),
                   let ciImage = cmSampleBuffer.getCiImage(),
                   let uiImage = UIImage(ciImage: ciImage).rotate(radians: .pi/2),
                   let pixelBuffer = uiImage.pixelBuffer()
                {
                    self.runModel(pixelBuffer)
                }
                
                // let newCGImage = createCGImage(from: capturedImage)
                // let newPixelBuffer = pixelBufferFromCGImage(image: newCGImage!)
                // if let newPixelBuffer = newPixelBuffer {
                //     print("inside")
                //     runModel(newPixelBuffer)
                // }
            }
        }
        else{
            if frameRate == 60 {
                frameRate = 0
            }
        }
        
        
        if let res = res{
            
            print("keypoints: ", res.keyPoints)
            print("runmodel")
        
            print("leftShoulder: ",res.keyPoints[5].coordinate) // leftshoulder
            print("rightShoulder: ",res.keyPoints[6].coordinate) // rightshoulder
            

            let indexLeftShoulder = self.getBodyPartIndex(arr: res.keyPoints, jointName: BodyPart.leftShoulder)
            let indexRightShoulder = self.getBodyPartIndex(arr: res.keyPoints, jointName: BodyPart.rightShoulder)
            print("ind: ",indexLeftShoulder)
            
            let leftShoulderPosition = res.keyPoints[5].coordinate
            let rightShoulderPosition = res.keyPoints[6].coordinate
            let nosePosition = res.keyPoints[0].coordinate
            let leftWristPosition = res.keyPoints[9].coordinate
            let rightWristPosition = res.keyPoints[10].coordinate
            
            print("leftWrist: ", res.keyPoints[9].bodyPart)
            print("rightWrist: ", res.keyPoints[10].bodyPart)
            
            // normalized coordinates of left shoulder
            let normalizedLeftShoulder = CGPoint(x: (leftShoulderPosition.x/1440), y: (leftShoulderPosition.y/1920))
            let normalizedRightShoulder = CGPoint(x: (rightShoulderPosition.x/1440), y: (rightShoulderPosition.y/1920))
            let normalizedNose = CGPoint(x: (nosePosition.x/1440), y: (nosePosition.y/1920))
            let normalizedLeftWrist = CGPoint(x: (leftWristPosition.x/1440), y: (leftWristPosition.y/1920))
            let normalizedRightWrist = CGPoint(x: (rightWristPosition.x/1440), y: (rightWristPosition.y/1920))
            print("cgPointLeftShoulder: ", normalizedLeftShoulder)
            
            // initially image is horizontal
            // to convert the image from horizontal to verticle
            // we will send ray through this point
            let screenPositionLS = normalizedLeftShoulder.applying(CGAffineTransform.identity.scaledBy(x: arView.frame.width , y: arView.frame.height))
            let screenPositionRS = normalizedRightShoulder.applying(CGAffineTransform.identity.scaledBy(x: arView.frame.width , y: arView.frame.height))
            let screenPositionNose = normalizedNose.applying(CGAffineTransform.identity.scaledBy(x: arView.frame.width , y: arView.frame.height))
            let screenPositionLeftWrist = normalizedLeftWrist.applying(CGAffineTransform.identity.scaledBy(x: arView.frame.width , y: arView.frame.height))
            let screenPositionRightWrist = normalizedRightWrist.applying(CGAffineTransform.identity.scaledBy(x: arView.frame.width , y: arView.frame.height))
            
            let cgpointLeftShoulder = CGPoint(x: CGFloat(leftShoulderPosition.x/1440),y: CGFloat(leftShoulderPosition.y/1920))
            let cgpointRightShoulder = CGPoint(x: CGFloat(rightShoulderPosition.x/1440),y: CGFloat(rightShoulderPosition.y/1920))
            let cgpointNose = CGPoint(x: CGFloat(nosePosition.x/1440),y: CGFloat(nosePosition.y/1920))
            let cgpointLeftWrist = CGPoint(x: CGFloat(leftWristPosition.x/1440),y: CGFloat(leftWristPosition.y/1920))
            let cgpointRightWrist = CGPoint(x: CGFloat(rightWristPosition.x/1440),y: CGFloat(rightWristPosition.y/1920))
            
            // used to calculate depth
            let avPointLS = cgpointLeftShoulder//.convertVisionToAVFoundation()
            let avPointRS = cgpointRightShoulder//.convertVisionToAVFoundation()
            let avPointNose = cgpointNose//.convertVisionToAVFoundation()
            let avPointLeftWrist = cgpointLeftWrist//.convertVisionToAVFoundation()
            let avPointRightWrist = cgpointRightWrist//.convertVisionToAVFoundation()
//            var screenSpacePoint:CGPoint = CGPoint(x:0,y:0)
            
            // get coordinates wrt screen
//            screenSpacePoint = arView?.convertAVFoundationToScreenSpace(avPoint) ?? CGPoint(x:0,y:0)
            
            let rayResultLS = arView.ray(through: screenPositionLS)
            let rayResultRS = arView.ray(through: screenPositionRS)
            let rayResultNose = arView.ray(through: screenPositionNose)
            let rayResultLeftWrist = arView.ray(through: screenPositionLeftWrist)
            let rayResultRightWrist = arView.ray(through: screenPositionRightWrist)
            print("rayResultLS: ", rayResultLS!)
            print("rayResultRS: ", rayResultRS!)
            
            // depth wrt camera as origin
            let depthAtPointLS = session.currentFrame?.sceneDepth?.depthMap.value(from: avPointLS)
            let depthAtPointRS = session.currentFrame?.sceneDepth?.depthMap.value(from: avPointRS)
            let depthAtPointNose = session.currentFrame?.sceneDepth?.depthMap.value(from: avPointNose)
            let depthAtPointLeftWrist = session.currentFrame?.sceneDepth?.depthMap.value(from: avPointLeftWrist)
            let depthAtPointRightWrist = session.currentFrame?.sceneDepth?.depthMap.value(from: avPointRightWrist)

            // get world position
            // nose
            let worldOffsetNose = rayResultNose!.direction * (depthAtPointNose ?? 0.0)
            let worldPositionNose = rayResultNose!.origin + worldOffsetNose
            // left shoulder
            let worldOffsetLS = rayResultLS!.direction * (depthAtPointLS ?? 0.0)
            print("depthAtPointLS: ",depthAtPointLS,depthAtPointLeftWrist)
            let worldPositionLS = rayResultLS!.origin + worldOffsetLS
            // right shoulder
            let worldOffsetRS = rayResultRS!.direction * (depthAtPointRS ?? 0.0)
            let worldPositionRS = rayResultRS!.origin + worldOffsetRS
            // left wrist
            let worldOffsetLeftWrist = rayResultLeftWrist!.direction * (depthAtPointLeftWrist ?? 0.0)
//            print("depthAtPointLeftWrist: ",depthAtPointLeftWrist)
            let worldPositionLeftWrist = rayResultLeftWrist!.origin + worldOffsetLeftWrist
            // right wrist
            let worldOffsetRightWrist = rayResultRightWrist!.direction * (depthAtPointRightWrist ?? 0.0)
            let worldPositionRightWrist = rayResultRightWrist!.origin + worldOffsetRightWrist
            
            
            sphereEntityLS.position = worldPositionLS
            sphereEntityRS.position = worldPositionRS
            sphereEntityNose.position = worldPositionNose
            sphereEntityLeftWrist.position = worldPositionLeftWrist
            sphereEntityRightWrist.position = worldPositionRightWrist
            
            print("worldPositionLeftRightWrist",worldPositionLeftWrist)
//                    let depthAtPoint1 = session.currentFrame?.sceneDepth.value(from: avPoint)
//                    let worldOffset1 = rayResult1.direction * (depthAtPoint1 ?? 0.0)
//                    let worldPosition1 = rayResult1.origin + worldOffset1
            
//            print("raycast", rayResult!.direction,rayResult!.origin,depthAtPointLS,worldOffset ,worldPosition)
            

//                    let vec = simd_float3(x: (raycastQueryResult?.direction.x)!, y: (raycastQueryResult?.direction.y)!, z: (raycastQueryResult?.direction.z)!)
//                    print("depth: ", raycastQueryResult!)
//                    model_01.position = vec
//
//
//                    print("model_01.position: ", model_01.position, vec)
//                    guard
//
//                        let rayResult = arView.ray(through: screenSpacePoint),
//                        let rayResult1 = arView.ray(through: screenSpacePoint1)
//
//                    else {return }
//
            // pass root point
//                    let rootPoint =
//                    let countSitUps = getSideSitUpsCount(T##point: CGPoint##CGPoint)
            
            let countHandsUp = getHandsUpCount()

        }
            
        
    }
    
    func getHandsUpCount(){
        let thresholdLow = 0
        let thresholdUp = 1
        var UpSide = false
        var downSide = false
        var count=0
        
        let y = 0
        
        if((!downSide && !UpSide) && y > thresholdUp){
            downSide = true
            UpSide = true
        }
        
        if((downSide && UpSide) && y < thresholdLow){
            count += 1
        }
        
        downSide = false
        UpSide = false
    }
    
    /// Run pose estimation on the input frame from the camera.
    private func runModel(_ pixelBuffer: CVPixelBuffer) {
        // Guard to make sure that there's only 1 frame process at each moment.
        guard !isRunning else { return }
        
        // Guard to make sure that the pose estimator is already initialized.
        guard let estimator = poseEstimator else { return }
        
        // Run inference on a serial queue to avoid race condition.
        queue.async {
            self.isRunning = true
            defer { self.isRunning = false }
            
            // Run pose estimation
            do {
                let (result, times) = try estimator.estimateSinglePose(
                    on: pixelBuffer)
                
                self.res = result
                
                // Return to main thread to show detection results on the app UI.
                DispatchQueue.main.async { [self] in
                    self.totalTimeLabel.text = String(format: "%.2fms",
                                                      times.total * 1000)
                    self.scoreLabel.text = String(format: "%.3f", result.score)
                    
                    // Allowed to set image and overlay
                    let image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
                    
                    // If score is too low, clear result remaining in the overlayView.
                    if result.score < self.minimumScore {
                        self.overlayView.image = image
                        return
                    }

                    // Visualize the pose estimation result.
                    self.overlayView.draw(at: image, person: result)
                }
            } catch {
                os_log("Error running pose estimation.", type: .error)
                return
            }
        }
    }
    
    func getSideSitUpsCount(_ pointLeftHip: CGPoint, _ pointRightHip: CGPoint) -> Int {
        // actions classifier
        // 1 side sit ups
        let thresholdNegativeSide = 0
        let thresholdPositiveSide = 1
        var happeningLeft = false
        var countSideSitUps = 0
        
        let xlh = Int(pointLeftHip.x)
        let xrh = Int(pointRightHip.x)
        
        // left side
        if xlh <= thresholdNegativeSide && !happeningLeft {
            happeningLeft = true
        }
        
        // right side
        if xrh >= thresholdPositiveSide && happeningLeft{
            countSideSitUps += 1
            happeningLeft = false
        }
        
        return countSideSitUps
    }
    
    func getBodyPartIndex(arr: [KeyPoint], jointName: PoseEstimation.BodyPart) -> Int{
    var index = 0
    for item in arr {
        if item.bodyPart == PoseEstimation.BodyPart.rightShoulder {
            return index + 1
        }
        index += 1
    }
    return 0
}
}

extension KeyPoint: Equatable {
    static func == (lhs: KeyPoint, rhs: KeyPoint) -> Bool {
        return lhs.bodyPart == rhs.bodyPart && lhs.coordinate == rhs.coordinate
    }
}

enum Constants {
    // Configs for the TFLite interpreter.
    static let defaultThreadCount = 4
    static let defaultDelegate: Delegates = .gpu
    static let defaultModelType: ModelType = .movenetThunder
    
    // Minimum score to render the result.
    static let minimumScore: Float32 = 0.2
}


extension CGPoint{
    // to get the origin from top left to bottom left
    func convertVisionToAVFoundation() -> CGPoint {
        return CGPoint(x: self.y, y: self.x)
    }
}



extension ARView {
    func convertAVFoundationToScreenSpace(_ point: CGPoint) -> CGPoint {
        //Convert from normalized AVFoundation coordinates (0,0 top-left, 1,1 bottom-right)
        //to screen-space coordinates.
        if
            let arFrame = session.currentFrame,
            let interfaceOrientation = window?.windowScene?.interfaceOrientation{
            let transform = arFrame.displayTransform(for: interfaceOrientation, viewportSize: frame.size)
//            let normalizedCenter = point.applying(transform)
            let center = point.applying(CGAffineTransform.identity.scaledBy(x: frame.width, y: frame.height))
            return center
        } else {
            return CGPoint()
        }
    }

    func convertScreenSpaceToAVFoundation(_ point: CGPoint) -> CGPoint? {
        //Convert to normalized pixel coordinates (0,0 top-left, 1,1 bottom-right)
        //from screen-space coordinates.
        guard
          let arFrame = session.currentFrame,
          let interfaceOrientation = window?.windowScene?.interfaceOrientation
        else {return nil}

          let inverseScaleTransform = CGAffineTransform.identity.scaledBy(x: frame.width, y: frame.height).inverted()
          let invertedDisplayTransform = arFrame.displayTransform(for: interfaceOrientation, viewportSize: frame.size).inverted()
          let unScaledPoint = point.applying(inverseScaleTransform)
          let normalizedCenter = unScaledPoint.applying(invertedDisplayTransform)
          return normalizedCenter
    }
}


public extension CVPixelBuffer {
    ///The input point must be in normalized AVFoundation coordinates. i.e. (0,0) is in the Top-Left, (1,1,) in the Bottom-Right.

    func value(from point: CGPoint) -> Float? {
        
        let width = CVPixelBufferGetWidth(self)

        let height = CVPixelBufferGetHeight(self)

        let colPosition = Int(point.x * CGFloat(width))
        
        let rowPosition = Int(point.y * CGFloat(height))
        
        
        return value(column: colPosition, row: rowPosition)
    }
    
    func value(column: Int, row: Int) -> Float? {

        guard CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_DepthFloat32 else { return nil }

        CVPixelBufferLockBaseAddress(self, .readOnly)
        
        
        
        if let baseAddress = CVPixelBufferGetBaseAddress(self) {
            
            
            
            let width = CVPixelBufferGetWidth(self)
            
            
            
            let index = column + (row * width)
            
            
            
            let offset = index * MemoryLayout<Float>.stride
   
            let value = baseAddress.load(fromByteOffset: offset, as: Float.self)
            
            
            
            CVPixelBufferUnlockBaseAddress(self, .readOnly)
            
            
            
            return value
            
            
            
        }
        
        
        
        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        
        
        
        return nil
        
        
        
    }
}


extension UIImage {
    
    
    
    
    
    
//    
//    func pixelBuffer() -> CVPixelBuffer? {
//        
//        
//        
//        let width = self.size.width
//        
//        
//        
//        let height = self.size.height
//        
//        
//        
//        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
//                     
//                     
//                     
//             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
//        
//        
//        
//        var pixelBuffer: CVPixelBuffer?
//        
//        
//        
//        let status = CVPixelBufferCreate(kCFAllocatorDefault,
//                                         
//                                         
//                                         
//                                         Int(width),
//                                         
//                                         
//                                         
//                                         Int(height),
//                                         
//                                         
//                                         
//                                         kCVPixelFormatType_32ARGB,
//                                         
//                                         
//                                         
//                                         attrs,
//                                         
//                                         
//                                         
//                                         &pixelBuffer)
//        
//        
//        
//        
//        
//        
//        
//        guard (status == kCVReturnSuccess) else {
//            
//            
//            
//            return nil
//            
//            
//            
//        }
//        
//        
//        
//        
//        
//        
//        
//        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
//        
//        
//        
//        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
//        
//        
//        
//        
//        
//        
//        
//        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
//        
//        
//        
//        let context = CGContext(data: pixelData,
//                                
//                                
//                                
//                                width: Int(width),
//                                
//                                
//                                
//                                height: Int(height),
//                                
//                                
//                                
//                                bitsPerComponent: 8,
//                                
//                                
//                                
//                                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
//                                
//                                
//                                
//                                space: rgbColorSpace,
//                                
//                                
//                                
//                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
//        
//        
//        
//        
//        
//        
//        
//        context?.translateBy(x: 0, y: height)
//        
//        
//        
//        context?.scaleBy(x: 1.0, y: -1.0)
//        
//        
//        
//        
//        
//        
//        
//        UIGraphicsPushContext(context!)
//        
//        
//        
//        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
//        
//        
//        
//        UIGraphicsPopContext()
//        
//        
//        
//        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
//        
//        
//        
//        
//        
//        
//        
//        return pixelBuffer
//        
//        
//        
//    }
//    
//    
    
    func rotate(radians: Float) -> UIImage? {

        var newSize = CGRect(origin: CGPoint.zero, size: self.size).applying(CGAffineTransform(rotationAngle: CGFloat(radians))).size

        // Trim off the extremely small float value to prevent core graphics from rounding it up

        newSize.width = floor(newSize.width)

        newSize.height = floor(newSize.height)

        UIGraphicsBeginImageContextWithOptions(newSize, false, self.scale)

        let context = UIGraphicsGetCurrentContext()!

        // Move origin to middle
        
        
        
        context.translateBy(x: newSize.width/2, y: newSize.height/2)

        // Rotate around middle

        context.rotate(by: CGFloat(radians))
        
        
        
        // Draw the image at its center
        
        
        
        self.draw(in: CGRect(x: -self.size.width/2, y: -self.size.height/2, width: self.size.width, height: self.size.height))

        let newImage = UIGraphicsGetImageFromCurrentImageContext()

        UIGraphicsEndImageContext()
        
        
        return newImage
        
        
        
    }
    
    
    
}




// normalizing to get the value in the range (0,1)
//
