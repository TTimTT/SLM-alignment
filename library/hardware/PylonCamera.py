import os
import cv2
import time
import torch
import queue
import subprocess
from collections import deque
import numpy as np
from pypylon import pylon
from pypylon import genicam

import matplotlib.pyplot as plt


__ref_frame_tol__ = 2
__queue_length__ = 32
__cam_nbuffer__ = 8
__xrandr_Ntry__ = 10
__screen_rst_time__ = 300


class DeQueue:
    def __init__(self, maxsize): 
        self.deque = deque(maxlen=maxsize)
    
    def put(self, elem):
        self.deque.append(elem)
    
    def get(self):
        while(not self.deque):
            time.sleep(0.001)
        return self.deque.popleft()

    def clear(self):
        self.deque.clear()

class ImageEventPrinter(pylon.ImageEventHandler):

    def __init__(self, queue_length=0): # If queue_length=0 then it is an infinite queue
        super().__init__()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_Mono8
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._imgQueue = DeQueue(maxsize=queue_length)

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")

    def OnImageGrabbed(self, camera, grabResult):
        if grabResult.GrabSucceeded():
            self._imgQueue.put([grabResult.GetArray()/255, grabResult.ChunkTimestamp.Value])
            grabResult.Release()
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())

    def clear_FIFO(self):
        """
        Clear the Pylon Camera FIFO to remove unecessary old stuff
        
        :return: None
        """
        self._imgQueue.clear()
        """
        q = self._imgQueue
        with q.mutex:
          unfinished = q.unfinished_tasks - len(q.queue)
          if unfinished <= 0:
            if unfinished < 0:
              raise ValueError('task_done() called too many times')
            q.all_tasks_done.notify_all()
          q.unfinished_tasks = unfinished
          q.queue.clear()
          q.not_full.notify_all()
        """

class PylonCamera(object):
    """
        Interface for the Pylon Basler Camera
    """
    def __init__(self, serial_id):

        # Look for camera devices
        self._tlf = pylon.TlFactory.GetInstance()
        dev_lst = self._tlf.EnumerateDevices()
        camID_lst = list(map(lambda x: x.GetSerialNumber(), dev_lst))
        print("Camera devices ID found:", camID_lst)
        
        # Declare new instance of Pylon camera
        self._camera = pylon.InstantCamera(self._tlf.CreateDevice(dev_lst[camID_lst.index(serial_id)]))
        self._camera.Open()

        # to get consistant results it is always good to start from "power-on" state
        self._camera.UserSetSelector.Value = "Default"
        self._camera.UserSetLoad.Execute()
        
        # Camera event processing must be activated first, the default is off.
        self._camera.MaxNumBuffer = __cam_nbuffer__
        self._img_handler = ImageEventPrinter(__queue_length__)
        self._camera.RegisterImageEventHandler(self._img_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # For frame synchronization
        self._reset_time = time.time()
    
    def __del__(self):
        """ Destructor to close the Basler camera object
        
        :return: None
        """
        # Releasing the resource
        self._camera.StopGrabbing()
        self._camera.DeregisterImageEventHandler(self._img_handler)
        self._camera.Close()

    def configure(self, exposure=5000, trigLine="Line2", triggerDelay=1500, frameRate=120, reverseX=True, reverseY=True):
        """ Function to open and connect to the camera, it automatically look for an available device and take the first one of the list

        :param exposure: Exposure value of the camera
        :param frameRate: FPS recording of the camera
        :param ReverseX: Boolean whether to flip horizontally
        :param ReverseY: Boolean whether to flip vertically
        
        :return: True if connection to the camera was sucessful otherwise False
        :rtype: Bool
        """
       
        # SPecify framerate to match display and avoid aliasing
        self._camera.AcquisitionFrameRate.Value = frameRate
        self._camera.AcquisitionFrameRateEnable.Value = True
        
        # Set the exposure time in ms
        self._camera.ExposureTime.Value = exposure
        self._camera.Gain.Value = 0.5

        self._camera.ReverseX.Value = reverseX
        self._camera.ReverseY.Value = reverseY

        self._camera.TriggerSelector.Value = "FrameStart"
        self._camera.TriggerMode.Value = "On"
        self._camera.TriggerSource.Value = trigLine
        self._camera.TriggerActivation.Value = "FallingEdge"
        self._camera.TriggerDelay.Value = triggerDelay

        try:
            self._camera.ChunkModeActive.Value = True
            self._camera.ChunkSelector.Value = "Timestamp"
            self._camera.ChunkEnable.Value = True
        except pylon.AccessException:
            pass


    def start(self, screens):
        # Grabing Continusely (video) with minimal delay
        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        except:
            print('Camera already grabbing')

        # Register the screens that the camera is master
        self._screens = screens
        
        # Resynchronize the screens
        time.sleep(2)
        self.reset_screen_trigger(sync_time=True)
    
    def capture_ref(self):
        """ Makes a capture to store as the internal threshold value for the SYNC Frame value comparison
        
        :return: None
        """
        self.clear_FIFO()
        for screen in self._screens:
            screen.clear()

        time.sleep(2)
        self.clear_FIFO()
        self._img_ref = self._img_handler._imgQueue.get()[0]

        self._sync_threshold = np.zeros(len(self._sync_slice))
        for k in range(len(self._sync_slice)):
            self._sync_threshold[k] = np.sum(self._img_ref[self._sync_slice[k]['y'], self._sync_slice[k]['x']])*__ref_frame_tol__

    def sync_time_calibration(self):

        # Clear the system to be in a controlled state
        for screen in self._screens:
            screen.clear()
        
        self._sync_slice = []
        for _ in range(__xrandr_Ntry__):
            # Display and let enough time for the camera to see the pattern
            img_cap_lst = []
            uD = self._screens[0]
            for id_mask, mask in enumerate(uD._sync_mask):
                uD.set_render_ready(True)
                uD.displayCUDA(imageData=mask * 255, sync_frame=False)
                self.clear_FIFO()

                time.sleep(2)
                img_cap_lst.append(self._img_handler._imgQueue.get()[0])
            
            # Post-processing getting the rectangle contour and deduce the coordinates
            for img in img_cap_lst:
                # Apply Gaussian blur to reduce noise and improve edge detection
                blurred = cv2.GaussianBlur((img * 255).astype(np.uint8), (5, 5), 0)
                
                # Otsu's thresholding after Gaussian filtering
                ret, thresh = cv2.threshold(blurred, 50, 255, 0)
                # Find contours in the edged image 
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                 
                # Loop over the contours 
                for contour in contours:
                    M = cv2.moments(contour)
                    if(M["m00"] > 64**2):
                        # Approximate the contour to a polygon 
                        epsilon = 0.1 * cv2.arcLength(contour, True) 
                        approx = cv2.approxPolyDP(contour, epsilon, True) 
                         
                        # Check if the approximated contour has 4 points (rectangle) 
                        if len(approx) == 4: 
                            x, y, w, h = cv2.boundingRect(contour)
                            #print(x, y, w, h)
                            self._sync_slice.append({'x': slice(x, x+w), 'y': slice(y, y+h)})
        
            if(len(self._sync_slice) != 4):
                print("Couldn't find all calibration pattern for camera")
            else:
                break
        
    def capture_image(self, n_frame=1, sync_frame=False, remove_ref=False):
        """ Return the copy of the buffer from the threaded method: self._img_handler.OnImageGrabbed()
        
        :return: Copy of the acquired image
        :rtype: np.array([buffer.Height(),  buffer.Width()])
        """
        # If the screen havenot been resetted for a while we need to do it
        if((time.time() - self._reset_time) >= __screen_rst_time__):
            #print("Screen reset time reached, auto-resetting before capturing new image")
            self.reset_screen_trigger()
        
        n, n_try = 0, 0
        _img_out = np.zeros((n_frame, 1200, 1920))  # TODO remove magick number
        _cam_trig_delta = np.zeros(n_frame)

        self.clear_FIFO()
        self._camera.TimestampLatch.Execute()
        self._cam_time_start = self._camera.TimestampLatchValue.Value
        self.trigger_screens()
        
        while(n < n_frame):
        
            if(not sync_frame):
                _img_out[n] = self._img_handler._imgQueue.get()[0]
                n = n + 1

            else:              
                screen_reset = True
                # Browse through the data until we have a frame (skipping empty images)
                for _ in range(__queue_length__):
                    
                    _img_out[n], trig_time = self._img_handler._imgQueue.get()
                    # Maybe make a tiny  function for the following line?
                    n_slice = n % len(self._sync_slice)
                    trig_sum = np.sum(_img_out[n][self._sync_slice[n_slice]['y'], self._sync_slice[n_slice]['x']])
                    
                    if(trig_sum > self._sync_threshold[n_slice]):
    
                        # Check that the frame is coming from after 10ms after the trigger of the screens
                        delta_t_frame = trig_time - self._cam_time_start
                        if(delta_t_frame > 9e7):
                            screen_reset = False
                            _cam_trig_delta[n] = delta_t_frame
                            break

                # Check that we dont have a duplicate!
                if((n_frame > 1) and (n+1 == n_frame) and (np.diff(_cam_trig_delta/1e6).std() > 1)):
                    #print("Issue with synchronisation, re-acquiring")
                    screen_reset = True
                    
                if(screen_reset):
                    n = 0
                    n_try += 1
                    # If after __detect_sync_frame__ trials we didnot get anything we reset the screen
                    # We make a new capture which suspend the screen and reset its queue then we reload
                    # the Cuda Tensor onto the C++ DisplayGL wrapper to retry
                    if(n_try > __xrandr_Ntry__):
                        #print("Resetting screens...")
                        self.reset_screen_trigger()
                        n_try = 0
                        
                    # When everything is ready, we get a capture for the ref of the Camera
                    self.clear_FIFO()
                    self._camera.TimestampLatch.Execute()
                    self._cam_time_start = self._camera.TimestampLatchValue.Value
                    self.trigger_screens()
                else:
                    n = n + 1
                        
        self._img_out = _img_out
        if(remove_ref):
            for k, _ in enumerate(_img_out):
                _img_out[k] = cv2.subtract(_img_out[k], self._img_ref)
        
        # TODO return maybe an iterator here TODO
        if(n_frame == 1):
            return _img_out[0]
        else:
            return _img_out
    
    def clear_FIFO(self):
        """ Clear the Pylon Camera FIFO to remove unecessary old stuff
        
        :return: None
        """
        self._img_handler.clear_FIFO()

    def trigger_screens(self):
        """ Trigger the display of the stack of images of all screens that the camera control
        
        :return: None
        """
        self._screens[0].displayCUDA()
        for screen in self._screens:
            screen.set_render_ready(True)

    def reset_screen_trigger(self, sync_time=False):
        """ Reset the trigger signals of all screens to be synchronized back to original state
        This should be unfortunately done every 10mins or so.
        
        :return: None
        """
        # First of all, we need to clear all screens
        # and stop rendering of the main window on screen[0]
        for screen in self._screens:
            screen.clear()
        self._screens[0].pause_render()

        # Have you tried turning it off/on again? (The IT Crowd)
        time.sleep(0.25)
        subprocess.run(["xrandr", "--output", "DP-1", "--off", "--nograb",
                                  "--output", "DP-2", "--off", "--nograb"])
        for n_try in range(__xrandr_Ntry__):
            time.sleep(1)
            sub_proc = subprocess.run(["xrandr", "--output", "DP-6", "--mode", "2560x1440",
                                       "--output", "DP-2", "--mode", "1920x1080",  "--right-of", "DP-6", "--nograb",
                                       "--output", "DP-1", "--mode", "1920x1200", "--right-of", "DP-2", "--nograb"])
            try:
                sub_proc.check_returncode()
            except subprocess.CalledProcessError:
                print("Error while restarting the screen")
                print("Attempt:", n_try)
                continue

            else:
                break

            finally:
                pass
        
        # When the screen is back, we can restore and render on the window on its corresponding screen as before
        time.sleep(1)
        self._screens[0].restore_window()
        time.sleep(0.1)
        self._screens[0].start_render()

        # We also reset the counter for autoreset
        self._reset_time = time.time()
        
        # When everything is ready, we get a capture for the ref of the Camera
        if(sync_time):
            self.sync_time_calibration()
        self.capture_ref()