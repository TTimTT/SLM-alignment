import os
import time
import torch
import screeninfo
import numpy as np
import cudacanvas
import matplotlib.pyplot as plt


__screen_width__ = 1920
__screen_height__ = 1080
__monitor_id__ = 1

class DisplayGL:
    """
        Interface any screen using CudaGLStreamer
    """

    # The CUDA GL Streamer once, it handles both windows    
    __GLstreamer = None
    _display_img = torch.zeros((__screen_height__, __screen_width__*2), device="cuda").to(torch.uint8).contiguous()
    __display_lst = [torch.zeros((__screen_height__, __screen_width__), device="cuda"), torch.zeros((__screen_height__, __screen_width__), device="cuda")]
    __sync_square__ = 128
    
    @classmethod
    def init_GLstreamer(cls, reverseX=False, reverseY=False):
        """
        Initialize the GLStreamer once
        :return: None
        """        
        # Initialize the GLStreamer once
        cls.__GLstreamer = cudacanvas.CudaGLStreamer()
        cls.__GLstreamer.set_image(torch.zeros((__screen_height__, __screen_width__*2), device="cuda"))
        cls.__GLstreamer.create_windows()
        cls.__GLstreamer.set_flip_xy(reverseX, reverseY)
        cls.__GLstreamer.set_windowfull(__monitor_id__)
        cls.__GLstreamer.start_render()

        # For synchronisation
        cls._sync_pattern = torch.zeros((__screen_height__, __screen_width__*2), device="cuda").to(torch.uint8).contiguous()

        # Create masking for synchronisation and duplication rejection
        cls._sync_mask = torch.zeros((4, __screen_height__, __screen_width__*2), device="cuda").to(torch.uint8).contiguous()
        for k, mask in enumerate(cls._sync_mask):
            mask[cls.__sync_square__*(k+2):cls.__sync_square__*(k+3), :cls.__sync_square__] = 1

    def __init__(self, screen_id=1, reverseX=False, reverseY=False, value_bg=0):
        """
        Initialize the Display for the given screen_id, bound it to the GLStreamer and the camera (optional)
        
        :param ReverseX: Boolean whether to flip horizontally, defaults to False
        :param ReverseY: Boolean whether to flip vertically, defaults to False
        :param value_bg: Value for the background, between [0,1] avoiding border effects with subimages, defaults to 0

        :return: Instance of a display interface for HoloEye devices
        :rtype: Display
        """
        self._value_bg = value_bg

        # Initialize the OpenGL context to stream CUDA tensor to the screen
        if self.__GLstreamer is None:
            self.init_GLstreamer(reverseX=reverseX, reverseY=reverseY)

        self._win_id = screen_id - 1
        # Wierdly enough we need to do that for single operation screen :S TODO
        self.display(np.zeros((__screen_height__, __screen_width__)))
        self.suspend()

        time.sleep(1)
        self.set_render_ready(True)
    
    def __del__(self):
        """
        Make sure we close the window when display is destroyed
        
        :return: None
        :rtype: None
        """
        self.__GLstreamer.stop_render()
        print("Warning: Display destructor not properly done!")
        
    def get_screen_size(self):
        """
        Returns the height and width of the Display object
        
        :return: self._screen.height, self._screen.width
        :rtype: Int, Int, Int
        """
        return __screen_height__, __screen_width__
        
    def toCuda(self, imageData, scaling=255):
        """
        Convert numpy array into a PyTorch CUDA tensor continuous encoded on uint8 with 256 levels
        
        :param imageData: Input data to be converted
        :param scaling: Value of maximum for the pixels on the screen (uD is 255, SLM is 128), defaults to 255

        :return: imageData CUDA
        :rtype: torch.Tensor
        """
        if(not torch.is_tensor(imageData)):
            imageData = torch.from_numpy(imageData + self._value_bg)
        return imageData.mul(scaling).to(torch.uint8).cuda()

    def display(self, imageData, sync_frame=False, CUDA2GL=True, scaling=255):
        """
        Display given image onto the associated devices screen, by converting it into a CUDA tensor
        
        :param imageData: Input data to be display as a numpy array
        :param sync_frame: Whether we add the sync frame before displaying images, defaults to False
        :param scaling: Value of maximum for the pixels on the screen (uD is 255, SLM is 127), defaults to 255
        
        :return: None
        :rtype: None
        """
        DisplayGL.__display_lst[self._win_id] = self.toCuda(imageData, scaling=scaling)

        if(DisplayGL.__display_lst[self._win_id].size()[-2:] != (__screen_height__, __screen_width__)):
            raise Exception(f"Incorrect image size {DisplayGL.__display_lst[self._win_id].size()[-2:]}! Should be ({__screen_height__}, {__screen_width__})")

        # Merge the images with the other screens images
        if(len(DisplayGL.__display_lst[0]) == len(DisplayGL.__display_lst[1])):
            DisplayGL._display_img = torch.cat((DisplayGL.__display_lst[0], DisplayGL.__display_lst[1]), -1)

            # Add sync mask onto the list/or single images ONLY ON THE UD!
            if(DisplayGL._display_img.dim() > 2):
                for k, img in enumerate(DisplayGL._display_img):
                    img += self._sync_mask[k%self._sync_mask.size()[0]] * scaling

            else:
                DisplayGL._display_img += self._sync_mask[0] * scaling

        if(CUDA2GL):
            self.displayCUDA(sync_frame=sync_frame)

    def displayCUDA(self, imageData=None, sync_frame=True):
        """
        Display given CUDA tensor(s) to the window by sending the pointer to C++ wrapper
        
        :param sync_frame: Whether we add the sync frame before displaying images, defaults to True
        
        :return: None
        :rtype: None
        """
        if(imageData is None):
            imageData = DisplayGL._display_img
            
        # Put the time synchronisation pattern at first
        if(sync_frame):
             self.__GLstreamer.set_imagePTR(self._sync_pattern.data_ptr())

        #print("foo max:", imageData.max())
        if(imageData.dim() > 2):
            for img in imageData:
                self.__GLstreamer.set_imagePTR(img.data_ptr())

        else:
            self.__GLstreamer.set_imagePTR(imageData.data_ptr())

        if(sync_frame):
            self.__GLstreamer.set_imagePTR(self._sync_pattern.data_ptr())

    def clear(self):
        """
        Send a blank image to clear the screens

        :return: None
        :rtype: None
        """
        self.__GLstreamer.clear_PTRQueue()
        self.__GLstreamer.set_imagePTR(self._sync_pattern.data_ptr())
    
    def suspend(self):
        """
        Put the screen in suspend state: it turns off the screen, clear the FIFO and set the rendering to False

        :return: None
        :rtype: None
        """
        self.__GLstreamer.clear_PTRQueue()
        self.display(torch.zeros((__screen_height__, __screen_width__), device="cuda"), CUDA2GL=False)
        self.__GLstreamer.set_imagePTR(self._sync_pattern.data_ptr())

    def pause_render(self):
        """
        Pause the rendering without deleted the window
        
        :return: None
        :rtype: None
        """
        self.__GLstreamer.pause_render()

    def start_render(self):
        """
        Start the rendering of GLstreamer wrapper
        
        :return: None
        :rtype: None
        """
        self.__GLstreamer.start_render()
    
    def set_render_ready(self, is_ready):
        """
        Activate the render by setting signal to OpenGL wrapper to announce the images stacks can be displayed
        :param is_ready: Boolean indicating if the images can be displayed or we are waiting for sync signal

        :return: None
        :rtype: None
        """
        self.__GLstreamer.set_render_ready(is_ready)

    def restore_window(self):
        """
        Move the window back to its original monitor or the screen_id passed in args
        
        :param screen_id: Monitor/Screen ID to restore the window to, defaults to None

        :return: None
        :rtype: None
        """
        self.__GLstreamer.restore_window(__monitor_id__)

    def set_value_bg(self, value_bg):
        """
        Set the bg value in case you want to change it during the experiment
        
        :param value_bg: Float between [0,1], where 1 is equivalent to white

        :return: None
        :rtype: None
        """
        if(isinstance(value_bg, np.ndarray)):
            if((value_bg.max() > 1) or (value_bg.min() < 0)):
                raise Exception("Background value should be between [0,1]")
            
            if(np.shape(value_bg)[-2:] != (__screen_height__, __screen_width__)):
                raise Exception("Invalid background image shape")

            self._value_bg = value_bg

        elif isinstance(value_bg, (int, float)):
            self._value_bg = value_bg
        else:
            raise Exception("Invalid type of background, need scalar or numpy array")
