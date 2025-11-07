import os
import cv2
import copy
import torch
import mergedeep
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

#Geometric figure import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # TODO whats the purpose?
from op_torch import GeomDMD

def stack_display(device, stackData, mult=1, margin=[0,0], marginvalue=0, padding=[0,0]):
    if torch.is_tensor(stackData):
        stackData = stackData.detach().cpu().numpy()
    stackData=np.repeat(np.repeat(stackData,mult,axis=-2),mult,axis=-1)
    m1,m2=margin[0],margin[1]
    stackData=np.pad(stackData,((0,0),(0,0),(m1,m1),(m2,m2)),constant_values=((0,0),(0,0),(marginvalue,marginvalue),(marginvalue,marginvalue)))
    px1,px2,py1,py2=padding[0]//2,padding[0]//2+padding[0]%2,padding[1]//2,padding[1]//2+padding[1]%2
    stackData=np.pad(stackData,((0,0),(0,0),(px1,px2),(py1,py2)))
    Nx1, Ny1 = device.get_screen_size()
    nim=stackData.shape[0]  #nb of images
    nx=np.floor(np.sqrt(nim)).astype('int') #One choice among others that ensures nx*ny>nim
    ny=nim//nx+1
    npad=nx*ny-nim
    ls=[stackData[i,0,::] for i in range(nim)]
    p=np.zeros_like(ls[0])
    ls+=[p for n in range(npad)]
    imageData=np.hstack([np.vstack([ls[j+i*nx] for j in range(nx)]) for i in range(ny)])
    nx1,ny1=imageData.shape
    pxa,pxb,pya,pyb=(Nx1-nx1)//2,(Nx1-nx1)//2,(Ny1-ny1)//2,(Ny1-ny1)//2
    if (pxa+nx1+pxb)%2==1:
        pxa,pxb,pya,pyb=pxa+pxa%2,pxb,pya,pyb #in odd first dim, shouldd generalize
    else:
        pass
    imageData=np.pad(imageData,((pxa,pxb),(pya,pyb)))
    device.display(imageData)
    return imageData
        
def White(device,value=1):
    Nx1, Ny1 = device.get_screen_size()
    blanc = np.ones([Nx1, Ny1])*value
    device.display(blanc.astype(np.uint8))
    
def Black(device):
    Nx1, Ny1 = device.get_screen_size()
    black = np.zeros([Nx1, Ny1])
    device.display(black.astype(np.uint8))
    
def Cible(device,value=1):
    Nx1, Ny1 = device.get_screen_size()
    Cible = GeomDMD.MotifCible(Nx1, Ny1, Nx1//3, Nx1//3-30, 30).astype(np.uint8)*value
    device.display(Cible.astype(np.uint8)//255)
    
def stack_White(device, *args, shape = [28, 28], fill_shape = [28,28], N = 128, value=1,  **kwargs):
    [Nx1, Ny1], [nx1, ny1] = shape, fill_shape
    blanc = np.zeros([Ny1, Nx1])
    blanc[Nx1//2-nx1//2:Nx1//2+nx1//2, Ny1//2-ny1//2:Ny1//2+ny1//2]=np.ones([nx1,ny1])*value
    stackData = [blanc for i in range(N)]
    stackData = np.asarray(stackData)[:,None,::].astype('int')
    return stack_display(device,  stackData,  *args, **kwargs)

def stack_Black(device, *args, shape = [28, 28], fill_shape = [28,28], N = 128, value=1,  **kwargs):
    [Nx1, Ny1], [nx1, ny1] = shape, fill_shape
    blanc = np.ones([Ny1, Nx1])*value
    blanc[Nx1//2-nx1//2:Nx1//2+nx1//2, Ny1//2-ny1//2:Ny1//2+ny1//2] = np.zeros([nx1,ny1])
    stackData = [blanc for i in range(N)]
    stackData = np.asarray(stackData)[:,None,::].astype('int')
    return stack_display(device,  stackData,  *args, **kwargs)

def stack_random( device,  *args, shape = [28, 28], fill_shape = [28,28], value=1, N = 128, **kwargs):
    [Nx1, Ny1], [nx1, ny1] = shape, fill_shape
    blanc = np.zeros([Ny1, Nx1])
    blanc[Nx1//2-nx1//2:Nx1//2+nx1//2, Ny1//2-ny1//2:Ny1//2+ny1//2]=np.random.uniform(size=[nx1,ny1])*value
    stackData = [blanc for i in range(N)]
    stackData = np.asarray(stackData)[:,None,::]
    return stack_display(device,  stackData,  *args, **kwargs)

def stack_cercle( device,  *args, shape = [28, 28], fill_shape = [28,28], value=1, N = 128, **kwargs):
    [Nx1, Ny1], [nx1, ny1] = shape, fill_shape
    blanc = GeomDMD.MotifCercle(Nx1,Ny1,nx1//2)/255
    stackData = [blanc for i in range(N)]
    stackData = np.asarray(stackData)[:,None,::]
    return stack_display(device,  stackData,  *args, **kwargs)

def stack_Cible(device,  *args, shape = [28, 28],  N = 128, value=1,  **kwargs):
    [Nx1, Ny1] = shape
    MiniCible = GeomDMD.MotifCible(Ny1, Nx1, Ny1//3, Ny1//3-3, 3)*value
    stackData = [MiniCible[None, ::] for i in range(N)]
    stackData = np.asarray(stackData).astype('int')/255
    return stack_display(device,  stackData,  *args, **kwargs)

def stack_Mixte(device,  *args,  N = 128, value=1, switch=False, **kwargs):
    Nx1, Ny1 = 28, 28
    MiniCible = GeomDMD.MotifCible(Ny1, Nx1, Ny1//3, Ny1//3-3, 3)*value/255
    blanc = np.ones([Ny1, Nx1])*value
    stackData=[]
    if switch==False:
        for i in range(N//2):
            stackData.append(MiniCible[None, ::])
            stackData.append(blanc[None,::])
    if switch==True:
        for i in range(N//2):
            stackData.append(blanc[None, ::])
            stackData.append(MiniCible[None,::])
    stackData = np.asarray(stackData)
    return stack_display(device,  stackData,  *args, **kwargs)
    
def channel_simulation(image, blur_kernel):
    """ Create a noisy linear channel when testing the calibration pattern

    :param image: The image pattern to blur

    :return: Gaussian Blurred Image
    :rtype: np.array
    """

    (Nx, Ny) = np.shape(image)
    # Create radial alpha/transparency layer. 255 in centre, 0 at edge
    Y = np.linspace(-1, 1, Ny)[None, :]*255
    X = np.linspace(-1, 1, Nx)[:, None]*255
    alpha = np.sqrt(0.15*X**2 + 0.15*Y**2)
    alpha = (255 - np.clip(0,255,alpha)).astype(np.uint8)
    return cv2.GaussianBlur(image, blur_kernel, 0) * (255-alpha)


def calibrate_circles(screen_dev, cam_dev, pat_size=(28,28), pat_pad=(2,2), cam_pad=(0,0), batch_layout=(5,5),
                      keys=['uD-in', 'uD-out'], new_centers=list(), preproc_func=lambda x: x,
                      blur_kernel=(1,1), capture_ref=True, simulation=False, update_widget=None, uD=None, img_slm=None, dev_cal="uD"):
    """ Function that realize the calibration steps using disk on a grid
    The pattern is divided into 4 separate iterations to avoid cross talks between nearby patterns.
    After displaying on the target screen and capturing on the target camera, the image are preprocessed with preproc_func.
    Contour and ellipse detection are done to obtain the coordinated from the input screen and output camera.
    The coordinates are then sort and metadata are added to the final calibration dictionnary.

    :param screen_dev: Screen device to calibrate (uD or SLM)
    :param cam_dev: Camera device to calibrate with
    :param pat_size: Size of each individual disks, defaults to (28, 28)
    :param pat_pad: Padding between each sub-pattern, defaults to (2, 2)
    :param cam_pad: Padding to add at the output dictionnary of the Camera (dx, dy), defaults to (0, 0)
    :param batch_layout: The layout of the grid also corresponds to the batch_size, defaults to (5, 5)
    :param keys: Keys to record the coordinates in the calibration dictionnary, defaults to ['uD-in', 'uD-out']
    :param new_centers: List of the new_centers to use when iterating the second screen realignment, defaults to list()
    :param preproc_func: Preprocessing routine to segment the disks in the output images, defaults to f(x)=x
    :param blur_kernel: Simulated kernel blur to apply if simulation is chosen, defaults to (1, 1)
    :param capture_ref: Whether to acquire a empty frame to substract and have better constrast in calibration
    :param simulation: Whether the calibration is simulated or physical, defaults to False
    :param update_widget: Callback for the Jupyter widget to display the calibration procedure, defaults to None

    :return: Calibration dictionnary containing the input (screen) and output (camera) coordinates for each sub-pattern position
    :rtype: dict()
    """
    # Retrieve the references images ( either black screen for uD calibration or partially-white for the SLM)
    uD.suspend()
    screen_dev.suspend()

    if(img_slm is not None):
        uD.display(img_slm*0.5, sync_frame=True, CUDA2GL=False)
        img_ref = cam_dev.capture_image(sync_frame=True)*255
    
    time.sleep(0.25)
    # Create calibration dictionnarx based on the known number of patterns
    cal_lst = list()
    
    # Create Checkboard
    for idc_offset, pat_offset in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        
        image = create_cal_pattern(batch_layout, pat_size, pat_pad, screen_dev.get_screen_size(), pat_offset, new_centers, img_slm=img_slm)

        # Check that we are not sending an empty image
        if(image.max() == 0):
            continue

        # Acquisition
        if(simulation):
            img_cap = channel_simulation(image, blur_kernel=blur_kernel)
        else:
            screen_dev.set_render_ready(False)
            if(img_slm is not None):
                uD.set_render_ready(False)
                # Display just a portion of the uD and the calibration onto the second screen
                screen_dev.display(image, sync_frame=True, CUDA2GL=False, scaling=127)
                uD.display(img_slm*0.5, sync_frame=True, CUDA2GL=False)   

            else:
                screen_dev.display(image, sync_frame=True, CUDA2GL=False)

            img_cap = cam_dev.capture_image(sync_frame=True)*255
            
            # Pre-Processing
            if((img_slm is not None) and capture_ref):
                img_sub = cv2.subtract(img_ref, img_cap)
            else:
                img_cap[cam_dev._sync_slice[0]['y'], cam_dev._sync_slice[0]['x']] = 0
                img_sub = img_cap
            img_gray = preproc_func(img_sub)
            img_rgb = cv2.cvtColor(img_gray.astype(np.float32), cv2.COLOR_GRAY2BGR)
    
            if(update_widget is not None):
                update_widget[0](img_cap)
                update_widget[1](img_gray)

            # In the case of the SLM we need to close the concentric circles
            if(img_slm is not None):
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9, 9))
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # Find the contours/Disk
            img_contours, img_hierarchy = cv2.findContours(cv2.bitwise_not((image*255).astype(np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cam_contours, cam_hierarchy = cv2.findContours(cv2.bitwise_not((img_gray).astype(np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            detect_ellipse(img_contours, pat_size, cal_lst, keys[0])
            detect_ellipse(cam_contours, pat_size, cal_lst, keys[1], cam_pad=cam_pad)
            # The second keys is the output, hence we add the cam_pad here
    
    # Sort the dictionnary
    cal_dict = sort_n_norm_cal(cal_lst, batch_layout, keys=keys)
    
    # add metadata
    cal_dict['info'][keys[0] + '_size'] = (cal_dict['data'][0][keys[0]]['rY']*2, cal_dict['data'][0][keys[0]]['rX']*2)
    cal_dict['info'][keys[1] + '_size'] = (cal_dict['data'][0][keys[1]]['rY']*2, cal_dict['data'][0][keys[1]]['rX']*2)
    cal_dict['info']['padding'] = pat_pad
    cal_dict['info']['batch'] = batch_layout
    return cal_dict

def create_cal_pattern(batch_layout, pat_size, pat_pad, screen_size, pat_offset, new_centers, img_slm=None):
    """ Function to create the disks calibration pattern.

    :param batch_layout: Layout for the batch, typically (8,8) it should create a grid
    :param pat_size: Size of the sub-pattern to be used for example (28, 28) or (56, 56)
    :param pat_pad: Padding to add between each sub-patterns
    :param screen_size: Size of the target screen
    :param pat_offset: Offset to apply for pattern (there are 2 vertical and 2 horizontal offsets)
    :param new_centers: Coordinates of the new centers in case of iterative alignment of a second screen

    :return: Image containing the calibration pattern
    :rtype: np.array
    """
    
    (img_h, img_w) = screen_size
    img_screen_tmp = np.zeros(screen_size)

    h, w, nh, nw = pat_size[0], pat_size[1], batch_layout[0], batch_layout[1]
    Dy, Dx = h + pat_pad[0], w + pat_pad[1]
    px, py = (img_w - nw*Dx)//2, (img_h - nh*Dy)//2,
    
    (i, j) = pat_offset
    for idx in range(batch_layout[1])[i::2]:
        for idy in range(batch_layout[0])[j::2]:
            id_xy = idx * batch_layout[0] + idy

            Cx = idx * Dx + Dx/2 + px
            Cy = idy * Dy + Dy/2 + py

            if new_centers:
                Cx = new_centers[id_xy][0]
                Cy = new_centers[id_xy][1]

            if(img_slm is not None):
                for wh in range(0, w, 4):
                    cv2.ellipse(img_screen_tmp, (int(np.round(Cx)), int(np.round(Cy))), (wh//2, wh//2), 0, 0, 360, (1.0, 1.0, 1.0), 1)
            else:
                cv2.ellipse(img_screen_tmp, (int(np.round(Cx)), int(np.round(Cy))), (w//2, h//2), 0, 0, 360, (1.0, 1.0, 1.0), -1)

    return img_screen_tmp

def detect_ellipse(contours, pat_size, cal_lst, key, cam_pad=(0,0)):
    """ Function detecting the ellipse given a list of contours by OpenCV.
    For each sub-patterns found, it stores the (centers, radius, x_pos, y_pos, translation) coordinates in the calibration dictionnary

    :param contours: List of the detected contours with OpenCV
    :param pat_size: Size of the sub-pattern to be used for example (28, 28) or (56, 56)
    :param cal_lst: Calibration dictionnary to update with new detected coordinates
    :param key: Key of the calibration dictionnary to update
    :param cam_pad: Padding to add at the output dictionnary of the Camera (dx, dy), defaults to (0, 0)

    :return: Calibration dictionnary containing the target device (screen) coordinates for each sub-pattern position
    :rtype: dict()
    """
    
    for c in contours:
        # calculate moments for each contour   # TODO CHECK FOR ECCENTRICITY BASED ON INITIAL IMAGE
        M = cv2.moments(c)
        if((M["m00"] > pat_size[0]*pat_size[1]//2) and (M["m00"] < 16*pat_size[0]*pat_size[1])):
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Get the width and height of ellipse
            (ell_h, ell_w) = cv2.fitEllipse(c)[1]
            ell_h2, ell_w2 = int(ell_h)//2 + cam_pad[1], int(ell_w)//2 + cam_pad[0]
            
            valuesList = [key, cX, cY, ell_w2, ell_h2, cX-ell_w2, cX+ell_w2, cY-ell_h2, cY+ell_h2]
            dict_map =  dict(zip(['key', 'cX', 'cY', 'rX', 'rY', 'xmin', 'xmax', 'ymin', 'ymax'], valuesList))
            cal_lst.append(dict_map)
            
def sort_n_norm_cal(cal_lst, batch_layout, keys):
    """ Function that sorts the calibration dictionnary to have the detected in (X,Y) coordinates ordered.

    :param cal_lst: Calibration dictionnary to sort with ALL the detected coordinates
    :param batch_layout: Layout for the batch, typically (8,8) it should create a grid
    :param keys: List of keys for all calibrated devices in the dictionnary

    :return: Sorted calibration dictionnary
    :rtype: dict()
    """
    
    # Convert list of dictionnaries to a DataFrame
    dict_df = pd.json_normalize(cal_lst)

    df_lst = []
    for key in keys:
        # Select only the current key from the dictionnary DataFrame
        key_df = dict_df[dict_df['key'] == key]
        key_df = key_df.drop('key', axis=1).astype(int)

        # Sorting
        key_df['cY_range'] = pd.cut(key_df['cY'], bins=batch_layout[0])
        key_df = key_df.sort_values(by=['cY_range', 'cX'], kind='stable').reset_index(drop=True)
        
        # Normalize the ellipse radius and create bbox coordinates
        key_df['rX'] = np.ceil(key_df['rX'].median())
        key_df['rY'] = np.ceil(key_df['rY'].median())

        key_df['xmin'] = key_df['cX'] - key_df['rX']
        key_df['xmax'] = key_df['cX'] + key_df['rX']
        key_df['ymin'] = key_df['cY'] - key_df['rY']
        key_df['ymax'] = key_df['cY'] + key_df['rY']
        
        # Saving for merge
        df_lst.append(key_df)


    merged_df = pd.concat(df_lst, axis=0, keys=keys)
    swapped_df = merged_df.swaplevel(0,1).sort_index(axis=0)
    sorted_df = swapped_df[['cX', 'cY', 'rX', 'rY', 'xmin', 'xmax', 'ymin', 'ymax']].astype(int)
    dict_sorted = {'data': {}, 'info': {}}
    dict_sorted['data'] = {level: sorted_df.xs(level).to_dict('index') for level in sorted_df.index.levels[0]}
    return dict_sorted


def display_calibration(cal_dict, update_widget, keys=['uD-in', 'uD-out'], disp_border=True, color=[(255,255,0), (0,0,255)], thickness=[3, 3], tsleep=0):
    """ Function displaying the calibration dictionnary for specified keys to plot.
    It will update the values of the Jupyter widget as to avoid any copy or creation of new plots.

    :param cal_lst: Calibration dictionnary to sort with ALL the detected coordinates
    :param update_widget: Callback of the Jupyter widget to display the calibration dictionnary keys coordinates
    :param keys: List of keys for all calibrated devices in the dictionnary to be plotted, defaults to ['uD-in', 'uD-out']
    :param disp_border: Whether the bounding box of the detected ellipse should be displayed, defaults to True

    :return: None
    """
    
    img_rgb = np.zeros((1080,1920,3), dtype=np.uint8)
    print(cal_dict['info'])
    for item in cal_dict['data'].values():
        for k, key in enumerate(keys):
            time.sleep(tsleep)
            cX, cY, rX, rY, xmin, xmax, ymin, ymax = item[key].values()

            cv2.ellipse(img_rgb, (cX, cY), (rX, rY), 0, 0, 360, color[k], thickness[k])
            
            if(disp_border):
                cv2.line(img_rgb, (xmin, ymin), (xmax, ymin), (255*abs(k-1), k*255, 255*abs(k-1)), thickness[k])
                cv2.line(img_rgb, (xmax, ymin), (xmax, ymax), (255*abs(k-1), k*255, 255*abs(k-1)), thickness[k])
                cv2.line(img_rgb, (xmax, ymax), (xmin, ymax), (255*abs(k-1), k*255, 255*abs(k-1)), thickness[k])
                cv2.line(img_rgb, (xmin, ymax), (xmin, ymin), (255*abs(k-1), k*255, 255*abs(k-1)), thickness[k])
    
    update_widget(img_rgb)

def calibrate_screens(screens, cam_dev, pat_size, pat_pad, batch_layout, n_iteration=1, update_widget=None, thres_lst=[50, 50], kern_lst=[7, 11], cam_pad=(0, 0), capture_ref=True):
    """ Functions to perform the full calibration of everything for ONN setup. It consider the first screen in the list to be the reference
    and will realign the remaining screens to make sure the display are concentric for each batched images.
    
    :param screens: List of screens devices to calibrate (uD or SLM), the first one act as the reference (typically uD)
    :param cam_dev: Camera device to measure and calibrate with
    :param pat_size: List of size of the sub-pattern to be used for example (28, 28) or (56, 56) for each screens
    :param pat_pad: Padding to add between each sub-patterns (constant between all screens)
    :param batch_layout: Layout for the batch, typically (8,8) it should create a grid
    :param n_iteration: Number of iterations to perform for the realignment of the n+1 screens
    :param update_widget: Callback for the Jupyter widget to display the calibration procedure, defaults to None

    :return: None
    """
    
    def preproc_func(img_gray, thresholding=50, kernel_size=7):
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
        #img_gray = clahe.apply(img_gray)
        _, img_gray = cv2.threshold(img_gray, thresholding, 255, cv2.THRESH_BINARY)
    
        for k in range(3, kernel_size, 2):
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k, k))
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        return img_gray

    print("Calibrating screen 0...")
    for screen in screens:
        screen.suspend()
    
    cal_dict_uD = calibrate_circles(screen_dev=screens[0],
                                    cam_dev=cam_dev,
                                    pat_size=pat_size[0],
                                    pat_pad=pat_pad,
                                    cam_pad=cam_pad,
                                    batch_layout=batch_layout,
                                    keys=['uD-in', 'uD-out'],
                                    preproc_func=lambda x: preproc_func(x, thresholding=thres_lst[0], kernel_size=kern_lst[0]),
                                    capture_ref=capture_ref,
                                    simulation=False,
                                    update_widget=update_widget, uD=screens[0])

    cal_dict = copy.deepcopy(cal_dict_uD)
    for screen_k, screen in enumerate(screens[1:]):
        cal_dict_tmp = copy.deepcopy(cal_dict_uD)
        print("Calibrating screen", screen_k + 1, "...")
        slm_offsets = np.zeros((batch_layout[0] * batch_layout[1], 2))
        for _ in range(n_iteration):
            new_centers = list()
            for idx in range(batch_layout[0]):
                for idy in range(batch_layout[1]):
                    
                    id_xy = idx * batch_layout[1] + idy
                    if('slm-in' in cal_dict_tmp['data'][id_xy]):        
                        # Compute diff to apply
                        Cx_uD_out, Cy_uD_out, _, _, _, _, _, _ = cal_dict_tmp['data'][id_xy]['uD-out'].values()
                        Cx_slm_out, Cy_slm_out, _, _, _, _, _, _ = cal_dict_tmp['data'][id_xy]['slm-out'].values()
                        Cx_slm_in, Cy_slm_in, _, _, _, _, _, _ = cal_dict_tmp['data'][id_xy]['slm-in'].values()
        
                        Cx_diff, Cy_diff = Cx_uD_out - Cx_slm_out, Cy_uD_out - Cy_slm_out
                        Cx_diff, Cy_diff = Cx_diff*0.75, Cy_diff*0.5
                        
                        # Save to the list to pass for change offset of circles
                        new_centers.append([Cx_slm_in + Cx_diff, Cy_slm_in + Cy_diff])
                        slm_offsets[id_xy, :] +=  np.array([Cx_diff, Cy_diff])
        
            # Display just a portion of the uD
            img_rgb = np.zeros(screens[0].get_screen_size())
            for item in cal_dict_tmp['data'].values():
                cX, cY, rX, rY, xmin, xmax, ymin, ymax = item['uD-in'].values()
                cv2.ellipse(img_rgb, (cX, cY), (rX*3, rY*3), 0, 0, 360, (1.0, 1.0, 1.0), -1)
        
            # SLM calibration step
            cal_dict_slm2 = calibrate_circles(screen_dev=screens[1],
                                              cam_dev=cam_dev,
                                              pat_size=pat_size[screen_k],
                                              pat_pad=pat_pad,
                                              cam_pad=cam_pad,
                                              batch_layout=batch_layout,
                                              keys=['slm-in', 'slm-out'],
                                              new_centers=new_centers,
                                              preproc_func=lambda x: preproc_func(x, thresholding=thres_lst[1], kernel_size=kern_lst[1]),
                                              capture_ref=capture_ref,
                                              simulation=False,
                                              update_widget=update_widget,
                                              uD=screens[0],
                                              img_slm=img_rgb)

            # Merge the 2 'screen' to get the final calibration dict
            cal_dict_tmp = mergedeep.merge(cal_dict_slm2, cal_dict_uD)
        cal_dict = mergedeep.merge(cal_dict_tmp, cal_dict)


    # Create the list of offsets to affect the SLM masks in the ONN
    slm_xy_offsets = []
    for slm_offset in slm_offsets:
        x_off = int(slm_offset[0])
        y_off = int(slm_offset[1])
        # (padding_left, padding_right, padding_top, padding_bottom)
        slm_xy_offsets.append((pat_pad[1]//2 + x_off, pat_pad[1]//2 - x_off, pat_pad[0]//2 + y_off, pat_pad[0]//2 - y_off))
        
    return cal_dict, slm_xy_offsets
