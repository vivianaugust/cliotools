import numpy as np
from astropy.io import fits
import os
from cliotools.global_badpixfix import *


##################### Bad pixel find and fix functions ###################
# badpixelsub uses the established CLIO bad pixel maps.  These functions #
# do an independent identification of bad pixels and replaces them with  #
# the median of their n neighbors along the x-direction, because CLIO    #
# seems to show patterns in columns.                                     #
# I am using the output of bditools.findstars because that file contains #
# a list of the good images in the dataset, the ones with useable star   #
# images, and won't waste time on images that won't be used in the       #
# analysis later 

def update_progress(n,max_value):
    import sys
    import time
    import numpy as np
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(np.float(n/max_value),decimals=2)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1.:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}% ({1} of {2}): |{3}|  {4}".format(np.round(progress*100,decimals=1), 
                                                  n, 
                                                  max_value, 
                                                  "#"*block + "-"*(barLength-block), 
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()

def raw_beam_count(k):
    '''Count the number of images in a dataset in each dither'''
    from astropy.io import fits
    count0, count1 = 0,0
    for i in range(len(k)):
        sp = k['filename'][i].split('_')
        filename = sp[0]+'_'+sp[1]+'.fit'
        imhdr = fits.getheader(filename)
        if imhdr['BEAM'] == 0:
            count0 += 1
        if imhdr['BEAM'] == 1:
            count1 += 1
    return count0,count1

def build_raw_stack(k):
    '''Stack all raw images into Nx512x1024 array for Nod 0 and Nod 1.
       Requires the pandas dataframe made from the file "ABLocations" 
       from bditools.findstars output
       Written by Logan A. Pearce, 2020
    
        Parameters:
        -----------
            k : pandas dataframe
                list of skysubbed files and x/y star locations
                output from bditools.findstars
            
        Returns:
        --------
            stack0, stack1 : Nx512x1024 array
                stack of raw images in nod 0 and nod 1.
            xca0, yca0, xcb0, ycb0, xca1, yca1, xcb1, ycb1 : 1d arrays
                list of star A and B pixel locations in the stack of
                nod 0 and nod 1

    '''
    count0,count1 = raw_beam_count(k)
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(k['filename'][0]).shape
    xca0, yca0, xcb0, ycb0 = [], [], [], []
    xca1, yca1, xcb1, ycb1 = [], [], [], []
    
    print('Stacking reference images for',k['filename'][0].split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(k)):
            sp = k['filename'][i].split('_')
            filename = sp[0]+'_'+sp[1]+'.fit'
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            
            # Divide by number of coadds:
            #if imhdr['COADDS'] != 1:
             #   image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
                
            # If its nod 0:
            if imhdr['BEAM'] == 0:
                # Stack the image cube into one array
                sky0_stack[count0:count0+image.shape[0],:,:] = image
                xca0.append(k['xca'][i])
                yca0.append(k['yca'][i]) 
                xcb0.append(k['xcb'][i]) 
                ycb0.append(k['ycb'][i])
                count0 += shape[0]
            # If its nod 1:
            if imhdr['BEAM'] == 1:
                # Stack the image cube into one array
                sky1_stack[count1:count1+image.shape[0],:,:] = image
                xca1.append(k['xca'][i])
                yca1.append(k['yca'][i]) 
                xcb1.append(k['xcb'][i]) 
                ycb1.append(k['ycb'][i])
                count1 += shape[0]  
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        sky1_stack = np.zeros((count1,shape[0],shape[1]))
        sky0_stack = np.zeros((count0,shape[0],shape[1]))
        count0, count1 = 0,0
        for i in range(len(k)):
            sp = k['filename'][i].split('_')
            filename = sp[0]+'_'+sp[1]+'.fit'
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                sky0_stack[count0,:,:] = image
                xca0.append(k['xca'][i])
                yca0.append(k['yca'][i]) 
                xcb0.append(k['xcb'][i]) 
                ycb0.append(k['ycb'][i])
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                xca1.append(k['xca'][i])
                yca1.append(k['yca'][i]) 
                xcb1.append(k['xcb'][i]) 
                ycb1.append(k['ycb'][i])
                count1 += 1
    print('I found ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    return sky0_stack, sky1_stack, xca0, yca0, xcb0, ycb0, xca1, yca1, xcb1, ycb1

def build_skyframe_stack(path, skip_list=False, K_klip = 10):
    import os
    os.system('ls '+path+'*sky_0* > list')
    print('Collecting sky images for',path.split('/')[0],'...')
    with open('list') as f:
        z = f.read().splitlines()
    image = fits.getdata(z[0])
    shape = image.shape
    sky_stack = np.zeros((len(z),*shape))
    count0 = 0
    xca0, yca0, xcb0, ycb0 = [], [], [], []
    for i in range(len(z)):
        # open the image
        image = fits.getdata(z[i])
        imhdr = fits.getheader(z[i])
        # Divide by number of coadds:
        image = image/imhdr['COADDS']
        # Stack the image:
        sky_stack[count0,:,:] = image
        count0 += 1
    print('I found ',sky_stack.shape[0],' sky frames')
    os.system('rm list')
    return sky_stack 

def skysubbed_beam_count(k):
    '''Count the number of images in a dataset in each dither'''
    from astropy.io import fits
    count0, count1 = 0,0
    for i in range(len(k)):
        filename = k['filename'][i]
        imhdr = fits.getheader(filename)
        if imhdr['BEAM'] == 0:
            count0 += 1
        if imhdr['BEAM'] == 1:
            count1 += 1
    return count0,count1

def build_skysubbed_stack(k):
    '''Stack skysubbed images into Nx512x1024 array for Nod 0 and Nod 1.
       Written by Logan A. Pearce, 2020

        Parameters:
        -----------
            k : pandas dataframe
                list of skysubbed files and x/y star locations
                output from bditools.findstars
            
        Returns:
        --------
            stack0, stack1 : Nx512x1024 array
                stack of skysubbed images in nod 0 and nod 1.
    '''
    count0,count1 = skysubbed_beam_count(k)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(k['filename'][0]).shape

    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(k)):
            filename = k['filename'][i]
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            
            # Divide by number of coadds:
            # if imhdr['COADDS'] != 1:
            #   image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
            # ^ Don't divide by number of coadds here because we already did it in
            # the pcaskysub step.
                
            # If its nod 0:
            if imhdr['BEAM'] == 0:
                # Stack the image cube into one array
                sky0_stack[count0:count0+image.shape[0],:,:] = image
                count0 += shape[0]
            # If its nod 1:
            if imhdr['BEAM'] == 1:
                # Stack the image cube into one array
                sky1_stack[count1:count1+image.shape[0],:,:] = image
                count1 += shape[0]  
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        sky1_stack = np.zeros((count1,shape[0],shape[1]))
        sky0_stack = np.zeros((count0,shape[0],shape[1]))
        count0, count1 = 0,0
        for i in range(len(k)):
            filename = k['filename'][i]
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            # Divide by number of coadds:
            # image = image/imhdr['COADDS'] #<- not needed because it was
            # done in the pcaskysub step.
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                sky0_stack[count0,:,:] = image
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                count1 += 1
    #print('I found ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    return sky0_stack, sky1_stack

def plot(image):
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize
    from matplotlib.colors import LogNorm
    #%matplotlib notebook
    plt.imshow(image, origin='lower', cmap='gray',norm = ImageNormalize(image, interval=ZScaleInterval(),))
    plt.show()
    
def mask(image, xc, yc, radius = 20):
    """ Mask the stars in an image
        Parameters:
        -----------
           image : 2d image array
               image to be masked
           xc, yc : flt
               x/y pixel location of the center of the star
           radius : int
               radius of circular mask to apply to star
        Returns:
        --------
            image_masked : 2d array
                masked image
    """
    import numpy as np
    # copy image:
    image_masked = image.copy()
    for i in range(len(xc)):
        # Make a meshgrid of the image centered at star locations:
        xx,yy = np.meshgrid(np.arange(image.shape[1])-xc[i],np.arange(image.shape[0])-yc[i])
        # Make an array of the distances of each pixel from that center:
        rA=np.hypot(xx,yy)
        #image0_masked = np.ma.masked_where(rA < radius, image0)
        image_masked[np.where(rA < radius)] = np.nan
    return image_masked

def highpassfilter(stack, size = 11):
    """ High pass filter (unsharp mask) of image

        Parameters:
        -----------
           stack : 3d array
               stack of images to apply hpf to
           size : int
               extent of hpf
        Returns:
        --------
            infilt : 3d array
                stack of images with hpf applied
    """
    from scipy import ndimage
    # copy image stack:
    imfilt = stack.copy()
    # for each image:
    for i in range(imfilt.shape[0]):
        # make hpf filter:
        imhpf = ndimage.median_filter(stack[i,:,:], size)
        # subtract hpf from image:
        im = stack[i]-imhpf
        # place filter image in output stack:
        imfilt[i,:,:] = im
        update_progress(i+1,imfilt.shape[0])
    return imfilt

def findbadpix(std, chunk, n=5):
    ''' Identify bad pixels in a map of the standard deviation
        in the time domain.
        Parameters:
        -----------
        std : 2d array
            map of std deviation in the time domain of masked
            stack of images from a single nod
        chunk : 2d array
            chunk of the std dev map that does not contain any
            visible bad pixels
        n : int
            threshold = median(chunk) + n*sigma, where sigma is std dev in the 
            the chunk
        Returns:
        --------
        badpix : 2d array
            list of indicies of bad pixels
        badpixmap : 2d array
            bad pixel map: identifies pix = 1, else pix = 0
    '''
    import numpy as np
    # Select a patch of sky with no obsvious highly-variable pixels
    # and take the standard deviation in that chunk to
    # sample the "correct" noise level:
    sigma = np.std(chunk)
    # set upper threshold as some n*sigma
    threshold = np.median(chunk) + n*sigma
    # find pixels exceeding threshold
    badpix = np.where(std>threshold)
    # make a bad pixel map:
    badpixmap = np.zeros(std.shape)
    badpixmap[badpix] = 1
    return badpix, badpixmap

def badpixfix(imagesr, badpixr, dx = 15, updateprog = True):
    ''' Replace bad pixels with median of dx nearest neighbors
        along x direction

        Parameters:
        -----------
        imagesr : 3d array
            stack of images to fix
        badpixr : 2d array
            list of bad pixels, output of findbadpix
        dx : int
            how many pixels to look to right and left to
            get median replacement value.  Should be odd.

        Returns:
        --------
        imfix : 3d array
            stack of bad pixel corrected images.
    '''
    import numpy as np
    # Interpolate bad pixels:
    imfix = imagesr.copy()
    for i in range(imagesr.shape[0]):
        for j in range(len(badpixr[1])):
            dx = dx # <- must be odd
            x, y = badpixr[1][j], badpixr[0][j]
            xarray = np.arange(x-dx,x+dx+1,2)
            xarray = xarray[np.where(xarray > 0)[0]]
            xarray = xarray[np.where(xarray < 1024)[0]]
            imfix[i,y,x] = np.nanmedian(imagesr[i,y,xarray])
        if updateprog:
            update_progress(i+1,imagesr.shape[0])
    return imfix

def badpixfix_singleimage(imagesr, badpixr, dx = 15):
    ''' Replace bad pixels with median of dx nearest neighbors
        along x direction

        Parameters:
        -----------
        imagesr : 3d array
            stack of images to fix
        badpixr : 2d array
            list of bad pixels, output of findbadpix
        dx : int
            how many pixels to look to right and left to
            get median replacement value.  Should be odd.

        Returns:
        --------
        imfix : 3d array
            stack of bad pixel corrected images.
    '''
    import numpy as np
    # Interpolate bad pixels:
    imfix = imagesr.copy()
    for j in range(len(badpixr[1])):
        dx = dx # <- must be odd
        x, y = badpixr[1][j], badpixr[0][j]
        xarray = np.arange(x-dx,x+dx+1,2)
        xarray = xarray[np.where(xarray > 0)[0]]
        xarray = xarray[np.where(xarray < 1024)[0]]
        imfix[y,x] = np.nanmedian(imagesr[y,xarray])
    return imfix


