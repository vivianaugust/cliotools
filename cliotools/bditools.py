from astropy.io import fits
import numpy as np
import os
from scipy import ndimage
import image_registration
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def plot(image):
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize
    from matplotlib.colors import LogNorm
    #%matplotlib notebook
    plt.imshow(image, origin='lower', cmap='gray',norm = ImageNormalize(image, interval=ZScaleInterval(),))
    plt.show()

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

############################ Finding stars in CLIO images ################################

def daostarfinder(scienceimage, x, y, boxsize = 100, threshold = 1e4, fwhm = 10, verbose = True):
    """Find the subpixel location of a single star in a single clio BDI image.
       Written by Logan A. Pearce, 2020

       Parameters:
       -----------
       image : 2d array
           boxsize by boxsize substamp image of a reference psf for cross correlation
       scienceimage_filename : string
           path to science image
       x, y : int
           integer pixel locations of star
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
       threshold : int
           threshold keyword for StarFinder
       fwhm : int
           fwhm keyword for StarFinder
           
       Returns:
       --------
       x_subpix, y_subpix : flt
           subpixel location of star
    """
    from photutils import DAOStarFinder
    image2 = scienceimage
    # The settings fwhm = 10.0 seems to be good for clio
    # Define box around source B:
    ymin, ymax = y-boxsize, y+boxsize
    xmin, xmax = x-boxsize, x+boxsize
    # Correct for sources near image edge:
    if ymin < 0:
        ymin = 0
    if ymax > scienceimage.shape[0]:
        ymax = scienceimage.shape[0]
    if xmin < 0:
        xmin = 0
    if xmax > scienceimage.shape[1]:
        xmax = scienceimage.shape[1]
    # Use DAOStarFinder to find subpixel locations:
    # If the threshold is too high and it can't find a point source, lower the threshold
    # until ~ 5e2, then declare failure.
    try:
        #print('threshold =',threshold)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
        sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                     np.int_(xmin):np.int_(xmax)])
        x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
        # If it finds too many sources, keep uping the search threshold until there is only one:
        while len(sources) > 1:
            threshold += 500
            daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
            sources = daofind(image2[np.int_(y-boxsize):np.int_(y+boxsize),np.int_(x-boxsize):np.int_(x+boxsize)])
            x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
    except:
        try:
            threshold=1e3
            #print('threshold =',threshold)
            daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
            sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                         np.int_(xmin):np.int_(xmax)])
            x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
            while len(sources) > 1:
                threshold += 250
                daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
                sources = daofind(image2[np.int_(y-boxsize):np.int_(y+boxsize),np.int_(x-boxsize):np.int_(x+boxsize)])
                x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
        except:
            try:
                threshold=5e2
                #print('threshold =',threshold)
                daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
                sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                             np.int_(xmin):np.int_(xmax)])
                x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
                while len(sources) > 1:
                    threshold += 250
                    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
                    sources = daofind(image2[np.int_(y-boxsize):np.int_(y+boxsize),np.int_(x-boxsize):np.int_(x+boxsize)])
                    x_subpix, y_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
            except:
                if verbose:
                    print("daostarfinder: Failed to find all stars.")
                x_subpix, y_subpix = np.nan, np.nan

    return x_subpix, y_subpix

def findstars(imstamp, scienceimage_filename, nstars, \
              boxsize = 100, threshold = 1e4, fwhm = 10, radius = 20):
    """Find the subpixel location of all stars in a single clio BDI image using DAOStarFinder 
       (https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html).
       Written by Logan A. Pearce, 2020
       Dependencies: numpy, astropy, scipy, photutils

       Parameters:
       -----------
       imstamp : 2d array
           boxsizex by boxsizey substamp image of a reference psf for cross correlation
       scienceimage_filename : string
           path to science image
       nstars : int
           number of stars in the image.
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
       threshold : int
           threshold keyword for DAO StarFinder
       fwhm : int
           fwhm keyword for DAO StarFinder
       radius : int
           radius to use when masking stars in image
           
       Returns:
       --------
       x_subpix, y_subpix : arr,flt
           1 x nstars array of subpixel x location and y location of stars
    """
    from scipy import signal
    from photutils import DAOStarFinder
    
    # Open science target image:
    image = fits.getdata(scienceimage_filename)
    # If image is a cube, take the first image:
    if len(image.shape) == 3:
        image = image[0]
    # Median filter to smooth image:
    image = ndimage.median_filter(image, 3)
    # Use cross-correlation to find int(y,x) of star A (brightest star) in image:
    corr = signal.correlate2d(image, imstamp, boundary='symm', mode='same')
    # Find the location of the brightest star in the image:
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    # Make container to hold results:
    x_subpix, y_subpix = np.array([]), np.array([])
    # Make a copy of the correlation image to mask:
    corr_masked = corr.copy()
    image_masked = image.copy()
    # For each star in the image:
    for i in range(nstars):
        # Use DAO Star Finder to find the subpixel location of the star at that location:
        xs, ys = daostarfinder(image_masked, x, y, boxsize = boxsize, threshold = threshold, fwhm = fwhm)
        x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
        # Make a mask around that star on the cross-correlation image:
        # Make a meshgrid of the image centered at the last found star:
        xx,yy = np.meshgrid(np.arange(image.shape[1])-x_subpix[i],np.arange(image.shape[0])-y_subpix[i])
        # Make an array of the distances of each pixel from that center:
        rA=np.hypot(xx,yy)
        # Mask wherever that distance is less than the set radius and
        # set those pixels to zero:
        corr_masked[np.where((rA < radius))] = 0
        image_masked[np.where((rA < radius))] = 0
        # Now find the new highest correlation which should be the next star:
        y, x = np.unravel_index(np.argmax(corr_masked), corr.shape)
        # Repeat until all stars are found.

    return x_subpix, y_subpix

def findstars_in_dataset(dataset_path, nstars, xca, yca, corrboxsizex = 40, corrboxsizey = 40, boxsize = 100, skip_list = False, \
                         append_file = False, threshold = 1e4, radius = 20, fwhm = 10):
    """Find the subpixel location of stars A and B in a clio BDI dataset.
       Written by Logan A. Pearce, 2020
       Dependencies: numpy, astropy, scipy, photutils

       Parameters:
       -----------
       dataset_path : string
           path to science images including image prefixes and underscores.  
           ex: An image set of target BDI0933 with filenames of the form BDI0933__00xxx.fit
               would take as input a path string of 'BDI0933/BDI0933__'
       xca, yca : int
           integer pixel locations of a star in the first image of the dataset, rough guess 
       corrrboxsize : int
           size of box to draw around star psfs for imstamp and cross correlation
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
       skip_list : bool
           By default script will make a list of all "skysub" images in given directory.
           Set to True if a list of paths to science files has already been made.  List
           must be named "list".  
        append_file : bool
            Set to True to append to an existing locations file, False to make a new file or 
            overwrite an old one.  Default = False.
        threshold : flt
            threshold for finding stars using DAOStarFinder

       Returns:
       --------
       writes subpixel location of stars to file called 'StarLocations' in order:
           image_filename   x   y   x   y   ...ect. for all requested stars.
    """
    from scipy import ndimage
    from cliotools.pcaskysub import update_progress
    # Supress warnings when failing to find point sources
    import warnings
    warnings.filterwarnings("ignore")
    # Make a file to store results:
    newfile = dataset_path.split('/')[0]+'/ABLocations'
    
    if append_file == False:
        string = '#     '
        for i in range(nstars):
            string += 'xc'+str(i+1) + '     ' + 'yc'+str(i+1)  + '     ' 
        string += "\n"
        k = open(newfile, 'w')
        k.write(string)
        k.close()
    
    # Make a list of all images in dataset:
    if skip_list == False:
        os.system('ls '+dataset_path+'0*_skysub.fit > list')
    with open('list') as f:
        ims = f.read().splitlines()
    # Open initial image in dataset:
    image = fits.getdata(ims[0])
    if len(image.shape) == 3:
        image = image[0]
    # Apply median filter to smooth bad pixels:
    image = ndimage.median_filter(image, 3)
    # Create referance stamp from initial image of A:
    imstamp = np.copy(image[np.int_(yca-corrboxsizey):np.int_(yca+corrboxsizey),np.int_(xca-corrboxsizex):np.int_(xca+corrboxsizex)])
    
    count = 0
    for im in ims:
        # For each image in the dataset, find subpixel location of stars:
        x_subpix, y_subpix = findstars(imstamp, im, nstars, \
                                                           boxsize = boxsize, threshold = threshold, \
                                                           radius = radius, fwhm = fwhm)
        if any(np.isnan(x_subpix)) or any(np.isnan(y_subpix)):
            # If any of the stars were failed to find, mark this entry with a comment:
            string = '# '+ im + ' '
        else:
            string = im + ' '
        for i in range(nstars):
            string += str(x_subpix[i]) + '     ' + str(y_subpix[i])  + '     ' 
        string += "\n"

        k = open(newfile, 'a')
        k.write(string)
        k.close()
        
        count+=1
        update_progress(count,len(ims))
    print('Done')
    os.system('rm list')
    os.system("say 'done'")

################################ prepare images for KLIP ##################################################

def rotate_clio(image, imhdr, center = None, interp = 'bicubic', bordermode = 'constant', cval = 0, scale = 1):
    """Rotate CLIO image to north up east left.  Uses OpenCV image processing package
       Written by Logan A. Pearce, 2020
       
       Dependencies: OpenCV

       Parameters:
       -----------
       image : 2d array
           2d image array
       imhdr : fits header object
           header for image to be rotated
       center : None or tuple
           (x,y) subpixel location for center of rotation.  If center=None,
           computes the center pixel of the image.
       interp : str
            Interpolation mode for OpenCV.  Either nearest, bilinear, bicubic, or lanczos4.
            Default = bicubic
       bordermode : str
            How should OpenCV handle the extrapolation at the edges.  Either constant, edge, 
            symmetric, reflect, or wrap.  Default = constant
       cval : int or np.nan
            If bordermode = constant, fill edges with this value.  Default = 0
       scale : int or flt
            scale parameter for OpenCV.  Scale = 1 does not scale the image.  Default = 1
           
       Returns:
       --------
       imrot : 2d arr 
           rotated image with north up east left
    """
    import cv2
    NORTH_CLIO = -1.80
    derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
    
    if interp == 'bicubic':
        intp = cv2.INTER_CUBIC
    elif interp == 'lanczos4':
        intp = cv2.INTER_LANCZOS4
    elif interp == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interp == 'nearest':
        intp = cv2.INTER_NEAREST
    else:
        raise ValueError('Interpolation mode: please enter nearest, bilinear, bicubic, or lanczos4')
        
    if bordermode == 'constant':
        bm = cv2.BORDER_CONSTANT 
    elif bordermode == 'edge':
        bm = cv2.BORDER_REPLICATE 
    elif bordermode == 'symmetric':
        bm = cv2.BORDER_REFLECT
    elif bordermode == 'reflect':
        bm = cv2.BORDER_REFLECT_101
    elif bordermode == 'wrap':
        bm = cv2.BORDER_WRAP
    else:
        raise ValueError('Border mode: please enter constant, edge, symmetric, reflect, or wrap')
        
    y, x = image.shape
    if not center:
        center = (0.5*((image.shape[1])-1),0.5*((image.shape[0])-1))
    M = cv2.getRotationMatrix2D(center, derot, scale)
    imrot = cv2.warpAffine(image, M, (x, y),flags=intp, borderMode=bm, borderValue=cval)

    return imrot
    

def ab_stack_shift(k, boxsize = 50, path_prefix='', verbose = True):
    """Prepare cubes for BDI by stacking and subpixel aligning image 
       postage stamps of star A and star B.
       Written by Logan A. Pearce, 2020
       
       Dependencies: astropy, scipy

       Parameters:
       -----------
       k : Pandas array
           Pandas array made from the output of bditools.findstars_in_dataset.  
           Assumes column names are ['filename', 'xca','yca', 'xcb', 'ycb']
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
       path_prefix : int
           string to put in front of filenames in input file in case the relative
           location of files has changed
           
       Returns:
       --------
       astamp, bstamp : 3d arr 
           stack of aligned psf's of star A and B for BDI.
    """
    from cliotools.bditools import daostarfinder
    from scipy import ndimage
    import warnings
    warnings.filterwarnings('ignore')
    # define center:
    center = (0.5*((2*boxsize)-1),0.5*((2*boxsize)-1))
    # define fwhm for starfinder
    fwhm = 7.8
    # open first image to get some info:
    i = 0
    image = fits.getdata(path_prefix+k['filename'][i])
     # create empty containers:
    if len(image.shape) == 2:
        astamp = np.zeros([len(k),boxsize*2,boxsize*2])
        bstamp = np.zeros([len(k),boxsize*2,boxsize*2])
    elif len(image.shape) == 3:
        astamp = np.zeros([len(k)*image.shape[0],boxsize*2,boxsize*2])
        bstamp = np.zeros([len(k)*image.shape[0],boxsize*2,boxsize*2])
    # For coadded images:
    if len(image.shape) == 2: 
        # initialize counter:
        count = 0
        # for each image:
        for i in range(len(k)):
            # Open a postage stamp of star A and B:
            # Star A:
            a = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                   np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
            # Star B:
            b = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                   np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
            try:
                # check to see if the postage stamp is the correct size, or if a star was
                # to close to an image edge to be contained in the same size box as requested.
                # If not, this step will fail and skip to the exception, and cut off
                # the cube at this image.
                astamp[count,:,:] = a.copy()
                bstamp[count,:,:] = b.copy()
                # Use DAOStarFinder to find the subpixel location of each star within the image stamp:
                xa,ya = daostarfinder(a, boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                xb,yb = daostarfinder(b, boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                # If StarFinder failed to find a star, just skip it:
                if np.isnan(xa):
                    print(k['filename'][i],'Failed')
                    pass
                elif np.isnan(xb):
                    print(k['filename'][i],'Failed')
                    pass
                else:
                    # Compute offset of star center from center of image:
                    dx,dy = xa-center[0],ya-center[0]
                    # shift new stamp by that amount:
                    ashift = ndimage.shift(a, [-dy,-dx], output=None, order=3, mode='constant', \
                                                  cval=0.0, prefilter=True)
                    # put into the final cube:
                    astamp[count,:,:] = ashift
                    # repeat for b:
                    dx,dy = xb-center[0],yb-center[0]
                    # shift new stamp by that amount:
                    bshift = ndimage.shift(b, [-dy,-dx], output=None, order=3, mode='constant', \
                                                  cval=0.0, prefilter=True)
                    bstamp[count,:,:] = bshift
                    # update counter so that skipped images are skipped over and the cube
                    # can be truncated at the end
                    count += 1
            except:
                # if the star got too close to the edge, truncate the stacking:
                if verbose:
                    print('PrepareCubes: Oops! the box is too big and one star is too close to an edge. I cut it off at i=',count)
                astamp = astamp[:count,:,:]
                bstamp = bstamp[:count,:,:]
                return astamp, bstamp

    # For image cubes:         
    if len(image.shape) == 3:
        coadd_count = 0
        count = 0
        for i in range(len(k)):
            try:
                a = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                                                                   np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
                astamp[coadd_count:coadd_count+a.shape[0],:,:] = a
                b = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                                                                   np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
                bstamp[coadd_count:coadd_count+b.shape[0],:,:] = b
                   
                for j in range(0,a.shape[0]):
                    count = count+1
                    xa,ya = daostarfinder(a[j], boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                    xb,yb = daostarfinder(b[j], boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                    if np.isnan(xa):
                        print(k['filename'][i],'Failed')
                        pass
                    elif np.isnan(xb):
                        print(k['filename'][i],'Failed')
                        pass
                    else:
                        dx,dy = xa-center[0],ya-center[0]
                        astamp[coadd_count+j,:,:] = ndimage.shift(astamp[coadd_count+j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                                  prefilter=True)
                        dx,dy = xb-center[0],yb-center[0]
                        bstamp[coadd_count+j,:,:] = ndimage.shift(bstamp[coadd_count+j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                                  prefilter=True)
                coadd_count += a.shape[0]
            except:
                if verbose:
                    print('ab_stack_shift: Oops! the box is too big and one star is too close to an edge. \
                          I cut it off at i=',count)
                astamp = astamp[:coadd_count,:,:]
                bstamp = bstamp[:coadd_count,:,:]
                return astamp, bstamp

    # Chop off the tops of the cubes to eliminate zero arrays from skipped stars:
    if astamp.shape[0] > count:
        astamp = astamp[:count,:,:]
    if bstamp.shape[0] > count:
        bstamp = bstamp[:count,:,:]
    return astamp,bstamp


def PrepareCubes(k, boxsize = 20, path_prefix='', verbose = True,\
                   # normalizing parameters:\
                   normalize = True, normalizebymask = False,  normalize_radius = [],\
                   # star masking parameters:\
                   inner_mask_core = True, inner_radius_format = 'pixels', inner_mask_radius = 1., cval = 0,\
                   # masking outer annulus:\
                   outer_mask_annulus = True, outer_radius_format = 'pixels', outer_mask_radius = None,\
                   # subtract radial profile from cubes:\
                   subtract_radial_profile = True,\
                   # User supplied cubes:
                   acube = None, bcube = None
                ):
    '''Assemble cubes of images and prepare them for KLIP reduction by: centering/subpixel-aligning images along
    vertical axis, normalizing images by dividing by sum of pixels in image, and masking the core of the central star.
    Written by Logan A. Pearce, 2020
    Dependencies: numpy, astropy.io.fits

    Parameters:
    -----------
    k : Pandas array
        Pandas array made from the output of bditools.findstars_in_dataset.  
        Assumes column names are ['filename', 'xca','yca', 'xcb', 'ycb'].  If
        acube,bcube are supplied, this will be a dummy variable.
    boxsize : int
        size of box to draw around star psfs for DAOStarFinder
    path_prefix : int
        string to put in front of filenames in input file in case the relative
        location of files has changed.
    verbose : bool
        if True, print status updates
    normalize : bool
        if True, normalize each image science and reference image integrated flux by dividing by each image
        by the sum of all pixels in image.  If False do not normalize images. Default = True
    normalizebymask : bool
        if True, normalize using only the pixels in a specified radius to normalize.  Default = False
    normalize_radius : flt
        if normalizebymask = True, set the radius of the aperture mask.  Must be in units of lambda/D.
    inner_mask_core : bool
        if True, set all pixels within mask_radius to value of 0 in all images.  Default = True
    inner_mask_format : str
        request inner core mask in lamdba/D or pixels.  Default = lambda/D
    inner_mask_radius : flt
        radius of inner mask in L/D or pixels.  Default = 1. lamdba/D
    mask_outer_radius : flt
        Set all pixels within this radius to 0 if mask_core set to True.  Must be units of lambda/D.
    mask_outer_annulus: bool
        if True, set all pixels outside the specified radius to zero.  Default = True.
    outer_radius : flt
        Set all pixels outside this radius to 0 if mask_outer_aunnulus set to True.
    subtract_radial_profile : bool
        If True, subtract the median radial profile from each image in the cubes.  Default = True.
    acube, bcube : 3d array
        User can supply already stacked and aligned image cubes and skip the alignment step.
        
    Returns:
    --------
    3d arr, 3d arr
        cube of images of Star A and B ready to be put into KLIP reduction pipeline.
    '''
    if np.size(acube) == 1:
        # collect and align postage stamps of each star:
        from cliotools.bditools import ab_stack_shift
        astack, bstack = ab_stack_shift(k, boxsize = boxsize,  path_prefix=path_prefix, verbose = verbose)
    else:
        # copy user-supplied cubes:
        astack, bstack = acube.copy(),bcube.copy()
        if astack.shape[0] != bstack.shape[0]:
            # if using different set of images as basis set, limit the size
            # of the two cubes to the smallest image set:
            minsize = np.min([astack.shape[0],bstack.shape[0]])
            astack = astack[:minsize]
            bstack = bstack[:minsize]
    # define center of images:
    center = (0.5*((astack[0].shape[0])-1),0.5*((astack[0].shape[1])-1))
    ######## Normalize ###########
    if normalize:
        from cliotools.bditools import normalize_cubes
        astack2,bstack2 = normalize_cubes(astack,bstack, normalizebymask = normalizebymask,  radius = normalize_radius)
    else:
        astack2,bstack2 = astack.copy(),bstack.copy()
    ####### Inner mask ###########
    if inner_mask_core:
        from cliotools.bditools import mask_star_core
        # Set all pixels within a radius of the center to cval:
        astack3,bstack3 = mask_star_core(astack2, bstack2, inner_mask_radius,center[0], center[1], 
                                    radius_format = inner_radius_format, cval = cval)
    else:
        astack3,bstack3 = astack2.copy(),bstack2.copy()
    ######## Outer mask ##########
    if outer_mask_annulus:
        from cliotools.bditools import mask_outer
        # Set all pixels exterior to a radius of the center to cval:
        if not outer_mask_radius:
            raise ValueError('Outer radius must be specified if mask_outer_annulus == True')
        astack4,bstack4 = mask_outer(astack3, bstack3, outer_mask_radius, center[0], center[1], radius_format = outer_radius_format, cval = cval)
    else:
        astack4,bstack4 = astack3.copy(),bstack3.copy()
    ######## Radial Profile ##########
    if subtract_radial_profile:
        from cliotools.bditools import radial_subtraction_of_cube
        if inner_mask_core:
            exclude_r = inner_mask_radius
        else:
            exclude_r = 0.
        if outer_mask_annulus:
            exclude_outer = outer_mask_radius
        else:
            exclude_outer = astack4[0].shape[0]

        astack5 = radial_subtraction_of_cube(astack4, exclude_r = exclude_r, exclude_outer = exclude_outer, update_prog = False)
        bstack5 = radial_subtraction_of_cube(bstack4, exclude_r = exclude_r, exclude_outer = exclude_outer, update_prog = False)
    else:
        astack5,bstack5 = astack4.copy(),bstack4.copy()

    return astack5, bstack5


def normalize_cubes(astack, bstack, normalizebymask = False, radius = []):
    ''' Normalize each image in a cube.

    Parameters:
    ----------
    astack, bstack : 3d array
        cube of unnormalized images
    normalizebymask : bool
        if True, normalize using only the pixels in a specified radius to normalize.  Default = False
    radius : flt
        if normalizebymask = True, set the radius of the aperture mask.  Must be in units of lambda/D.
        
    Returns:
    --------
    3d arr
        cube of normalized images
    '''
    a,b = astack.copy(),bstack.copy()
    for i in range(astack.shape[0]):
        # for star A:
        data = astack
        if normalizebymask:
            from photutils import CircularAperture
            r = lod_to_pixels(5., 3.9)/2
            positions = (astamp.shape[1]/2,astamp.shape[2]/2)
            aperture = CircularAperture(positions, r=radius)
            # Turn it into a mask:
            aperture_masks = aperture.to_mask(method='center')
            # pull out just the data in the mask:
            aperture_data = aperture_masks.multiply(data[i])
            aperture_data[np.where(aperture_data == 0)] = np.nan
            summed = np.nansum(aperture_data)
        else:
            # get the sum of pixels in the mask:
            summed = np.sum(data[i])
            # normalize to that sum:
            a[i] = data[i] / summed
        # repeat for B:
        data = bstack
        if normalizebymask:
            from photutils import CircularAperture
            r = lod_to_pixels(5., 3.9)/2
            positions = (astamp.shape[1]/2,astamp.shape[2]/2)
            aperture = CircularAperture(positions, r=radius)
            # Turn it into a mask:
            aperture_masks = aperture.to_mask(method='center')
            # pull out just the data in the mask:
            aperture_data = aperture_masks.multiply(data[i])
            aperture_data[np.where(aperture_data == 0)] = np.nan
            summed = np.nansum(aperture_data)
        else:
            # get the sum of pixels in the mask:
            summed = np.sum(data[i])
            b[i] = data[i] / summed
    return a,b

def mask_star_core(astack, bstack, radius, xc, yc, radius_format = 'pixels', cval = 0):
    ''' Set the core of the star flux to zero pixel value

    Parameters:
    ----------
    astack, bstack : 3d array
        cube of unnormalized images
    radius : flt
        Exclude pixels interior to this radius  
    xc, yc : flt
        x/y of star center
    mask_format : str
        What quantity is the requested mask radius. Either 'lamdba/D' or 'pixels'.  Default = 'pixels'.
    cval : flt or nan
        value to fill masked pixels.  Warning: if set to nan, this will cause problems in 
        the reduction step.
        
    Returns:
    --------
    3d arr
        cube of normalized images
    '''
    shape = astack[0].shape
    xx,yy = np.meshgrid(np.arange(shape[0])-xc,np.arange(shape[1])-yc)
    # compute radius of each pixel from center:
    r=np.hypot(xx,yy)
    if radius_format == 'lambda/D':
        from cliotools.cliotools import lod_to_pixels
        radius = lod_to_pixels(radius, 3.9)
    elif radius_format == 'pixels':
        pass
    else:
        raise ValueError('please specify mask_format = lambda/D or pixels')
    anans,bnans = astack.copy(),bstack.copy()
    for i in range(astack.shape[0]):
        anans[i][np.where(r<radius)] = cval
        bnans[i][np.where(r<radius)] = cval
    return anans,bnans

def mask_outer(astack, bstack, radius, xc, yc, radius_format = 'pixels', cval = 0):
    ''' Set the core of the star flux to zero pixel value

    Parameters:
    ----------
    astack, bstack : 3d array
        cube of images
    radius : flt
        Exclude pixels exterior to this radius
    xc, yc : flt
        x/y of star center
    radius_format : str
        How to handle requested outer annulus mask radius. Either 'pixels' or 'lambda/D'.
    cval : flt or nan
        value to fill masked pixels.  Warning: if set to nan, this will cause problems in 
        the reduction step.
        
    Returns:
    --------
    3d arr
        cube of normalized images
    '''
    shape = astack[0].shape
    xx,yy = np.meshgrid(np.arange(shape[0])-xc,np.arange(shape[1])-yc)
    # compute radius of each pixel from center:
    r=np.hypot(xx,yy)
    if radius_format == 'lambda/D':
        from cliotools.cliotools import lod_to_pixels
        radius = lod_to_pixels(radius, 3.9)
    elif radius_format == 'pixels':
        pass
    else:
        raise ValueError('please specify mask_format = lambda/D or pixels')
    anans,bnans = astack.copy(),bstack.copy()
    for i in range(astack.shape[0]):
        anans[i][np.where(r>radius)] = cval
        bnans[i][np.where(r>radius)] = cval
    return anans,bnans

def radial_subtraction_of_cube(cube, exclude_r = 5., exclude_outer = 50., update_prog = True):
    from cliotools.bditools import update_progress
    from cliotools.miscellany import radial_data_median_only, CenteredDistanceMatrix
    from scipy.interpolate import interp1d
    Nimages = cube.shape[0]
    radsub = cube.copy()
    # Create integer distance matrix
    r = np.int_(CenteredDistanceMatrix(cube[0].shape[0]))
    # For each image in cube:
    for N in range(Nimages):
        # Compute 1d median radial profile:
        RadialProfile = radial_data_median_only(cube[N])
        x = np.arange(np.max(r)+1)
        # interpolate into 2d:
        f = interp1d(x, RadialProfile)
        p = f(np.int_(r.flat)).reshape(r.shape)
        p[np.where(r<exclude_r)] = 0
        p[np.where(r>exclude_outer)] = 0
        radsub[N] = cube[N] - p
        if update_prog:
            update_progress(N+1,Nimages)
    return radsub

def psfsub_cube_header(dataset, K_klip, star, shape, stampshape):
    """ Make a header for writing psf sub BDI KLIP cubes to fits files
        in the subtract_cubes function
    """
    import time
    header = fits.Header()
    header['COMMENT'] = '         ************************************'
    header['COMMENT'] = '         **  BDI KLIP PSF subtraction cube **'
    header['COMMENT'] = '         ************************************'
    header['COMMENT'] = 'Postagestamp cube of PSF subtraction using KLIP algorithm and BDI method'
    try:
        header['NAXIS1'] = str(shape[1])
        header['NAXIS2'] = str(shape[2])
        header['NAXIS3'] = str(shape[0])
    except:
        header['NAXIS1'] = str(shape[0])
        header['NAXIS2'] = str(shape[1])
    header['DATE'] = time.strftime("%m/%d/%Y")
    header['DATASET'] = dataset
    header['STAR'] = str(star)
    header['BASIS MODE CUTOFFS'] = str(K_klip)
    header['STAMPAXIS'] = str(stampshape)
    header['COMMENT'] = 'by Logan A Pearce'
    return header

############################# KLIP math #############################################################


def psf_subtract(scienceimage, ref_psfs, K_klip, covariances = None, use_basis = False,
                 basis = None, mean_image = None, return_basis = False, return_cov = False,
                 verbose = True):
    """Build an estimator for the psf of a BDI science target image from a cube of reference psfs 
       (use the ab_stack_shift function).  Follows steps of Soummer+ 2012 sec 2.2
       
    Written by Logan A. Pearce, 2020
    Heavily influenced by the lovely coding over at PyKLIP (https://pyklip.readthedocs.io/en/latest/)

    What is returned based on keywords:
    if (return_cov and return_basis) is True:
        return outputimage, Z, immean, cov, lamb, c
    elif return_basis:
        return outputimage, Z, immean
    elif return_cov:
        return outputimage, cov
    else:
        return outputimage
    
    Dependencies: numpy, image_registration, scipy

    Parameters:
    -----------
    scienceimage : 2d array
        science target image of shape mxn
    ref_psfs : 3d array
        reference psfs array of shape Nxmxn where N = number of reference psfs
    K_klip : int or arr
        Number of basis modes desired to use.  Can be integer or array of len b
    covariances : arr
        covariance matrix for ref psfs can be passed as an argument to avoid needing
        to calculate it
    basis : 3d arr
        psf basis modes (Z of Soummer 2.2.2) for star A or B can be passed as an argument to avoid needing
        to calculate it
    return_estimator : bool
        if set to True, return the estimated psf(s) used to subtract from the science target
        
    Returns:
    --------
    bxmxn arr
        psf subtracted image
    Nxp arr
        psf model basis modes (if return_basis = True)
    mxn arr
        mean image that accompanies basis set Z (if return_basis = True)
    NxN arr
        return the computed covariance matrix is return_cov = True. So it can be used in future
        calcs without having to recalculate
    """
    from scipy import ndimage
    import image_registration
    # Shift science image to line up with reference psfs (aligned during the cubing step):
    #dx,dy,edx,edy = image_registration.chi2_shift(np.sum(ref_psfs,axis=0), scienceimage, upsample_factor='auto')
    #scienceimage = ndimage.shift(scienceimage, [-dy,-dx], output=None, order=4, mode='constant', \
    #                          cval=0.0, prefilter=True)
    # Start KLIP math:
    from scipy.linalg import eigh
    
    # Soummer 2012 2.2.1:
    ### Prepare science target:
    shape=scienceimage.shape
    p = shape[0]*shape[1]
    
    # KL Basis modes:
    if use_basis is True:
        Z = basis
        immean = mean_image
    else:
        # Build basis modes:
        ### Prepare ref psfs:
        refshape=ref_psfs.shape
        N = refshape[0]
        if N < np.min(K_klip):
            if verbose:
                print("Oops! All of your requested basis modes are more than there are ref psfs.")
                print("Setting K_klip to number of ref psfs.  K_klip = ",N-1)
            K_klip = N-1
        if N < np.max(K_klip):
            if verbose:
                print("Oops! You've requested more basis modes than there are ref psfs.")
                print("Setting where K_klip > N-1 to number of ref psfs - 1.")
            K_klip[np.where(K_klip > N-1)] = N-1
            if verbose:
                print("K_klip = ",K_klip)
        K_klip = np.clip(K_klip, 0, N)
        R = np.reshape(ref_psfs,(N,p))
        # Make the mean image:
        immean = np.nanmean(R, axis=0)
        # Subtract mean image from each reference image:
        R_meansub = R - immean[None,:]#<- makes an empty first dimension to make
        # the vector math work out
    
        # Soummer 2.2.2:
        # compute covariance matrix of reference images:
        if covariances is None:
            cov = np.cov(R_meansub)
        else:
            cov = covariances
        # compute eigenvalues (lambda) and corresponding eigenvectors (c)
        # of covariance matrix.  Compute only the eigenvalues/vectors up to the
        # desired number of bases K_klip.
        lamb,c = eigh(cov, eigvals = (N-np.max(K_klip),N-1))
        # np.cov returns eigenvalues/vectors in increasing order, so
        # we need to reverse the order:
        index = np.flip(np.argsort(lamb))
        # sort corresponding eigenvalues:
        lamb = lamb[index]
        # check for any negative eigenvalues:
        check_nans = np.any(lamb <= 0)
        # sort eigenvectors in order of descending eigenvalues:
        c = c.T
        c = c[index]
        # np.cov normalizes the covariance matrix by N-1.  We have to correct
        # for that because it's not in the Soummer 2012 equation:
        lamb = lamb * (p-1)
        # Take the dot product of the reference image with corresponding eigenvector:
        Z = np.dot(R.T, c.T)
        # Multiply by 1/sqrt(eigenvalue):
        Z = Z * np.sqrt(1/lamb)
    
    
    # Reshape science target into 1xp array:
    T_reshape = np.reshape(scienceimage,(p))
    # Subtract mean from science image:
    T_meansub = T_reshape - immean[None,:]
    # Make K_klip number of copies of science image
    # to use fast vectorized math:
    T_meansub = np.tile(T_meansub, (np.max(K_klip), 1))
    

    # Soummer 2.2.4
    # Project science target onto KL Basis:
    projection_sci_onto_basis = np.dot(T_meansub,Z)
    # This produces a (K_klip,K_klip) sized array of identical
    # rows of the projected science target.  We only need one row:
    projection_sci_onto_basis = projection_sci_onto_basis[0]
    # This fancy math let's you use fewer modes to subtract:
    lower_triangular = np.tril(np.ones([np.max(K_klip), np.max(K_klip)]))
    projection_sci_onto_basis_tril = projection_sci_onto_basis * lower_triangular
    # Create the final psf estimator by multiplying by the basis modes:
    Ihat = np.dot(projection_sci_onto_basis_tril[K_klip-1,:], Z.T)
    
    # Soummer 2.2.5
    # Truncate the science image to the different number of requested modes to use:
    outputimage = T_meansub[:np.size(K_klip),:]
    outputimage1 = outputimage.copy()
    # Subtract estimated psf from science image:
    outputimage = outputimage - Ihat
    # Reshape to 
    outputimage = np.reshape(outputimage, (np.size(K_klip),*shape))

    if (return_cov and return_basis) is True:
        return outputimage, Z, immean, cov, lamb, c
    elif return_basis:
        return outputimage, Z, immean
    elif return_cov:
        return outputimage, cov
    else:
        return outputimage

###################### Perform KLIP on a science target to search for companions #######################
def SubtractCubes(acube, bcube, K_klip, k,\
                   # optional user inputs:\
                   a_covariances=None, b_covariances=None, a_estimator=None, b_estimator=None, \
                   # other parameters:\
                   verbose = True, interp = 'bicubic', rot_cval = 0.0
                ):
    """
    KLIP reduce cubes

    SubtractCubes performs the BDI KLIP math.  For the cube of Star A, it builds a PCA basis set
    using the cube of Star B, then steps through the stack of images on Star A, projecting each image
    onto the basis set to build a PSF model, and then subtracting the reconstructed model from the
    original image.  The subtracted image is derotated to North Up East Left using OpenCV2 rotation function.  
    When all images in the cube have been subtracted, it takes a sigma-clipped mean along the vertical 
    axis as the final reduced image.  Then repeats for Star B, using Star A as basis set.

    Written by Logan A. Pearce, 2020
    Dependencies: numpy, astropy.io.fits, OpenCV

    Parameters:
    -----------
    acube, bcube : 3d array
        cube of unnormalized postage stamp images of star A and star B, array of shape Nxmxn 
        where N = number of images in dataset (or truncated cube if boxsize prevents 
        using every image).  Output of ab_stack_shift function.
    K_klip : int or arr
        Number of basis modes desired to use.  Can be integer or array of len b
    k : Pandas array
        Pandas array made from the output of bditools.findstars_in_dataset.  
        Assumes column names are ['filename', 'xca','yca', 'xcb', 'ycb']. Same Pandas array used in
        the ab_stack_shift function.
    a_covariances, b_covariances : arr
        covariance matrix for ref psfs for star A or B can be passed as an argument to avoid needing
        to calculate it
    a_estimator, b_estimator : 2d arr
        psf model (Z of Soummer 2.2.2) for star A or B can be passed as an argument to avoid needing
        to calculate it
    write_to_disk : bool
        when set to True, output arrays will be written to fits files in the specified directory. 
        Default = True
    write_directory : str
        directory to write fits cubes of output files.
    outfilesuffix : str
        optional suffix to end of output cube filename.  For example, note that the boxsize used for a
        a reduction was 100 pixels by setting outfilesuffix = box100, filename will be dataset_klipcube_a_box100.fit
    headercomment : str
        optional comment to add to header in klip cube
    verbose : bool
        if True, print status updates
    interp : str
        Interpolation mode for OpenCV.  Either nearest, bilinear, bicubic, or lanczos4.
        Default = bicubic
    rot_cval : flt or nan
        fill value for rotated images
        
    Returns:
    --------
    a_final, b_final : 3d arr
        cube of psf subtracted and sigma-clipped mean combined postagestamp images
        of star A and B, with z = # of KLIP mode cutoff values.
        If write_to_disk = True the cubes are written to fits files with
        custom headers with filename of system and suffix "_klipcube_a.fits" 
        and "_klipcube_b.fits"

    """
    from cliotools.bditools import rotate_clio, psfsub_cube_header, psf_subtract, normalize_cubes
    from astropy.stats import sigma_clip

    # If single value of K_klip provided, make into array to prevent
    # later issues:
    if type(K_klip) == int:
        K_klip = np.array([K_klip])
    N = acube.shape[0]
    if N < np.max(K_klip):
        if verbose:
            print("Oops! You've requested more basis modes than there are ref psfs.")
            print("Setting where K_klip > N to number of ref psfs.")
        K_klip[np.where(K_klip > N)] = N-1
        if verbose:
            print("K_klip = ",K_klip)
    # measure the final product image dimensions by performing rotation
    # on first image in cube:
    imhdr = fits.getheader(k['filename'][0])
    a0_rot = rotate_clio(acube[0], imhdr)
    a_final = np.zeros([np.size(K_klip),a0_rot.shape[0],a0_rot.shape[1]])
    b_final = np.zeros(a_final.shape)
    
    # For each KL mode cutoff value:
    for j in range(np.size(K_klip)):
        ############### star A: ##################
        # for a single K_klip value and a single star A:
        if verbose:
            print('Subtracting using KL basis mode cutoff K_klip =',K_klip[j])
        # Use the first science image to create basis modes for psf model from star B:
        i = 0
        #Fa, Zb = psf_subtract(astamp[i], bstamp, K_klip[j], return_basis = True, verbose = verbose)
        Fa, Zb, immeanb = psf_subtract(acube[i], bcube, K_klip[j], return_basis = True, verbose = verbose)
        # get header and rotate image:
        imhdr = fits.getheader(k['filename'][i])
        Fa_rot = rotate_clio(Fa[0], imhdr, interp = interp, cval = rot_cval)
        # make a cube to store results:
        a = np.zeros([acube.shape[0],Fa_rot.shape[0],Fa_rot.shape[1]])
        # Place the psf subtracted image into the container cube:
        a[i,:,:] = Fa_rot
        # Use this basis to subtract all the remaining A images:
        for i in range(1,a.shape[0]):
            # subtract:
            F2a  = psf_subtract(acube[i], bcube, K_klip[j], use_basis = True, basis = Zb, mean_image = immeanb, verbose = verbose)
            # rotate:
            imhdr = fits.getheader(k['filename'][i])
            F2a_rot = rotate_clio(F2a[0], imhdr, interp = interp, cval = rot_cval)
            # store:
            a[i,:,:] = F2a_rot
        # final product is combination of subtracted and rotated images:
        #a_final[j,:,:] = np.median(a, axis = 0)
        a_final[j,:,:] = np.nanmean(sigma_clip(a, sigma = 3, axis = 0), axis = 0)
        
        ############### star B: ##################
        # Repeat for star B:
        i = 0
        #Fb, Za = psf_subtract(bstamp[i], astamp, K_klip[j], return_basis = True, verbose = verbose)
        Fb, Za, immeana = psf_subtract(bcube[i], acube, K_klip[j], return_basis = True, verbose = verbose)
        Fb_rot = rotate_clio(Fb[0], imhdr, interp = interp, cval = rot_cval)
        b = np.zeros(a.shape)
        b[i,:,:] = Fb_rot
        for i in range(1,b.shape[0]):
            # subtract:
            F2b  = psf_subtract(bcube[i], acube, K_klip[j], use_basis = True, basis = Za, mean_image = immeana, verbose = verbose)
            # rotate:
            imhdr = fits.getheader(k['filename'][i])
            F2b_rot = rotate_clio(F2b[0], imhdr, interp = interp, cval = rot_cval)
            # store:
            b[i,:,:] = F2b_rot
        #b_final[j,:,:] = np.median(b, axis = 0)
        b_final[j,:,:] = np.nanmean(sigma_clip(b, sigma = 3, axis = 0), axis = 0)

    if np.size(K_klip) == 1:
        return a_final[0], b_final[0]
    else:
        return a_final, b_final


##########################################################################
#  Functions for injecting synthetic planet signals                      #
from cliotools.bdi_signal_injection_tools import *


def contrast_curve(path, Star, sep = np.arange(1,7,1), C = np.arange(3,7,0.2), curves_file = [],
                   cmap = 'viridis', Ncontours_cmap=100, Ncontours_label = 5, 
                   fontsize=15, plotstyle = 'magrathea'):
    
    """After running DoSNR for a range of seps and contrasts, generate a map of SNR
        with contours at some intervals for contrast curves.  Uses scipy interpolate to
        expand sep/C to a square matrix and fill in intervals in what was tested.

    Parameters
    -----------
    path : str
        dataset folder
    Star : 'A' or 'B'
        which star had the signal injection
    sep, C : 1d arr
        arrays of separations and contrasts that were tested in the 
        DoSNR run
    curves_file : str
        string of the pickle file containing the output from the DoSNR run.
        If blank, assumes the file is named path/snrs[STAR].pkl
    Ncontours_cmap : int
        number of contours to draw and fill for the SNR map
    Ncontours_label : int
        number of contours to draw and label ontop of the map

    Returns
    -------
    figure
        snr plot
        
    """
    from scipy import interpolate
    if not len(curves_file):
        snrs = pickle.load( open( path+"snrs"+Star+".pkl", "rb" ) )
    else:
        snrs = pickle.load( open( path+curves_file, "rb" ) )
    resep = np.linspace(np.min(sep),np.max(sep),len(C))
    m = np.max([len(sep),len(C)])
    newSNRs = np.zeros((m,m))
    for i in range(m):
        f = interpolate.interp1d(sep, snrs[i])
        newSNR = f(resep)
        newSNRs[i] = newSNR

    try:
        plt.style.use(plotstyle)
    except:
        plt.style.use('default')
    fig = plt.figure()
    contours = plt.contour(resep,C,newSNRs, Ncontours_label, colors='red',linestyles=(':',))
    plt.clabel(contours, inline=True, fontsize=fontsize,fmt='%1.0f')
    contour = plt.contour(resep,C,newSNRs,levels = [5.0],
                     colors=('r',),linestyles=('-',),linewidths=(2,))
    plt.clabel(contour, inline=True, fontsize=fontsize,fmt='%1.0f')
    plt.contourf(resep,C,newSNRs,Ncontours_cmap,cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel(r'Sep [$\frac{\lambda}{D}$]')
    plt.ylabel('contrast [mags]')
    plt.title(path.split('/')[0]+' '+Star)
    return fig

