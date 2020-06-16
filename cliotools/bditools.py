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
                    print("Failed to find all stars.")
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
            overwrite an old one.  Defautl = False.
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

def rotate_clio(image, imhdr, **kwargs):
    """Rotate CLIO image to north up east left.
       Written by Logan A. Pearce, 2020
       
       Dependencies: scipy

       Parameters:
       -----------
       image : 2d array
           2d image array
       imhdr : fits header object
           header for image to be rotated
       kwargs : for scipy.ndimage
           
       Returns:
       --------
       imrot : 2d arr 
           rotated image with north up east left
    """
    from scipy import ndimage
    NORTH_CLIO = -1.80
    derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
    imrot = ndimage.rotate(image, derot, **kwargs)
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
                    pass
                elif np.isnan(xb):
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
        coadd_count = image.shape[0]
        count = 0
        for i in range(len(k)):
            try:
                a = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                                                                   np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
                astamp[coadd_count:coadd_count+a.shape[0],:,:] = a
                b = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                                                                   np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
                bstamp[coadd_count:coadd_count+b.shape[0],:,:] = b
                   
                for j in range(coadd_count,coadd_count+a.shape[0]):
                    xa,ya = daostarfinder(a[j], boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                    xb,yb = daostarfinder(b[j], boxsize, boxsize, boxsize = boxsize, fwhm = fwhm)
                    if np.isnan(xa):
                        pass
                    elif np.isnan(xb):
                        pass
                    else:
                        dx,dy = xa-center[0],ya-center[0]
                        astamp[j,:,:] = ndimage.shift(astamp[j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                                  prefilter=True)
                        dx,dy = xb-center[0],yb-center[0]
                        bstamp[j,:,:] = ndimage.shift(bstamp[j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                                  prefilter=True)
                coadd_count += a.shape[0]
            except:
                if verbose:
                    print('ab_stack_shift: Oops! the box is too big and one star is too close to an edge. I cut it off at i=',count)
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
                   inner_mask_core = True, inner_mask_format = 'lambda/D', inner_mask_radius = 1.,\
                   # masking outer annulus:\
                   outer_mask_annulus = False, outer_mask_radius = None
                ):
    ''' Assemble cubes of images and prepare them for KLIP reduction by: centering/subpixel-aligning images along
        vertical axis, normalizing images by dividing by sum of pixels in image, and masking the core of the central star.
        Written by Logan A. Pearce, 2020
       Dependencies: numpy, astropy.io.fits

       Parameters:
       -----------
       k : Pandas array
           Pandas array made from the output of bditools.findstars_in_dataset.  
           Assumes column names are ['filename', 'xca','yca', 'xcb', 'ycb']
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
           if True, set all pixels outside the specified radius to zero.  Default = False
       outer_radius : flt
           Set all pixels outside this radius to 0 if mask_outer_aunnulus set to True.
           
       Returns:
       --------
       acube, bcube : 3d arr
           cube of images of Star A and B ready to be put into KLIP reduction pipeline.
    '''
    from cliotools.bditools import ab_stack_shift
    astack, bstack = ab_stack_shift(k, boxsize = boxsize,  path_prefix=path_prefix, verbose = verbose)
    center = (0.5*((astack[0].shape[0])-1),0.5*((astack[0].shape[1])-1))
    if normalize:
        from cliotools.bditools import normalize_cubes
        # Normalize cubes:
        acube,bcube = normalize_cubes(astack,bstack, normalizebymask = normalizebymask,  radius = normalize_radius)
    elif normalize == False:
        acube,bcube = astack.copy(),bstack.copy()
    if inner_mask_core:
        from cliotools.bditools import mask_star_core
        # Set all pixels within a radius of the center to zero:
        acube2,bcube2 = acube.copy(),bcube.copy()
        acube,bcube = mask_star_core(acube2, bcube2, inner_mask_radius,center[0], center[1], mask_format = inner_mask_format)
    if outer_mask_annulus:
        from cliotools.bditools import mask_outer
        if outer_mask_radius == None:
            raise ValueError('Outer radius must be specified if mask_outer_annulus == True')
        acube2,bcube2 = acube.copy(),bcube.copy()
        acube,bcube = mask_outer(acube2, bcube2, outer_mask_radius, center[0], center[1])
    return acube,bcube



def normalize_cubes(astack,bstack, normalizebymask = False,  radius = []):
    '''
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
       acube, bcube : 3d arr
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

def mask_star_core(astack, bstack, radius, xc, yc, mask_format = 'lambda/D'):
    ''' Set the core of the star flux to zero pixel value
       Parameters:
       ----------
       astack, bstack : 3d array
           cube of unnormalized images
       radius : flt
           if normalizebymask = True, set the radius of the aperture mask.  Must be in units of lambda/D.
       xc, yc : flt
           x/y of star center
       mask_format : str
           request inner core mask in lamdba/D or pixels
           
       Returns:
       --------
       anans, bnans : 3d arr
           cube of normalized images
    '''
    from cliotools.cliotools import lod_to_pixels
    shape = astack[0].shape
    xx,yy = np.meshgrid(np.arange(shape[0])-xc,np.arange(shape[1])-yc)
    # compute radius of each pixel from center:
    r=np.hypot(xx,yy)
    if mask_format == 'lambda/D':
        radius = lod_to_pixels(radius, 3.9)
    elif mask_format == 'pixels':
        pass
    else:
        raise ValueError('please specify mask_format = lambda/D or pixels')
    anans,bnans = astack.copy(),bstack.copy()
    for i in range(astack.shape[0]):
        anans[i][np.where(r<radius)] = 0
        bnans[i][np.where(r<radius)] = 0
    return anans,bnans

def mask_outer(astack, bstack, radius, xc, yc, radius_format = 'pixels'):
    ''' Set the core of the star flux to zero pixel value
       Parameters:
       ----------
       astack, bstack : 3d array
           cube of unnormalized images
       radius : flt
           if normalizebymask = True, set the radius of the aperture mask.  Must be in units of lambda/D.
       xc, yc : flt
           x/y of star center
       radius_format : str
           How to handle requested outer annulus mask radius. Either pixels or lambda/D.
           
       Returns:
       --------
       anans, bnans : 3d arr
           cube of normalized images
    '''
    from cliotools.cliotools import lod_to_pixels
    shape = astack[0].shape
    xx,yy = np.meshgrid(np.arange(shape[0])-xc,np.arange(shape[1])-yc)
    # compute radius of each pixel from center:
    r=np.hypot(xx,yy)
    if radius_format == 'lambda/D':
        radius = lod_to_pixels(radius, 3.9)
    anans,bnans = astack.copy(),bstack.copy()
    for i in range(astack.shape[0]):
        anans[i][np.where(r>radius)] = 0
        bnans[i][np.where(r>radius)] = 0
    return anans,bnans
                
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
        header['NAXIS3'] = str(1)
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
        if (return_cov and return_basis) is True:
            return outputimage, Z, immean, cov, lamb, c
        elif return_basis:
            return outputimage, Z, immean
        elif return_cov:
            return outputimage, cov
        else:
            return outputimage

       outputimage : bxmxn arr
           psf subtracted image
       Z : Nxp arr
           psf model basis modes (if return_basis = True)
       immean : mxn arr
           mean image that accompanies basis set Z (if return_basis = True)
       cov : NxN arr
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
                   # writing parameters:\
                   write_to_disk = True, write_directory = '.', outfilesuffix = '',headercomment = None, \
                   # other parameters:\
                   reshape = False, verbose = True, cval = 0.0, order = 1
                ):
    """For each image in the astamp/bstamp cubes, PSF subtract image, rotate to
       north up/east left; then median combine all images for star A and star B into
       single reduced image.  Do this for an array of KLIP mode cutoff values.  Writes
       reduced images to disk in fits cube with z axis = # of KLIP mode cutoff values.

       Written by Logan A. Pearce, 2020
       Dependencies: numpy, astropy.io.fits

       Parameters:
       -----------
       astack, bstack : 3d array
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
       reshape : bool
           argument passed to ndimage.rotate.  If True, final output image size will change to fit entire rotated
           image, but this causes problems when different images in a set are rotated by different amounts, causing the 
           final arrays to be different sizes.  Should be set to False, which makes rotated image be the same dimensions as
           input cubes.
       verbose : bool
           if True, print status updates
       cval : flt or nan
           fill value for rotated images
       order : int [0,5]
           order of interpolation when rotating, fed into ndimage.rotate
           
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
    a0_rot = rotate_clio(acube[0], imhdr, order = 4, reshape = reshape, mode='constant', cval=cval)
    a_final = np.zeros([np.size(K_klip),a0_rot.shape[0],a0_rot.shape[1]])
    b_final = np.zeros(a_final.shape)
    
    # If single value of K_klip provided, make into array to prevent
    # later issues:
    if type(K_klip) == int:
        K_klip = np.array([K_klip])
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
        Fa_rot = rotate_clio(Fa[0], imhdr, order = order, reshape = reshape, mode='constant', cval=cval)
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
            F2a_rot = rotate_clio(F2a[0], imhdr, order = order, reshape = reshape, mode='constant', cval=cval)
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
        Fb_rot = rotate_clio(Fb[0], imhdr, order = order, reshape = reshape, mode='constant', cval=cval)
        b = np.zeros(a.shape)
        b[i,:,:] = Fb_rot
        for i in range(1,b.shape[0]):
            # subtract:
            F2b  = psf_subtract(bcube[i], acube, K_klip[j], use_basis = True, basis = Za, mean_image = immeana, verbose = verbose)
            # rotate:
            imhdr = fits.getheader(k['filename'][i])
            F2b_rot = rotate_clio(F2b[0], imhdr, order = order, reshape =reshape, mode='constant', cval=cval)
            # store:
            b[i,:,:] = F2b_rot
        #b_final[j,:,:] = np.median(b, axis = 0)
        b_final[j,:,:] = np.nanmean(sigma_clip(b, sigma = 3, axis = 0), axis = 0)

    if write_to_disk is True:
        if verbose:
            print('Writing finished cubes to file... done!')
        newhdr = psfsub_cube_header(k['filename'][0].split('/')[0], K_klip, 'A', a_final.shape, acube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(k['filename'][0].split('_')[0]+'_klipcube_a'+outfilesuffix+'.fit',a_final,newhdr,overwrite=True)
        newhdr = psfsub_cube_header(k['filename'][0].split('/')[0], K_klip, 'B', b_final.shape, bcube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(k['filename'][0].split('_')[0]+'_klipcube_b'+outfilesuffix+'.fit',b_final,newhdr,overwrite=True)
    return a_final, b_final

'''
def do_bdi(k, K_klip, **kwargs):
    """Wrapper function for BDI psf subtraction functions
    """
    from cliotools.bditools import ab_stack_shift, subtract_cubes
    # create postage stamp cubes:
    astamp, bstamp = ab_stack_shift(k, **kwargs)
    # psf subtract cubes
    a_final, b_final = subtract_cubes(astamp, bstamp, K_klip, k, **kwargs)'''

##########################################################################
#  Functions for injecting synthetic planet signals                      #

def mag(image, x, y, radius = 9., returnflux = False, returntable = False):
    ''' Compute instrument magnitudes of one object.  Defaults are set to CLIO 3.9um optimal.
        Parameters:
        -----------
        image : 2d array
            science image
        x,y : flt
            x and y pixel location of center of star
        radius : flt
            pixel radius for aperture.  Default = 9, approx location of 1st null in
            CLIO 3.9um 
        r_in, r_out : flt
            inside and outside radius for background annulus.  Default = 10,12
        returnflux : bool
            if true, return the instrument mag and the raw flux value.
        returntable : bool
            if true, return the entire photometry table.
        Returns:
        --------
        mag : flt
            instrument magnitudes of source
        snr : flt
            signal to noise ratio
    '''
    from photutils import CircularAperture, CircularAnnulus, aperture_photometry
    # Position of star:
    positions = [(x,y)]
    # Get sum of all pixel values within radius of center:
    aperture = CircularAperture(positions, r=radius)
    # Do photometry on star:
    phot_table = aperture_photometry(image, aperture)
    m =(-2.5)*np.log10(phot_table['aperture_sum'][0])
    if returnflux:
        return m, phot_table['aperture_sum'][0]
    if returntable:
        phot_table['Mag'] = m
        return m, phot_table
    return m

def contrast(image1,image2,pos1,pos2,**kwargs):
    ''' Return contrast of component 2 relative to 1 in magnitudes
        Parameters:
        ------------
        image1 : 2d array
            science image
        image2 : 2d array
            image of other object
        pos1 : arr
            x and y pixel location of center of star1 in order [x1,y1]
        pos2 : arr
            x and y pixel location of center of star2 in order [x2,y2]
        kwargs : 
            args to pass to mag function
        Returns:
        --------
        contrast : flt
            contrast in magnitudes of B component relative to A component
        
    '''
    from cliotools.bditools import mag
    mag1 = mag(image1,pos1[0],pos1[1], **kwargs)
    mag2 = mag(image2,pos2[0],pos2[1], **kwargs)
    return mag2 - mag1

def makeplanet(template, C, TC):
    ''' Make a simulated planet psf with desired contrast using template psf
        Parameters:
        -----------
        template : 2d image
            sample PSF for making simulated planet
        C : flt
            desired contrast in magnitudes
        TC : flt
            known contrast of template psf relative to science target
        Returns:
        --------
        Pflux : 2d image
            scaled simulated planet psf with desired contrast to science target
    '''
    # Amount of magnitudes to scale template by to achieve desired
    # contrast with science target:
    D = C - TC
    # Convert to flux:
    scalefactor = 10**(-D/2.5)
    # Scale template pixel values:
    Pflux = template*scalefactor
    return Pflux

def injectplanet(image, imhdr, template, sep, pa, contrast, TC, xc, yc, 
                 sepformat = 'lambda/D', 
                 pixscale = 15.9,
                 wavelength = 'none',
                 box = 70
                ):
    ''' Using a template psf, place a fake planet at the desired sep, pa, and
        contrast from the central object.  PA is measured relative to true north
        (rather than up in image)
        Parameters:
        -----------
        image : 2d array
            science image
        imhdr : fit header
            science image header
        template : 2d array
            template psf with known contrast to central object
        sep : flt or fltarr
            separation of planet placement in either arcsec, mas, pixels, or lambda/D
        pa : flt or fltarr
            position angle of planet relative to north in DEG
        contrast : flt or fltarr
            desired contrast of planet with central object
        TC : flt
            template contrast, known contrast of template psf relative to science target
        xc, yc : flt
            x,y pixel position of central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'pixels'
        pixscale : flt
            pixelscale in mas/pixel.  Default = 15.9 mas/pix, pixscale for CLIO narrow camera
        wavelength : flt
            central wavelength of filter, needed if sepformat = 'lambda/D'
        box : int
            size of template box.  Template will be box*2 x box*2
        
        Returns:
        --------
        synth : 2d array
            image with fake planet with desired parameters. 
    '''
    from cliotools.bditools import makeplanet
    # sep input into pixels
    if sepformat == 'arcsec':
        pixscale = pixscale/1000 # convert to arcsec/pix
        sep = sep / pixscale
    if sepformat == 'mas':
        sep = sep / pixscale
    if sepformat == 'lambda/D':
        from cliotools.cliotools import lod_to_pixels
        if wavelength == 'none':
            raise ValueError('wavelength input needed if sepformat = lambda/D')
        sep = lod_to_pixels(sep, wavelength)
    # pa input - rotate from angle relative to north to angle relative to image up:
    #    do the opposite of what you do to derotate images
    NORTH_CLIO = -1.80
    derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
    pa = (pa + derot)
        
    # Get cartesian location of planet:
    xx = sep*np.sin(np.radians((pa)))
    yy = sep*np.cos(np.radians((pa)))
    xs = np.int_(np.floor(xc-xx))
    ys = np.int_(np.floor(yc+yy))
    # Make planet from template at desired contrast
    Planet = makeplanet(template, contrast, TC)
    # Make copy of image:
    synth = image.copy()
    # Get shape of template:
    boxy, boxx = np.int_(Planet.shape[0]/2),np.int_(Planet.shape[1]/2)
    x,y = xs,ys
    ymin, ymax = y-boxy, y+boxy
    xmin, xmax = x-boxx, x+boxx
    # Correct for sources near image edge:
    delta = 0
    if ymin < 0:
        delta = ymin
        ymin = 0
        Planet = Planet[(0-delta):,:]
    if ymax > image.shape[0]:
        delta = ymax - image.shape[0]
        ymax = image.shape[0]
        Planet = Planet[:(2*boxy-delta) , :]
    if xmin < 0:
        delta = xmin
        xmin = 0
        Planet = Planet[:,(0-delta):]
    if xmax > image.shape[1]:
        delta = xmax - image.shape[1]
        xmax = image.shape[1]
        Planet = Planet[:,:(2*boxx-delta)]
    # account for integer pixel positions:
    if synth[ymin:ymax,xmin:xmax].shape != Planet.shape:
        try:
            synth[ymin:ymax+1,xmin:xmax] = synth[ymin:ymax+1,xmin:xmax] + (Planet)
        except:
            synth[ymin:ymax,xmin:xmax+1] = synth[ymin:ymax,xmin:xmax+1] + (Planet)
    else:
        synth[ymin:ymax,xmin:xmax] = synth[ymin:ymax,xmin:xmax] + (Planet)
    return synth

def injectplanets(image, imhdr, template, sep, pa, contrast, TC, xc, yc, **kwargs):
    ''' Wrapper for injectplanet() that allows for multiple fake planets in one image.
        Parameters are same as injectplanet() except sep, pa, and contrast must all be
        arrays of the same length.  **kwargs are passed to injectplanet().
    '''
    from cliotools.bditools import injectplanet
    synth = image.copy()
    try:
        for i in range(len(sep)):
            synth1 = injectplanet(synth, imhdr, template, sep[i], pa[i], contrast[i], TC, xc, yc, 
                                      **kwargs)
            synth = synth1.copy()
    except:
        synth = injectplanet(synth, imhdr, template, sep, pa, contrast, TC, xc, yc, 
                                      **kwargs)
    return synth



########### Perform KLIP on injected planet signals ##########################

def DoInjection(path, Star, K_klip, sep, pa, C,
                sepformat = 'lambda/D', box = 100, 
                template = [], TC = None, use_same = True, verbose = True,
                # Params for handling KLIP bases:
                returnZ = False, Z = [], immean = [],
                # Params for user supplied image cubes:
                returnstamps = False, sciencecube = [], refcube = [], templatecube = [],
                # Params for preparing images for KLIP:
                normalize = True, normalizebymask = False,  normalize_radius = [],
                mask_core = True, mask_radius = 1.
                    ):
    ''' Inject fake planet signal into images of Star 1 using Star 2 as template (or user
        supplied template) in all images in a dataset, and perform KLIP reduction and psf subtraction
        Parameters:
        -----------
        path : str
            dataset folder
        Star : 'A' or 'B'
            star to put the fake signal around
        K_klip : int
            number of KLIP modes to use in psf subtraction
        sep : flt or fltarr
            separation of planet placement in either arcsec, mas, pixels, or lambda/D
        pa : flt or fltarr
            position angle of planet relative to north in DEG
        C : flt or fltarr
            desired contrast of planet with central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'pixels'
        box : int
            size of box around template star
        template : 2d array
            option to supply a template psf for creating fake signal with known contrast to science target
        TC : float
            known contrast of template psf, required only if external supplied template is used
        use_same : bool
            if True, use the star's own psf as template for injecting signal.  If False, use the opposite star as template.
            Same star seems to give higher snr thus is prefered.  Default = True
        verbose : bool
            if true allow functions to print statements
        returnZ : bool
            if true, return the computed basis set from reference star
        Z : 2d array
            option to supply a basis set and skip computing step.  Useful if performing
            multiple operations on the same dataset.  You can compute Z only once and return it,
            then supply it for further calculations
        immean : 2d arr
            mean image for basis set used in Z
        returnstamps : bool
            if True, return image stamps.  Useful to toggle on and off to avoid having to stack and shift
            images base every time you test a new injected planet, which takes time.  Default = False
        sciencecub, refcube, templatecube : 3d arr
            user input the base images to use in injection for science images, psf templates, and KLIP reference basis sets.  
            Typically will be the output of a previous DoInjection run where returnstamps=True
        normalize : bool
           if True, normalize each image science and reference image integrated flux by dividing by each image
           by the sum of all pixels in image.  If False do not normalize images. Default = True
        normalizebymask : bool
           if True, normalize using only the pixels in a specified radius to normalize.  Default = False
        radius : bool
           if normalizebymask = True, set the radius of the aperture mask.  Must be in units of lambda/D.
        mask_core : bool
           if True, set all pixels within mask_radius to value of 0 in all images.  Default = True
        mask_radius : flt
           Set all pixels within this radius to 0 if mask_core set to True.  Must be units of lambda/D.
        
        Returns
        -------
        kliped : 2d arr
            KLIP reduced image with injected planet signal
        astamp,bstamp : 3d arr
            If returnstamps = True: registered stack of non-normalized images with no injected signal.  
            Useful to return if performing multiple injection/recovery tests, allows you to collect and
            register the images only once.
    '''
    from cliotools.bditools import psf_subtract
    k = pd.read_csv(path+'CleanList', comment='#')
    
    # First, if cubes aren't provided, build up the science cube (science target),
    # template cube (for creating synthetic planet signal)
    # and reference cube (for generating KLIP reduction basis set)
    # from the provided directory:
    if not len(sciencecube):
        from cliotools.bditools import PrepareCubes
        # Collect and register the science images, but do no
        # normalize or mask yet, that must be done after signal injection:
        astamp, bstamp = PrepareCubes(k, boxsize = box, verbose = verbose, \
                                      normalize=False, mask_core=False)
        # Assign astamp/bstamp to PSF template and science cubes:
        if use_same:
            if Star == 'A':
                sciencecube = astamp.copy()
                templatecube = astamp.copy()
            elif Star == 'B':
                sciencecube = bstamp.copy()
                templatecube = bstamp.copy()
        else:
            if Star == 'A':
                sciencecube = astamp.copy()
                templatecube = bstamp.copy()
            elif Star == 'B':
                sciencecube = bstamp.copy()
                templatecube = astamp.copy()
        # Assign astamp/bstamp to KLIP reduction reference cubes, opposite star from
        # science star:
        if Star == 'A':
            refcube = bstamp.copy()
        elif Star == 'B':
            refcube = astamp.copy()
                
    # Inject planet signal into science target star
    synthcube = np.zeros(np.shape(sciencecube))
    from cliotools.bditools import injectplanets
    if not len(template):
        # If template PSF is not provided by user (this is most common):
        for i in range(sciencecube.shape[0]):
            from cliotools.bditools import contrast
            # Get template constrast of refcube to sciencecube
            TC = contrast(sciencecube[i],templatecube[i],[box+0.5,box+0.5],[box+0.5,box+0.5])
            # image header must be provided to 
            # accomodate rotation from north up reference got PA to image reference:
            imhdr = fits.getheader(k['filename'][i]) 
            # Inject the desired signal into the science cube:
            synth = injectplanets(sciencecube[i], imhdr, templatecube[i], sep, pa, C, TC, box, box, 
                                          sepformat = sepformat, wavelength = 3.9, box = box)
            # place signal-injected image into stack of images:
            synthcube[i,:,:] = synth
    else:
        # If external template is provided: (this might happen if other star is saturated, etc)
        if not TC:
            # Known contrast of template to science star must be provided
            raise ValueError('template contrast needed')
        # inject signal:
        for i in range(sciencecube.shape[0]):
            imhdr = fits.getheader(k['filename'][i])
            synth = injectplanets(sciencecube[i], imhdr, template, sep, pa, C, TC, box, box, 
                                          sepformat = sepformat, wavelength = 3.9, box = box)
            synthcube[i,:,:] = synth
    
    
    # Normalize and mask both science target with injected signal, and reference star basis set:    
    from cliotools.bditools import normalize_cubes, mask_star_core
    if normalize:
        synthcube,refcube = normalize_cubes(synthcube, refcube, normalizebymask = False)
    if mask_core:
        synthcube,refcube = mask_star_core(synthcube, refcube, mask_radius, synthcube[0].shape[0]/2, \
                                       synthcube[0].shape[1]/2)
            
    if not len(Z):
        # Use psf subtraction function to build a basis from opposite star, and
        # throw away the subtracted image (F) because all we need is the basis set.
        F, Z, immean = psf_subtract(sciencecube[0], refcube, K_klip, return_basis = True, return_cov = False)
        # If you tried to use more basis modes than there are reference psfs, the size of Z will
        # be the max number of available modes.  So we need to reset our ask to the max number 
        # of available modes:
        if Z.shape[1] < K_klip:
            K_klip = Z.shape[1]
    
    # Perform klip reduction:
    from cliotools.bditools import rotate_clio
    klipcube = np.zeros(np.shape(synthcube))
    for i in range(synthcube.shape[0]):
        # Use the basis from before to subtract each synthetic image:
        # Supplying the external basis makes use of refcube to align the science image with
        # the basis we made from refcube only, but skips making a new basis set out of refcube.
        if not len(immean):
            raise ValueError('Mean image needed if using extenally computed basis set')
        F = psf_subtract(synthcube[i], refcube, K_klip, use_basis = True, basis = Z, 
                             mean_image = immean, verbose = True)
        # rotate:
        imhdr = fits.getheader(k['filename'][i])
        Frot = rotate_clio(F[0], imhdr, order = 4, reshape = False, mode='constant', cval=np.nan)
        klipcube[i,:,:] = Frot
    from astropy.stats import sigma_clip
    # Take sigma-clipped mean:
    kliped = np.mean(sigma_clip(klipcube, sigma = 3, axis = 0), axis = 0)
    
    if returnstamps and returnZ:
        return kliped, Z, immean, astamp, bstamp
    if returnstamps:
        return kliped, astamp, bstamp
    if returnZ:
        return kliped, Z
    return kliped

def getsnr(image, sep, pa, xc, yc, wavelength = 3.9):
    ''' Get SNR of injected planet signal using method and Student's T-test
        statistics described in Mawet 2014
        Parameters:
        -----------
        image : 2d arr
            KLIP reduced image with injected planet signal at (sep,pa)
        sep : flt
            separation of injected signal in L/D units
        pa : flt
            position angle of inject signal relative to north in degrees
        xc, yc : flt or int
            (x,y) pixel location of center of star
        wavelength : flt
            central wavelength in microns of image filter. Used for converting 
            from L/D units to pixels.  Default = 3.9
            
    '''
    from cliotools.cliotools import lod_to_pixels
    from photutils import CircularAperture, aperture_photometry
    radius = lod_to_pixels(1., wavelength)/2
    lod = 2*radius
    # convert sep in L/D to pixels:
    seppix = lod_to_pixels(sep, wavelength)
    # Number of 1L/D apertures that can fit on the circumference at separation:
    Napers = np.floor(sep*2*np.pi)
    # Change in angle from center of one aper to the next:
    dTheta = 360/Napers
    # Create array around circumference, excluding the ones immediately before and after
    # where the planet is:
    pas = np.arange(pa+2*dTheta,pa+360-dTheta,dTheta)%360
    # create emptry container to store results:
    noisesums = np.zeros(len(pas))
    # for each noise aperture:
    for i in range(len(pas)):
        # lay down a photometric aperture at that point:
        xx = seppix*np.sin(np.radians((pas[i])))
        yy = seppix*np.cos(np.radians((pas[i])))
        xp,yp = xc-xx,yc+yy
        aperture = CircularAperture([xp,yp], r=radius)
        # sum pixels in aperture:
        phot = aperture_photometry(image, aperture)
        # add to noise container:
        noisesums[i] = phot['aperture_sum'][0]
    # the noise value is the std dev of pixel sums in each
    # noise aperture:
    noise = np.std(noisesums)
    # Compute signal of injected planet in signal aperture:
    xx = seppix*np.sin(np.radians((pa)))
    yy = seppix*np.cos(np.radians((pa)))
    xp,yp = xc-xx,yc+yy
    # Lay down aperture at planet location:
    aperture = CircularAperture([xp,yp], r=radius)
    # compute pixel sum in that location:
    phot = aperture_photometry(image, aperture)
    signal = phot['aperture_sum'][0]
    # compute mean background:
    bkgd = np.mean(noisesums)
    # Eqn 9 in Mawet 2014:
    signal = signal - bkgd
    snr = signal / ( noise * np.sqrt(1+ (1/np.size(pas))) )
    return snr

def DoSNR(path, Star, K_klip, sep, C, sepformat = 'lambda/D', box = 100, returnsnrs = False, writeklip = True, verbose = False,  update_prog = True, use_same = True,
         **kwargs):
    ''' For a single value of separation and contrast of an injected planet signal around Star A/B, 
        compute the SNR using the method in Mawet 2014.  kwargs are fed to DoInjection

        Parameters:
        -----------
        path : str
            dataset folder
        Star : 'A' or 'B'
            star to put the fake signal around
        K_klip : int
            number of KLIP modes to use in psf subtraction
        sep : flt
            separation of planet placement in either arcsec, mas, pixels, or lambda/D
        C : flt
            desired contrast of planet with central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'lambda/D'
        box : int
            size of box of size "2box x 2box" for image stamps
        returnsnrs : bool
            if True, return both the mean SNR and the array of SNRs for each aperture
            in the ring.
        writeklip : bool
            if True, write a fits image of the KLIP-reduced image of the first injected
            planet signal into the directory specified in "path".  Default = True
        update_prog : bool
            if True, update progress bar on performing SNR calc on each aperture in ring.  If
            False, do not show progess bar.  Default - True
            
        Returns:
        --------
        snr : flt
            mean SNR for all apertures of size L/D in ring at r=sep
        snrs : flt arr
            if returnsnrs = True, return array of SNR in each aperture in the ring.  snrs[0]
            is for aperture at pa=270 deg, and indices proceed counterclockwise from there.
    '''
    from cliotools.pcaskysub import update_progress
    from cliotools.bditools import DoInjection, getsnr
    # Define starting point pa:
    pa = 270.
    # define center of box (location of star):
    xc,yc = box-1,box-1
    # Number of 1L/D apertures that can fit on the circumference at separation:
    Napers = np.floor(sep*2*np.pi)
    # Change in angle from one aper to the next:
    dTheta = 360/Napers
    # Create array around circumference, excluding the ones immediately before and after
    # where the planet is:
    pas = np.arange(pa+2*dTheta,pa+360-dTheta,dTheta)%360
    # create empty container to store results:
    snrs = np.zeros(len(pas))
    
    # Do initial run on 1st pa location to create stamps and KLIP basis set:
    kliped, Z, immean, astamp, bstamp = DoInjection(path, Star, K_klip, sep, pa, C, 
                    sepformat = sepformat, box = box, 
                    verbose = verbose, returnZ = True, returnstamps = True, use_same = use_same, **kwargs)
    if writeklip:
        from astropy.io import fits
        name = path+'/injectedsignal_star'+Star+'_sep'+'{:.0f}'.format(sep)+'_C'+'{:.1f}'.format(C)+'.fit'
        fits.writeto(name,kliped.data,overwrite=True)
    # compute SNR for first injected location:
    s = getsnr(kliped, sep, pa, xc, yc)
    # place in container
    snrs[0] = s
    if update_prog:
        # Update progress:
        update_progress(0+1,len(pas))
    
    if Star == 'A':
        sciencecube = astamp.copy()
        refcube = bstamp.copy()
        if use_same:
            templatecube = astamp.copy()
        else:
            templatecube = bstamp.copy()
            
    elif Star == 'B':
        sciencecube = bstamp.copy()
        refcube = astamp.copy()
        if use_same:
            templatecube = bstamp.copy()
        else:
            templatecube = astamp.copy()
    
    # Repeat for all apertures in the ring:
    for i in range(1,len(pas)):
        # Speed up by skipping calculation of basis and making cubes
        kliped = DoInjection(path, Star, K_klip, sep, pas[i], C, 
                             sepformat = sepformat, box = box, 
                             verbose = False, returnZ = False, 
                             # Use previously computed basis set and mean imageL
                             Z = Z, immean = immean,
                             # Use previously made data cubes:
                             sciencecube = sciencecube, refcube = refcube, templatecube = templatecube, **kwargs)
        # compute and store SNR:
        snr = getsnr(kliped, sep, pas[i], xc, yc)
        snrs[i] = snr
        if update_prog:
            update_progress(i+1,len(pas))
        
    if returnsnrs:
        return np.mean(snrs), snrs
        
    return np.mean(snrs)


def contrast_curve(path, Star, sep = np.arange(1,7,1), C = np.arange(3,7,0.2), curves_file = [],
                   cmap = 'viridis', Ncontours_cmap=100, Ncontours_label = 5, 
                   fontsize=15, plotstyle = 'magrathea'):
    
    """ After running DoSNR for a range of seps and contrasts, generate a map of SNR
        with contours at some intervals for contrast curves.  Uses scipy interpolate to
        expand sep/C to a square matrix and fill in intervals in what was tested.

        Parameters:
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
        fig : matplotlib figure object
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
    

##########################################################################
#              DEPRECATED                                                #

def DoInjection_deprecated(path, Star, sep, pa, C, sepformat = None, box=100, template = None, TC = None, outsuffix = '', returng = False, showprog = True):
    ''' Wrapper function to inject fake planet signal into all images in dataset
        Parameters:
        -----------
        path : str
            dataset folder
        Star : 'A' or 'B'
            star to put the fake signal around
        sep : flt or fltarr
            separation of planet placement in either arcsec, mas, pixels, or lambda/D
        pa : flt or fltarr
            position angle of planet relative to north in DEG
        C : flt or fltarr
            desired contrast of planet with central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'pixels'
        box : int
            size of box around template star
        template : 2d array
            option to supply a template psf for creating fake signal with known contrast to science target
        TC : float
            known contrast of template psf, required only if external supplied template is used
        returng : bool
            if true, return a pandas dataframe of the filename of each file with injected planet and star
            positions in those files
        showprog : bool
            if true, display a progess bar for inserting a planet in all images, else display nothing
    '''
    from cliotools.cliotools import make_imagestamp
    # pandas thorws a warning if you replace an entry on a copy of a dataframe, but it's fine so:
    import warnings
    warnings.filterwarnings('ignore')
    
    if not sepformat:
        raise ValueError('need sepformat')
    k = pd.read_csv(path+'CleanList', comment='#')
    g = k.copy()
    for i in range(len(k)):
        # open image:
        image = fits.getdata(k['filename'][i])
        imhdr = fits.getheader(k['filename'][i])
        # Get template contrast:
        if Star == 'A':
            if not template:
                TC = contrast(image, [k['xca'][i],k['yca'][i],k['xcb'][i],k['ycb'][i]])
            # Make template from Star B:
                imstamp = make_imagestamp(image, k['xcb'][i],k['ycb'][i], boxsizex = box, boxsizey = box)[0]
            else:
                imstamp = template
            synth = injectplanets(image, imhdr, imstamp, sep, pa, C, TC, k['xca'][i], k['yca'][i], 
                                  sepformat = sepformat, wavelength = 3.9, box = box)
        elif Star == 'B':
            if not template:
                TC = contrast(image, [k['xcb'][i],k['ycb'][i],k['xca'][i],k['yca'][i]])
                # Make template from Star A:
                imstamp = make_imagestamp(image, k['xca'][i],k['yca'][i], boxsizex = box, boxsizey = box)[0]
            else:
                imstamp = template
            synth = injectplanets(image, imhdr, imstamp, sep, pa, C, TC, k['xcb'][i], k['ycb'][i], 
                                  sepformat = sepformat, wavelength = 3.9, box = box)

        newname = (k['filename'][i]).split('_')[0]+'_'+(k['filename'][i]).split('_')[1]+'_synthsig'+outsuffix+'.fit'
        fits.writeto(newname,synth,overwrite=True, header=imhdr)
        g['filename'][i] = newname
        if showprog:
            update_progress(i+1,len(k))
    if returng:
        return g

def snyth_snr(image,xc,yc,xp,yp, radius = 9, r_in = None, r_out = None):
    ''' Compute the SNR of a synthetic planet signal for single planet or array of planets
        Parameters:
        -----------
        image : 2d array
            image
        xc,yc : int
            location of central object
        xp,yp : int
            location of synthetic planet
        radius : int
            radius for photutils aperture.  Default value corresponds to first null in CLIO 3.9um 
        r_in,r_out : int
            inner and outer radius of photutils annulus around central object.  Should be large 
            enough to exclude snyth planet signal from noise estimation.
        Returns:
        --------
        snr : flt
            signal to noise ratio of snyth planet signal
    '''
    from photutils import CircularAperture, CircularAnnulus, aperture_photometry
    if not r_in or not r_out:
        raise ValueError('annulus radii nedded')
    aperture = CircularAperture([xp,yp], r=radius)
    annulus_aperture = CircularAnnulus([xc,yc], r_in=r_in, r_out=r_out)
    phot = aperture_photometry(image, aperture)
    bkgd = aperture_photometry(image, annulus_aperture)
    bkgdmean = bkgd['aperture_sum']/annulus_aperture.area
    bkgdsum = bkgdmean*aperture.area
    signal = phot['aperture_sum'] - bkgdsum
    noise = np.sqrt(np.abs(bkgdsum)) + np.sqrt(np.abs(signal))
    snr = signal/noise
    return np.array(snr), aperture, annulus_aperture


def DoInjectionRecoverySNR_depricated(path, Star, sep, pa, C, radius = 9, r_in=None, r_out=None, sepformat='pixels',
                          K_klip=30, box = 100, g = []):
    ''' Wrapper function to inject fake planet signal into all images in dataset and compute
            snr of recovered signal in klipcube
        Parameters:
        -----------
        path : str
            dataset folder
        Star : 'A' or 'B'
            star to put the fake signal around
        sep : flt or fltarr
            separation of planet placement in either arcsec, mas, pixels, or lambda/D
        pa : flt or fltarr
            position angle of planet relative to north in DEG
        C : flt or fltarr
            desired contrast of planet with central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'pixels'
        radius, r_in,r_out : int
            radii for photutils photometry
        box : int
            size of box around template star
        K_klip : int
            desired klip modes for subtraction
        box : int
            boxsize for klipcube
        g : None or pandas dataframe
            if not None, inject planet signal into images through this function.  Otherwise, supply the pandas
            dataframe object of already injected signals which is output form DoInjection
    '''
    from cliotools.bditools import ab_stack_shift, psf_subtract, subtract_cubes, DoInjection
    if not any(g):
        print('not g')
        # inject signals:
        g = DoInjection(path, Star, sep, pa, C, sepformat = sepformat, returng = True, showprog=False)
    
    # create postage stamp cubes:
    astamp, bstamp = ab_stack_shift(g, boxsize = box, verbose = False)
    # psf subtract cubes
    a_final, b_final = subtract_cubes(astamp, bstamp, K_klip, g, write_to_disk = False, verbose = False)

    if Star == 'A':
        klipcube = a_final.copy()
    elif Star == 'B':
        klipcube = b_final.copy()
        
    xc,yc = box-1,box-1
    if sepformat == 'lambda/D':
        from cliotools.cliotools import lod_to_pixels
        seppix = lod_to_pixels(sep, 3.9)
    if sepformat == 'arcsec':
        pixscale = 15.9
        pixscale = pixscale/1000
        seppix = sep / pixscale
    if sepformat == 'mas':
        pixscale = 15.9
        seppix = sep / pixscale
    xx = seppix*np.sin(np.radians((pa)))
    yy = seppix*np.cos(np.radians((pa)))
    xp,yp = xc-xx-1,yc+yy-1
    if not r_in or not r_out:
        raise ValueError('annulus radii nedded')
    snr = snyth_snr(klipcube[-1],xc,yc,xp,yp, radius = radius, r_in=r_in, r_out=r_out)[0]
    return snr


def psf_subtract_deprecated(scienceimage, ref_psfs, K_klip, covariances = None, use_basis = False, basis = None, return_basis = False, return_cov = False,
                     verbose = True):
    """Build an estimator for the psf of a BDI science target image from a cube of reference psfs 
       (use the ab_stack_shift function).  Follows steps of Soummer+ 2012 sec 2.2
       
       Written by Logan A. Pearce, 2020
       Heavily influenced by the lovely coding over at PyKLIP (https://pyklip.readthedocs.io/en/latest/)
       
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
       outputimage : bxmxn arr
           psf subtracted image
       Z : Nxp arr
           psf model basis modes (if return_basis = True)
       cov : NxN arr
           return the computed covariance matrix is return_cov = True. So it can be used in future
           calcs without having to recalculate

    """
    from scipy import ndimage
    import image_registration
    # Shift science image to line up with reference psfs (aligned during the cubing step):
    dx,dy,edx,edy = image_registration.chi2_shift(np.sum(ref_psfs,axis=0), scienceimage, upsample_factor='auto')
    scienceimage = ndimage.shift(scienceimage, [-dy,-dx], output=None, order=4, mode='constant', \
                              cval=0.0, prefilter=True)
    # Start KLIP math:
    from scipy.linalg import eigh
    
    # Soummer 2012 2.2.1:
    ### Prepare science target:
    shape=scienceimage.shape
    p = shape[0]*shape[1]
    # Reshape science target into 1xp array:
    T_reshape = np.reshape(scienceimage,(p))
    # Subtract mean from science image:
    T_meansub = T_reshape - np.nanmean(T_reshape)
    # Make K_klip number of copies of science image
    # to use fast vectorized math:
    T_meansub = np.tile(T_meansub, (np.max(K_klip), 1))
    
    # KL Basis modes:
    if use_basis is True:
        Z = basis
    else:
        # Build basis modes:
        ### Prepare ref psfs:
        refshape=ref_psfs.shape
        N = refshape[0]
        if N < np.min(K_klip):
            if verbose:
                print("Oops! All of your requested basis modes are more than there are ref psfs.")
                print("Setting K_klip to number of ref psfs.  K_klip = ",N)
            K_klip = N
        if N < np.max(K_klip):
            if verbose:
                print("Oops! You've requested more basis modes than there are ref psfs.")
                print("Setting where K_klip > N to number of ref psfs.")
            K_klip[np.where(K_klip > N)] = N
            if verbose:
                print("K_klip = ",K_klip)
        K_klip = np.clip(K_klip, 0, N)
        R = np.reshape(ref_psfs,(N,p))
        # Compute the mean pixel value of each reference image:
        immean = np.nanmean(R, axis=1)
        # subtract mean:
        R_meansub = R - immean[:, None] #<- makes an empty first dimension to make
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
    # Subtract estimated psf from science image:
    outputimage = outputimage - Ihat
    # Reshape to 
    outputimage = np.reshape(outputimage, (np.size(K_klip),*shape))

    if (return_cov and return_basis) is True:
        return outputimage, Z, cov, lamb, c
    elif return_basis:
        return outputimage, Z
    elif return_cov:
        return outputimage, cov
    else:
        return outputimage
    
def subtract_cubes_deprecated(astamp, bstamp, K_klip, k, a_covariances=None, \
                   b_covariances=None, a_estimator=None, b_estimator=None, \
                   write_to_disk = True, write_directory = '.', outfilesuffix = '', \
                   headercomment = None, reshape = False, verbose = True
                       ):
    """For each image in the astamp/bstamp cubes, PSF subtract image, rotate to
       north up/east left; then median combine all images for star A and star B into
       single reduced image.  Do this for an array of KLIP mode cutoff values.  Writes
       reduced images to disk in fits cube with z axis = # of KLIP mode cutoff values.

       Written by Logan A. Pearce, 2020
       Dependencies: numpy, astropy.io.fits

       Parameters:
       -----------
       acube, bcube : 3d array
           cube of postage stamp images of star A and star B, array of shape Nxmxn 
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
           
       Returns:
       --------
       a_final, b_final : 3d arr
           cube of psf subtracted and sigma-clipped mean combined postagestamp images
           of star A and B, with z = # of KLIP mode cutoff values.
           If write_to_disk = True the cubes are written to fits files with
           custom headers with filename of system and suffix "_klipcube_a.fits" 
           and "_klipcube_b.fits"

    """
    from cliotools.bditools import rotate_clio, psfsub_cube_header, psf_subtract_deprecated
    from astropy.stats import sigma_clip

    acube, bcube = astamp.copy(),bstamp.copy()
    N = astamp.shape[0]
    if N < np.max(K_klip):
        if verbose:
            print("Oops! You've requested more basis modes than there are ref psfs.")
            print("Setting where K_klip > N to number of ref psfs.")
        K_klip[np.where(K_klip > N)] = N
        if verbose:
            print("K_klip = ",K_klip)
    # measure the final product image dimensions by performing rotation
    # on first image in cube:
    imhdr = fits.getheader(k['filename'][0])
    a0_rot = rotate_clio(astamp[0], imhdr, order = 4, reshape = reshape, mode='constant', cval=np.nan)
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
        Fa, Zb = psf_subtract_deprecated(acube[i], bcube, K_klip[j], return_basis = True, verbose = verbose)
        #Fa, Zb, immeanb = psf_subtract(astamp[i], bstamp, K_klip[j], return_basis = True, verbose = verbose)
        # get header and rotate image:
        imhdr = fits.getheader(k['filename'][i])
        Fa_rot = rotate_clio(Fa[0], imhdr, order = 4, reshape = reshape, mode='constant', cval=np.nan)
        # make a cube to store results:
        a = np.zeros([acube.shape[0],Fa_rot.shape[0],Fa_rot.shape[1]])
        # Place the psf subtracted image into the container cube:
        a[i,:,:] = Fa_rot
        # Use this basis to subtract all the remaining A images:
        for i in range(1,a.shape[0]):
            # subtract:
            F2a  = psf_subtract_deprecated(acube[i], bcube, K_klip[j], use_basis = True, basis = Zb, verbose = verbose)
            # rotate:
            imhdr = fits.getheader(k['filename'][i])
            F2a_rot = rotate_clio(F2a[0], imhdr, order = 4, reshape = reshape, mode='constant', cval=np.nan)
            # store:
            a[i,:,:] = F2a_rot
        # final product is combination of subtracted and rotated images:
        #a_final[j,:,:] = np.median(a, axis = 0)
        a_final[j,:,:] = np.mean(sigma_clip(a, sigma = 3, axis = 0), axis = 0)
        
        ############### star B: ##################
        # Repeat for star B:
        i = 0
        Fb, Za = psf_subtract_deprecated(bcube[i], acube, K_klip[j], return_basis = True, verbose = verbose)
        Fb_rot = rotate_clio(Fb[0], imhdr, order = 4, reshape = reshape, mode='constant', cval=np.nan)
        b = np.zeros(a.shape)
        b[i,:,:] = Fb_rot
        for i in range(1,b.shape[0]):
            # subtract:
            F2b = psf_subtract_deprecated(bcube[i], acube, K_klip[j], use_basis = True, basis = Za, verbose = verbose)
            # rotate:
            imhdr = fits.getheader(k['filename'][i])
            F2b_rot = rotate_clio(F2b[0], imhdr, order = 4, reshape =reshape, mode='constant', cval=np.nan)
            # store:
            b[i,:,:] = F2b_rot
        #b_final[j,:,:] = np.median(b, axis = 0)
        b_final[j,:,:] = np.mean(sigma_clip(b, sigma = 3, axis = 0), axis = 0)

    if write_to_disk is True:
        if verbose:
            print('Writing finished cubes to file... done!')
        newhdr = psfsub_cube_header(k['filename'][0].split('/')[0], K_klip, 'A', a_final.shape, acube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(k['filename'][0].split('_')[0]+'_klipcube_a'+outfilesuffix+'.fit',a_final,newhdr,overwrite=True)
        newhdr = psfsub_cube_header(k['filename'][0].split('/')[0], K_klip, 'B', b_final.shape, bcube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(k['filename'][0].split('_')[0]+'_klipcube_b'+outfilesuffix+'.fit',b_final,newhdr,overwrite=True)
    return a_final, b_final

def ab_stack_shift_deprecated(k, boxsize = 20, path_prefix='', verbose = True,
                ):
    """Prepare stacks for BDI/ADI by stacking and subpixel aligning image 
       postage stamps of star A and star B.
       Written by Logan A. Pearce, 2020
       
       Dependencies: astropy, image_registration, scipy

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
    import warnings
    warnings.filterwarnings("ignore")
    
    # Open first image in dataset:
    i = 0
    image = fits.getdata(path_prefix+k['filename'][i])
    # Make stamps:
    if len(image.shape) == 2:
        first_a = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                                                                 np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
        first_b = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                                                                 np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
    elif len(image.shape) == 3:
        first_a = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                                                                 np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
        first_b = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                                                                 np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
    # create empty containers:
    if len(image.shape) == 2:
        astamp = np.zeros([len(k),*first_a.shape])
        bstamp = np.zeros([len(k),*first_b.shape])
    elif len(image.shape) == 3:
        astamp = np.zeros([len(k)*image.shape[0],first_a.shape[1],first_a.shape[2]])
        bstamp = np.zeros([len(k)*image.shape[0],first_b.shape[1],first_b.shape[2]])
    
    # For each subsequent image, make the imagestamp for a and b, and 
    # align them with the first image, and add to stack:
    if len(image.shape) == 2:
        # place first a at bottom of stack:
        i = 0
        astamp[i,:,:] = first_a
        bstamp[i,:,:] = first_b
        for i in range(1,len(k)):
        # make stamp:
            a = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                   np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
            # place in container:
            try:
                astamp[i,:,:] = a
                # measure offset from the first image stamp:
                dx,dy,edx,edy = image_registration.chi2_shift(astamp[0,:,:], astamp[i,:,:], upsample_factor='auto')
                # shift new stamp by that amount:
                astamp[i,:,:] = ndimage.shift(astamp[i,:,:], [-dy,-dx], output=None, order=3, mode='constant', \
                                              cval=0.0, prefilter=True)
            except:
                if verbose:
                    print('PrepareCubes: Oops! the box is too big and one star is too close to an edge. I cut it off at i=',i)
                astamp = astamp[:i,:,:]
                bstamp = bstamp[:i,:,:]
                return astamp, bstamp
            # repeat for b:
            b = fits.getdata(path_prefix+k['filename'][i])[np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                   np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
            try:
                bstamp[i,:,:] = b
                # measure offset from the first image stamp:
                dx,dy,edx,edy = image_registration.chi2_shift(bstamp[0,:,:], bstamp[i,:,:], upsample_factor='auto')
                # shift new stamp by that amount:
                bstamp[i,:,:] = ndimage.shift(bstamp[i,:,:], [-dy,-dx], output=None, order=3, mode='constant', \
                              cval=0.0, prefilter=True)
            except:
                if verbose:
                    print('PrepareCubes: Oops! the box is too big and one star is too close to an edge. I cut it off at i=',i)
                astamp = astamp[:i,:,:]
                bstamp = bstamp[:i,:,:]
                return astamp, bstamp
    if len(image.shape) == 3:
        count = first_a.shape[0]
        astamp[0:count,:,:] = first_a
        bstamp[0:count,:,:] = first_b
        for i in range(1,len(k)):
            try:
                a = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['yca'][i]-boxsize):np.int_(k['yca'][i]+boxsize),\
                                                                   np.int_(k['xca'][i]-boxsize):np.int_(k['xca'][i]+boxsize)]
                astamp[count:count+a.shape[0],:,:] = a
                b = fits.getdata(path_prefix+k['filename'][i])[:,np.int_(k['ycb'][i]-boxsize):np.int_(k['ycb'][i]+boxsize),\
                                                                   np.int_(k['xcb'][i]-boxsize):np.int_(k['xcb'][i]+boxsize)]
                bstamp[count:count+b.shape[0],:,:] = b
                   
                for j in range(count,count+a.shape[0]):
                    dx,dy,edx,edy = image_registration.chi2_shift(astamp[0,:,:], astamp[j,:,:], upsample_factor='auto')
                    astamp[j,:,:] = ndimage.shift(astamp[j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                              prefilter=True)
                    dx,dy,edx,edy = image_registration.chi2_shift(bstamp[0,:,:], bstamp[j,:,:], upsample_factor='auto')
                    bstamp[j,:,:] = ndimage.shift(bstamp[j,:,:], [-dy,-dx], output=None, order=3, mode='constant', cval=0.0, \
                                                              prefilter=True)
                count += a.shape[0]
            except:
                if verbose:
                    print('ab_stack_shift: Oops! the box is too big and one star is too close to an edge. I cut it off at i=',count)
                astamp = astamp[:count,:,:]
                bstamp = bstamp[:count,:,:]
                return astamp, bstamp
    return astamp, bstamp
