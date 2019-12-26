def findstars(imstamp, scienceimage_filename, xca, yca, boxsizex = 100, boxsizey = 100, threshold = 1e4):
    """Find the subpixel location of stars A and B in a single clio BDI image.
       Parameters:
       -----------
       imstamp : 2d array
           boxsizex by boxsizey substamp image of a reference psf for cross correlation
       scienceimage_filename : string
           path to science image
       xca, yca : int
           integer pixel locations of star A
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
           
       Returns:
       --------
       xca_subpix, yca_subpix, xcb_subpix, ycb_subpix : flt
           subpixel location of star A and star B
    """
    from scipy import signal, ndimage
    from photutils import DAOStarFinder
    from astropy.io import fits
    import numpy as np
    # Supress warnings when failing to find point sources
    import warnings
    warnings.filterwarnings("ignore")
    
    # Open science target image:
    image2 = fits.getdata(scienceimage_filename)
    if len(image2.shape) == 3:
        image2 = image2[0]
    image2 = ndimage.median_filter(image2, 3)
    # Use cross-correlation to find int(y,x) of star A (brightest star) in image:
    corr = signal.correlate2d(image2, imstamp, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    # Define the star finder parameters:
    # The settings fwhm = 10.0 threshold = 1e4 are best for distinguishing good psf
    # from bad psf, by trial and error.
    daofind = DAOStarFinder(fwhm=10.0, threshold=threshold) 
    # Find sub-pixel location of star A in science image by using DAOStarFinder on a postagestamp centered at
    # cross-correlation x,y position results:
    sources = daofind(image2[np.int_(y-boxsizey):np.int_(y+boxsizey),np.int_(x-boxsizex):np.int_(x+boxsizex)])
    
    # If it finds too many sources, keep uping the search threshold until there is only one:
    while len(sources) > 1:
        threshold += 500
        daofind = DAOStarFinder(fwhm=10.0, threshold=threshold) 
        sources = daofind(image2[np.int_(y-boxsizey):np.int_(y+boxsizey),np.int_(x-boxsizex):np.int_(x+boxsizex)])
        
    # Get image location of star A:
    xca_subpix, yca_subpix = (x-boxsizex+sources['xcentroid'])[0], (y-boxsizey+sources['ycentroid'])[0]
    
    # Make a mask around the brightest star on the cross-correlation image:
    radius = 20
    # Make a meshgrid of the image centered at star A:
    xx,yy = np.meshgrid(np.arange(image2.shape[1])-xca_subpix,np.arange(image2.shape[0])-yca_subpix)
    # Make an array of the distances of each pixel from that center:
    rA=np.hypot(xx,yy)
    # Mask wherever that distance is less than the set radius:
    corr_masked = corr.copy()
    # Set those pixels to zero:
    corr_masked[np.where((rA < radius))] = 0
    # Now find the new highest correlation which should be the other star:
    y2, x2 = np.unravel_index(np.argmax(corr_masked), corr.shape)

    # Define box around source B:
    ymin, ymax = y2-boxsizey, y2+boxsizey
    xmin, xmax = x2-boxsizex, x2+boxsizex
    # Correct for sources near image edge:
    if ymin < 0:
        ymin = 0
    if ymax > 512:
        ymax = 512
    if xmin < 0:
        xmin = 0
    if xmax > 1024:
        xmax = 1024
    # Use DAOFind to find subpixel location of B:
    # If the threshold is too high and it can't find a point source, lower the threshold
    # until ~ 1e3, then declare the image might be bad.
    try:
        daofind = DAOStarFinder(fwhm=10.0, threshold=threshold) 
        sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                     np.int_(xmin):np.int_(xmax)])
        xcb_subpix, ycb_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
    except:
        try:
            daofind = DAOStarFinder(fwhm=10.0, threshold=5e3) 
            sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                         np.int_(xmin):np.int_(xmax)])
            xcb_subpix, ycb_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
        except:
            try:
                daofind = DAOStarFinder(fwhm=10.0, threshold=1e3) 
                sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                                             np.int_(xmin):np.int_(xmax)])
                xcb_subpix, ycb_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
            except:
                print('I just cant find it, it might be too faint or not there.')

    # If it finds too many sources, keep uping the search threshold until there is only one:
    while len(sources) > 1:
        threshold += 500
        daofind = DAOStarFinder(fwhm=10.0, threshold=threshold) 
        sources = daofind(image2[np.int_(y-boxsizey):np.int_(y+boxsizey),np.int_(x-boxsizex):np.int_(x+boxsizex)])
        xcb_subpix, ycb_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
        
    return xca_subpix, yca_subpix, xcb_subpix, ycb_subpix

def findstars_in_dataset(dataset_path, xca, yca, corrboxsizex = 40, corrboxsizey = 40, boxsizex = 100, boxsizey = 100, skip_list = False, \
                         append_file = False):
    """Find the subpixel location of stars A and B in a clio BDI dataset.
       Parameters:
       -----------
       dataset_path : string
           path to science images including image prefixes and underscores.  
           ex: An image set of target BDI0933 with filenames of the form BDI0933__00xxx.fit
               would take as input a path string of 'BDI0933/BDI0933__'
       xca, yca, xcb, ycb : int
           integer pixel locations of star A and B in the first image of the dataset, rough guess 
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

       Returns:
       --------
       writes subpixel location of star A and star B to file called 'ABlocations' in order
           xca_subpix, yca_subpix, xcb_subpix, ycb_subpix
    """
    from scipy import ndimage
    from astropy.io import fits
    import os
    from cliotools.pcaskysub import update_progress
    import numpy as np
    # Supress warnings when failing to find point sources
    import warnings
    #warnings.filterwarnings("ignore")
    # Make a file to store results:
    newfile = dataset_path.split('/')[0]+'/ABlocations'
    if append_file == False:
        k = open(newfile, 'w')
        k.write('#     xca     yca     xcb     ycb' + "\n")
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
        try:
            xca_subpix, yca_subpix, xcb_subpix, ycb_subpix = findstars(imstamp, im, \
                                                           xca, yca, \
                                                           boxsizex = boxsizex, boxsizey = boxsizey)
            if count % 25 == 0:
                xca, yca, xcb, ycb = xca_subpix, yca_subpix, xcb_subpix, ycb_subpix
            string = im + ' ' + str(xca_subpix)+' '+str(yca_subpix)+' '+str(xcb_subpix)+' '+str(ycb_subpix)
        except:
            # DAOStarFinder failed to find a point source due to poor image quality.  Make
            # a comment in the locations file
            print(im,': failed to find stars')
            # play beep if failed:
            os.system('echo -e "\a"')
            string = '# ' + im + ' ' + str('...')+' '+str('...')+' '+str('...')+' '+str('...')
        # Write results to file:
        k = open(newfile, 'a')
        k.write(string + "\n")
        k.close()
        # update x,y locations:
        
        count+=1
        update_progress(count,len(ims))
    print('Done')
    os.system("say 'done'")


def findstars_depreicated(imstamp, scienceimage_filename, xca, yca, xcb, ycb, boxsizex = 60, boxsizey = 60):
    """Find the subpixel location of stars A and B in a single clio BDI image. **DEPRICATED**
       Parameters:
       -----------
       imstamp : 2d array
           boxsizex by boxsizey substamp image of a reference psf for cross correlation
       scienceimage_filename : string
           path to science image
       xca, yca, xcb, ycb : int
           integer pixel locations of star A and B rough guess
       boxsize : int
           size of box to draw around star psfs for DAOStarFinder
           
       Returns:
       --------
       xca_subpix, yca_subpix, xcb_subpix, ycb_subpix : flt
           subpixel location of star A and star B
    """
    from scipy import signal, ndimage
    from photutils import DAOStarFinder
    from astropy.io import fits
    import numpy as np
    
    # Open science target image:
    image2 = fits.getdata(scienceimage_filename)
    if len(image2.shape) == 3:
        image2 = image2[0]
    image2 = ndimage.median_filter(image2, 3)
    # Use cross-correlation to find int(y,x) of star A (brightest star) in image:
    corr = signal.correlate2d(image2[:,0:275], imstamp, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    # Define the star finder parameters:
    # The settings fwhm = 10.0 threshold = 1e4 are best for distinguishing good psf
    # from bad psf, by trial and error.
    daofind = DAOStarFinder(fwhm=10.0, threshold=1e3) 
    # Find sub-pixel location of star A in science image by using DAOStarFinder on a postagestamp centered at
    # cross-correlation x,y position results:
    sources = daofind(image2[np.int_(y-boxsizey):np.int_(y+boxsizey),np.int_(x-boxsizex):np.int_(x+boxsizex)])
    # Get image location of star A:
    xca_subpix, yca_subpix = (x-boxsizex+sources['xcentroid'])[0], (y-boxsizey+sources['ycentroid'])[0]

    # Find integer pixel location of B relative to A:
    deltax, deltay = xcb - xca, ycb - yca
    
    # Define box around source B:
    ymin, ymax = yca_subpix+deltay-boxsizey, yca_subpix+deltay+boxsizey
    xmin, xmax = xca_subpix+deltax-boxsizex, xca_subpix+deltax+boxsizex
    # Correct for sources near image edge:
    if ymin < 0:
        ymin = 0
    if ymax > 512:
        ymax = 512
    if xmin < 0:
        xmin = 0
    if xmax > 1024:
        xmax = 1024
    # Use DAOFind to find subpixel location of B in an image stamp centered at A's location plus the
    # delta pixels of B from A:
    daofind = DAOStarFinder(fwhm=10.0, threshold=1e3)
    sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                             np.int_(xmin):np.int_(xmax)])
    xcb_subpix, ycb_subpix = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
    return xca_subpix, yca_subpix, xcb_subpix, ycb_subpix
