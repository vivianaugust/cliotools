def daostarfinder(scienceimage, x, y, boxsize = 100, threshold = 1e4, fwhm = 10):
    """Find the subpixel location of a single star in a single clio BDI image.
       Written by Logan A. Pearce, 2020

       Parameters:
       -----------
       image : 2d array
           boxsizex by boxsizey substamp image of a reference psf for cross correlation
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
    import numpy as np
    image2 = scienceimage
    # The settings fwhm = 10.0 seems to be good for clio
    # Define box around source B:
    ymin, ymax = y-boxsize, y+boxsize
    xmin, xmax = x-boxsize, x+boxsize
    # Correct for sources near image edge:
    if ymin < 0:
        ymin = 0
    if ymax > scienceimage.shape[1]:
        ymax = scienceimage.shape[1]
    if xmin < 0:
        xmin = 0
    if xmax > scienceimage.shape[0]:
        xmax = scienceimage.shape[0]
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
            daofind = DAOStarFinder(fwhm=10.0, threshold=threshold) 
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
    from scipy import signal, ndimage
    from photutils import DAOStarFinder
    from astropy.io import fits
    import numpy as np
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
    from astropy.io import fits
    import os
    from cliotools.pcaskysub import update_progress
    import numpy as np
    # Supress warnings when failing to find point sources
    import warnings
    warnings.filterwarnings("ignore")
    # Make a file to store results:
    newfile = dataset_path.split('/')[0]+'/StarLocations'
    
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
