def findstars(imstamp, scienceimage_filename, xca, yca, xcb, ycb, boxsizex = 60, boxsizey = 60):
    """Find the subpixel location of stars A and B in a single clio BDI image.
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
    #file2 = 'BDI0933/BDI0933__00098_skysub.fit'
    image2 = fits.getdata(scienceimage_filename)
    image2 = ndimage.median_filter(image2, 3)
    # Use cross-correlation to find int(y,x) of star A (brightest star) in image:
    corr = signal.correlate2d(image2, imstamp, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    # Define the star finder parameters:
    daofind = DAOStarFinder(fwhm=8.0, threshold=1e4) 
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
    sources = daofind(image2[np.int_(ymin):np.int_(ymax),\
                             np.int_(xmin):np.int_(xmax)])
    xcb_subpix, ycb_subpix = (xca_subpix+deltax-boxsizex+sources['xcentroid'])[0], \
            (yca_subpix+deltay-boxsizey+sources['ycentroid'])[0]
    return xca_subpix, yca_subpix, xcb_subpix, ycb_subpix

def findstars_in_dataset(dataset_path, xca, yca, xcb, ycb, boxsizex = 60, boxsizey = 60, skip_list = False, \
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
    warnings.filterwarnings("ignore")
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
    # Apply median filter to smooth bad pixels:
    image = ndimage.median_filter(image, 3)
    # Create referance stamp from initial image of A:
    imstamp = np.copy(image[np.int_(yca-boxsizey):np.int_(yca+boxsizey),np.int_(xca-boxsizex):np.int_(xca+boxsizex)])
    
    count = 0
    for im in ims:
        # For each image in the dataset, find subpixel location of stars:
        try:
            xca_subpix, yca_subpix, xcb_subpix, ycb_subpix = findstars(imstamp, im, \
                                                           xca, yca, xcb, ycb, \
                                                           boxsizex = 60, boxsizey = 60)
            #xca, yca, xcb, ycb = xca_subpix, yca_subpix, xcb_subpix, ycb_subpix
            string = im + ' ' + str(xca_subpix)+' '+str(yca_subpix)+' '+str(xcb_subpix)+' '+str(ycb_subpix)
        except:
            # DAOStarFinder failed to find a point source due to poor image quality.  Make
            # a comment in the locations file
            print(im,': failed to find stars')
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
