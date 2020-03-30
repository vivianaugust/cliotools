def lam_over_d(lamb):
    ''' Return lambda-over-D in mas for CLIO narrow camera
        on Magellan Clay telescope.  Lambda must be in micrometers,
        or any distance unit if astropy unit object.
    '''
    import astropy.units as u
    try:
        lamb = lamb.to(u.m)
    except:
        lamb = (lamb*u.um).to(u.m)
    D = 6.5*u.m
    loverd = (1.12*(lamb/D)).value
    loverd = loverd*u.rad.to(u.mas)
    return loverd

def pixels_to_lamoverd(lamb,pixels):
    ''' Given the distance of something in pixel space,
        return the number of lambda-over-D it is for CLIO narrow camera
        on Magellan Clay telescope.  Lambda must be in micrometers,
        or any distance unit if astropy unit object.
    '''
    import astropy.units as u
    from cliotools.cliotools import lam_over_d
    loverd = lam_over_d(lamb)
    pixscale = 15.9 #mas/pixel
    loverd_pix = loverd / pixscale
    return pixels / loverd_pix

def lamoverd_to_pixels(lamb,pixels):
    ''' Given the distance of something in L/D space,
        return the pixel distance it is for CLIO narrow camera
        on Magellan Clay telescope.  Lambda must be in micrometers,
        or any distance unit if astropy unit object.
    '''
    import astropy.units as u
    from cliotools.cliotools import lam_over_d
    loverd = lam_over_d(lamb)
    pixscale = 15.9 #mas/pixel
    loverd_pix = loverd / pixscale
    return pixels / loverd_pix


def rotate_clio(image, imhdr, rotate_image = True, NORTH_CLIO = -1.80, **kwargs):
    """Rotate CLIO image to north up east left.
       Written by Logan A. Pearce, 2020
       
       Dependencies: scipy

       Parameters:
       -----------
       image : 2d array
           2d image array
       imhdr : fits header object
           header for image to be rotated
       rotate_image : bool
           if set to true, return rotated image.  If set
           to false, return angle by which to rotate an image
           to make North up East left
       NORTH_CLIO : flt
           NORTH_CLIO value.  Default = -1.80
           value taken from: https://magao-clio.github.io/zero-wiki/d017e/Astrometric_Calibration.html
       kwargs : for scipy.ndimage
           
       Returns:
       --------
       if rotate_image = True:
       imrot : 2d arr 
           rotated image with north up east left
       If rotate_image = False:
       derot : flt
           angle by which to rotate the image to
           get north up east left
    """
    
    NORTH_CLIO = -1.80
    derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
    if rotate_image == True:
        imrot = ndimage.rotate(image, derot, **kwargs)
        return imrot
    else:
        return derot


def daostarfinder(imagestamp, threshold = 1e4, fwhm = 10):
    """Find the subpixel location of a single star in a single clio BDI image.
       Written by Logan A. Pearce, 2020

       Parameters:
       -----------
       imagetamp : 2d array
           image postage stamp for looking for sources
       threshold : int
           threshold keyword for StarFinder
       fwhm : int
           fwhm keyword for StarFinder
           
       Returns:
       --------
       sources : table
           DAOStarFinder output table of sources
    """
    from photutils import DAOStarFinder
    import warnings
    import numpy as np
    warnings.filterwarnings('ignore')
    # Use DAOStarFinder to find subpixel locations:
    # If the threshold is too high and it can't find a point source, lower the threshold
    # until it finds something
    logthreshold = np.log10(threshold)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=10**logthreshold)
    sources = daofind(imagestamp)
    print(source)
    while sources==None:
        logthreshold -= 0.1
        #print(logthreshold)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=10**logthreshold)
        sources = daofind(imagestamp)

    return sources

def make_imagestamp(image, x, y, boxsizex=50, boxsizey=50):
    import numpy as np
    ymin, ymax = y-boxsizey, y+boxsizey
    xmin, xmax = x-boxsizex, x+boxsizex
    if ymin < 0:
        ymin = 0
    if ymax > image.shape[0]:
        ymax = image.shape[0]
    if xmin < 0:
        xmin = 0
    if xmax > image.shape[1]:
        xmax = image.shape[1]
    return image[np.int_(ymin):np.int_(ymax),\
                 np.int_(xmin):np.int_(xmax)], \
            xmin, xmax, ymin, ymax
    

def findtrapB(image, imstamp, boxsize=50, threshold=1e4, fwhm=10, x1=0, y1=0):
    """Gotta get fancy to find just those four Trapezium B stars in CLIO.
       Written by Logan A. Pearce, 2020

       Parameters:
       -----------
       image : 2d array
           image within which to find stars
       imagetamp : 2d array
           image postage stamp for looking for sources
       boxsize : int
            size to draw box around stars for DAOStarFinder
            to look for stars.  Box will have sides of length 2*boxsize
       threshold : int
           threshold keyword for StarFinder
       fwhm : int
           fwhm keyword for StarFinder
       x1, y1 : flt
           tell findtrapB where to look for star B1.  If set
           to 0, findtrapB will find the brightest correlation
           pixel as location of B1.  Tell findtrapB where to look
           if B1 is not the brightest star in the image (ex: in L band
           B2 might be brighter, but in Kp B1 is brightest)
           
       Returns:
       --------
       x_subpix, y_subpix : flt arr
           X and Y subpixel locations of star B1, B2, B3, B4 respectively
    """
    from scipy import signal, ndimage
    from photutils import DAOStarFinder
    import numpy as np
    from cliotools.cliotools import daostarfinder, make_imagestamp
    
    # Median filter to smooth image:
    image = ndimage.median_filter(image, 3)
    # Create cross-correlation image:
    corr = signal.correlate2d(image, imstamp, boundary='symm', mode='same')
    # Initialize output arrays:
    x_subpix, y_subpix = np.array([]), np.array([])
    
    ########### Find B1: ####################
    # Get x,y:
    if x1 != 0:
        x, y, = x1, y1
    else:
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
    # Creat image postage stamp of B1:
    imagestamp, xmin, xmax, ymin, ymax = make_imagestamp(image, x, y, boxsizex=boxsize, boxsizey=boxsize)
    # Use starfinder to find B1 subpixel location
    sources = daostarfinder(imagestamp)
    xs, ys = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
    # Append to source list:
    x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
    
    ######### Find B2: #######################
    # Mask B1:
    corr_masked = corr.copy()
    image_masked = image.copy()
    i=0
    xx,yy = np.meshgrid(np.arange(image.shape[1])-x_subpix[i],np.arange(image.shape[0])-y_subpix[i])
    rA=np.hypot(xx,yy)
    radius = 20
    corr_masked[np.where((rA < radius))] = 0
    image_masked[np.where((rA < radius))] = 0
    # Find location of B2:
    y, x = np.unravel_index(np.argmax(corr_masked), corr.shape)
    # Create image postage stamp of B2:
    imagestamp, xmin, xmax, ymin, ymax = make_imagestamp(image, x, y, boxsizex=boxsize, boxsizey=boxsize)
    # Use starfinder to find B2 subpixel location
    sources = daostarfinder(imagestamp)
    xs, ys = (xmin+sources['xcentroid'])[0], (ymin+sources['ycentroid'])[0]
    # Append to source list:
    x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
    
    ######### Find B3: ##########################
    # Mask B2:
    corr_masked2 = corr_masked.copy()
    image_masked2 = image_masked.copy()
    i=1
    xx,yy = np.meshgrid(np.arange(image.shape[1])-x_subpix[i],np.arange(image.shape[0])-y_subpix[i])
    rA=np.hypot(xx,yy)
    radius=5
    corr_masked2[np.where((rA < radius))] = 0
    image_masked2[np.where((rA < radius))] = 0
    ####### cut out:
    # We're getting to the two faint stars now, so
    # let's cut out just the Trapezium B cluster so we don't get fooled by other objects.
    # Take the mean location of the two bright stars:
    x2,y2 = np.mean(x_subpix), np.mean(y_subpix)
    # Find if x-axis or y-axis orientation is bigger:
    diff = np.array([np.abs(np.diff(x_subpix))[0], np.abs(np.diff(y_subpix))[0]])
    # Set the boxsize according to orientation:
    if np.where(np.max([np.diff(x_subpix), np.diff(y_subpix)]))[0]==1:
        # If oriented along x-axis:
        boxsizex = 70
        boxsizey = 50
    else:
        # if oriented along y-axis:
        boxsizex = 50
        boxsizey = 70
    # Cut out TrapB section of image and correlation map:   
    image_masked2, xmin2, xmax2, ymin2, ymax2 = make_imagestamp(image_masked2, x2,y2, \
                                                                boxsizex=boxsizex, boxsizey=boxsizey)
    corr_masked2, xmin2, xmax2, ymin2, ymax2 = make_imagestamp(corr_masked2, x2,y2, \
                                                               boxsizex=boxsizex, boxsizey=boxsizey)
    # Find B3:
    y, x = np.unravel_index(np.argmax(corr_masked2), corr_masked2.shape)
    # Create image postage stamp of B3:
    imagestamp, xmin, xmax, ymin, ymax = make_imagestamp(image_masked2, x, y, boxsizex=boxsize, boxsizey=boxsize)
    # Use starfinder to find B2 subpixel location
    sources = daostarfinder(imagestamp)
    xs, ys = (xmin2+xmin+sources['xcentroid'])[0], (ymin2+ymin+sources['ycentroid'])[0]
    x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
    
    ######### Find B4: ##########################
    # Mask B3:
    corr_masked3 = corr_masked.copy()
    image_masked3 = image_masked.copy()
    i=2
    xx,yy = np.meshgrid(np.arange(image.shape[1])-x_subpix[i],np.arange(image.shape[0])-y_subpix[i])
    rA=np.hypot(xx,yy)
    radius=20
    corr_masked3[np.where((rA < radius))] = 0
    image_masked3[np.where((rA < radius))] = 0
    # Do a new cutout with large B2/3 mask:
    x2,y2 = np.median(x_subpix[0:2]), np.median(y_subpix[0:2])
    image_masked3, xmin2, xmax2, ymin2, ymax2 = make_imagestamp(image_masked3, x2,y2, \
                                                                boxsizex=boxsizex, boxsizey=boxsizey)
    corr_masked3, xmin2, xmax2, ymin2, ymax2 = make_imagestamp(corr_masked3, x2,y2, \
                                                               boxsizex=boxsizex, boxsizey=boxsizey)
    # Skipping unravel index because it is too faint:
    sources = daostarfinder(image_masked3)
    xs, ys = (xmin2+sources['xcentroid'])[0], (ymin2+sources['ycentroid'])[0]
    x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
    return x_subpix, y_subpix
