from astropy.io import fits
import numpy as np
import os
from scipy import ndimage
import pandas as pd
import pickle
import matplotlib.pyplot as plt


################ general clio tools ###########
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

def lambdaoverD_to_arcsec(lamb, D = 6.5):
    """ Compute lamb/D.  Default is for Magellan mirror and CLIO narrow camera pixelscale.
        Inputs:
            lamb [um]: central wavelength of filter in microns.  Astropy unit object preferred
            D [m]: primary mirror diameter.  Astropy unit object required
            pixscale [mas/pix]: 
        Returns:
            loverd [arcsec]: lambda/D in arcsec per L/D
    """
    arcsec = (0.2063*(lamb/D))
    return arcsec

def lambdaoverD_pix(lamb, pixscale = 15.9):
    from cliotools.bditools import lambdaoverD_to_arcsec
    import astropy.units as u
    loverd = lambdaoverD_to_arcsec(lamb)
    loverd_pix = loverd*u.arcsec.to(u.mas) / pixscale
    return loverd_pix

def lod_to_arcsec(lod):
    """ Lambda/D into arcsec for Magellan and 3.9 um filter
    """
    return lod * 0.12378

def arcsec_to_lod(arcsec):
    """ arcsec into Lambda/D for Magellan and 3.9 um filter
    """
    return arcsec / 0.12378

def pixels_to_lod(pixels, lamb, pixscale = 15.9):
    """ Convert a distance in pixels to lambda over D
    """
    from cliotools.bditools import lambdaoverD_pix
    loverd_pix = lambdaoverD_pix(lamb)
    return pixels/loverd_pix

def lod_to_pixels(lod, lamb, pixscale = 15.9):
    """ Convert separation in lambda/D to pixels
    """
    from cliotools.bditools import lambdaoverD_pix
    loverd_pix = lambdaoverD_pix(lamb)
    return lod*loverd_pix

def lod_to_physical(lod, distance, lamb):
    ''' Convert a distance in lamda over D to AU
        Inputs:
            lod [arcsec]: # of lambda over D to convert
            distance [pc]: distance to system in parsecs
            lamb [um]: filter central wavelength in microns
        Returns:
        
    '''
    from cliotools.bditools import lambdaoverD_to_arcsec
    import astropy.units as u
    # 1 lambda/D in arcsec per l/D:
    loverd = lambdaoverD_to_arcsec(lamb)
    # convert to arcsec:
    arcsec = lod*loverd
    return (arcsec * distance)*u.AU
    
    
def physical_to_lod(au, distance, lamb):
    ''' Convert a physucal distance in AU to lambda over D
    '''
    from cliotools.bditools import lambdaoverD_to_arcsec
    arcsec = au/distance
    loverd = lambdaoverD_to_arcsec(lamb)
    return arcsec/loverd

def pixel_seppa(x1,y1,x2,y2,imhdr=None):
    sep = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    sep = sep*15.9
    sep = sep/1000
    if imhdr:
        pa = (np.degrees(np.arctan2((x2-x1),(x2-x1)))+270)%360
        NORTH_CLIO = -1.80
        derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
        pa = (pa + derot)
        return sep,pa%360.
    else:
        return sep

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

########## Mag and contrast: ###########
def mag(image, x, y, radius = 9., r_in = 10., r_out = 12., returnflux = False, returntable = False):
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
    '''
    from photutils import CircularAperture, CircularAnnulus, aperture_photometry
    # Position of star:
    positions = [(x,y)]
    # Use radius equal to the firsl null, 1 lamb/d ~ 8.7 pixels:
    radius = 9.
    # Get sum of all pixel values within radius of center:
    apertures = CircularAperture(positions, r=radius)
    # Get sum of all pixels in annulus around center to sample background:
    annulus_apertures = CircularAnnulus(positions, r_in=10., r_out=12.)
    # Put into list:
    apers = [apertures, annulus_apertures]
    # Do photometry on star and background:
    phot_table = aperture_photometry(image, apers)
    # Background mean:
    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area
    # Backgroud within star aperture:
    bkg_sum = bkg_mean * apertures.area
    # Subtract background from star flux:
    final_sum = phot_table['aperture_sum_0'] - bkg_sum 
    # Add column to table with this final flux
    phot_table['Final_aperture_flux'] = final_sum  
    # Put into instrument magnitudes:
    mag=(-2.5)*np.log10(phot_table['Final_aperture_flux'][0])
    if returnflux:
        return mag, final_sum[0]
    if returntable:
        phot_table['Mag'] = mag
        return mag, phot_table
    return mag

def contrast(image, pos, **kwargs):
    ''' Return contrast of component B relative to A in magnitudes
        Parameters:
        ------------
        image : 2d array
            science image
        pos : arr
            x and y pixel location of center of star in order [xa,ya,xb,yb]
        kwargs : 
            args to pass to mag function
        Returns:
        --------
        contrast : flt
            contrast in magnitudes of B component relative to A component
        
    '''
    Amag = mag(image,pos[0],pos[1], **kwargs)
    Bmag = mag(image,pos[2],pos[3], **kwargs)
    return Bmag - Amag


########## Find Trap B stars: ###########

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
    from cliotools.bditools import daostarfinder, make_imagestamp
    
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


def daostarfinder_trapB(imagestamp, threshold = 1e4, fwhm = 10):
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
              boxsize = 100, threshold = 1e4, fwhm = 10, radius = 20,
              a_guess = [], b_guess = []):
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
        a_guess, b_guess : tuple
            (x,y) pixel tuple of rought guess of location of star A and B in image
           
       Returns:
       --------
       x_subpix, y_subpix : arr,flt
           1 x nstars array of subpixel x location and y location of stars
    """
    from scipy import signal
    
    # Open science target image:
    image = fits.getdata(scienceimage_filename)
    # If image is a cube, take the first image:
    if len(image.shape) == 3:
        image = image[0]
    # Median filter to smooth image:
    image = ndimage.median_filter(image, 3)
    # Make container to hold results:
    x_subpix, y_subpix = np.array([]), np.array([])

    if len(a_guess) != 0:
        # Run starfinder at location of guess:
        x,y = a_guess[0],a_guess[1]
        xs, ys = daostarfinder(image, x, y, boxsize = boxsize, threshold = threshold, fwhm = fwhm)
        x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
        # repeat for B:
        x,y = b_guess[0],b_guess[1]
        xs, ys = daostarfinder(image, x, y, boxsize = boxsize, threshold = threshold, fwhm = fwhm)
        x_subpix, y_subpix = np.append(x_subpix, xs), np.append(y_subpix, ys)
    
    else:
        # Use cross-correlation to find int(y,x) of star A (brightest star) in image:
        corr = signal.correlate2d(image, imstamp, boundary='symm', mode='same')
        # Find the location of the brightest star in the image:
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
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
                         append_file = False, threshold = 1e4, radius = 20, fwhm = 10, filesuffix = '_skysub'):
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
        os.system('ls '+dataset_path+'0*'+filesuffix+'.fit > list')
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
    

def ab_stack_shift(k, boxsize = 50, fwhm = 7.8, path_prefix='', verbose = True):
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
                    print('ab_stack_shift: Oops! the box is too big and one star is too close to an edge. Skipping ',k['filename'][i])
                #count += 1

    # Chop off the tops of the cubes to eliminate zero arrays from skipped stars:
    if astamp.shape[0] > count:
        astamp = astamp[:count,:,:]
    if bstamp.shape[0] > count:
        bstamp = bstamp[:count,:,:]
    return astamp,bstamp


def PrepareCubes(k, boxsize = 20, path_prefix='', verbose = True,\
                   # normalizing parameters:\
                   normalize = True, normalizebymask = False,  normalizing_radius = [],\
                   # star masking parameters:\
                   inner_mask_core = True, inner_radius_format = 'pixels', inner_mask_radius = 1., cval = 0,\
                   # masking outer annulus:\
                   outer_mask_annulus = True, outer_radius_format = 'pixels', outer_mask_radius = None,\
                   # subtract radial profile from cubes:\
                   subtract_radial_profile = True,\
                   # User supplied cubes:
                   acube = None, bcube = None,
                   # DAOStarfinder parameters:
                   fwhm = 7.8
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
    fwhm : flt
        FWHM for Starfinder centering
        
    Returns:
    --------
    3d arr, 3d arr
        cube of images of Star A and B ready to be put into KLIP reduction pipeline.
    '''
    if np.size(acube) == 1:
        # collect and align postage stamps of each star:
        from cliotools.bditools import ab_stack_shift
        astack, bstack = ab_stack_shift(k, boxsize = boxsize,  fwhm = fwhm, path_prefix=path_prefix, verbose = verbose)
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
        astack2,bstack2 = normalize_cubes(astack,bstack, normalizebymask = normalizebymask,  radius = normalizing_radius)
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

def circle_mask(radius, xsize, ysize, xc, yc, radius_format = 'pixels', cval = 0):
    xx,yy = np.meshgrid(np.arange(xsize)-xc,np.arange(ysize)-yc)
    r=np.hypot(xx,yy)
    return np.where(r<radius)

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
            positions = (astack.shape[1]/2,astack.shape[2]/2)
            aperture = CircularAperture(positions, r=radius)
            # Turn it into a mask:
            aperture_masks = aperture.to_mask(method='center')
            # pull out just the data in the mask:
            aperture_data = aperture_masks.multiply(data[i])
            aperture_data[np.where(aperture_data == 0)] = np.nan
            summed = np.nansum(aperture_data)
            a[i] = data[i] / summed
        else:
            # get the sum of pixels in the mask:
            summed = np.sum(data[i])
            # normalize to that sum:
            a[i] = data[i] / summed
        # repeat for B:
        data = bstack
        if normalizebymask:
            from photutils import CircularAperture
            positions = (astack.shape[1]/2,astack.shape[2]/2)
            aperture = CircularAperture(positions, r=radius)
            # Turn it into a mask:
            aperture_masks = aperture.to_mask(method='center')
            # pull out just the data in the mask:
            aperture_data = aperture_masks.multiply(data[i])
            aperture_data[np.where(aperture_data == 0)] = np.nan
            summed = np.nansum(aperture_data)
            b[i] = data[i] / summed
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
        from cliotools.bditools import lod_to_pixels
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
        from cliotools.bditools import lod_to_pixels
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

def deconvolve_app_mag(magT,k):
    from cliotools.bditools import mag
    # compute average deltamag in images:
    deltamagarray = np.array([])
    for i in range(len(k)):
        image = fits.getdata(k['filename'][i])
        if len(image.shape) == 3:
            image = image[0]
        else:
            pass
        magA, fluxA = mag(image,k['xca'][i],k['yca'][i], returnflux = True)
        magB, fluxB = mag(image,k['xcb'][i],k['ycb'][i], returnflux = True)
        deltamag = magA-magB
        deltamagarray = np.append(deltamagarray,deltamag)
    deltamag = np.median(deltamagarray)
    # Compute magA:
    exp1 = -(deltamag + magT) / 2.5
    num = 10**exp1
    denom = 1 + 10**(-deltamag / 2.5)
    fA = num / denom
    magA = -2.5*np.log10(fA)
    # compute magB:
    fB = 10**(-magT/2.5) - fA
    magB = -2.5*np.log10(fB)
    return magA,magB

def mkheader(dataset, star, shape, normalized, inner_masked, outer_masked):
    """ Make a header for writing psf sub BDI KLIP cubes to fits files
        in the subtract_cubes function
    """
    import time
    from astropy.io import fits
    header = fits.Header()
    header['COMMENT'] = '         ************************************'
    header['COMMENT'] = '         **  Cube of Star '+star+' PSFs          **'
    header['COMMENT'] = '         ************************************'
    header['COMMENT'] = 'Postagestamp cube of PSF images that have been aligned and bad pixel detailed'
    header['COMMENT'] = 'and are ready to go into PrepareCubes'
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
    header['NORMALIZED'] = normalized
    header['INNER MASKED'] = inner_masked
    header['OUTER_MASKED'] = outer_masked
    header['COMMENT'] = 'by Logan A Pearce'
    return header



def GetSNR(path, Star, K_klip, sep, pa, C, boxsize = 50,
                sepformat = 'lambda/D',
                returnsnrs = False, writeklip = False, update_prog = False, 
                sciencecube = [],
                refcube = [],
                templatecube = [], 
                mask_core = True, mask_outer_annulus = True, mask_radius = 5., outer_mask_radius = 50., subtract_radial_profile = True,
                normalize = True, normalizebymask = False, normalizing_radius = [],
                wavelength = 3.9):
    ''' For a single value of separation, position angle, and contrast, inject a fake signal and perform KLIP reduction.

    sciencecube, refcube, and templatecube are optional varaibles for supplying a previously constructed
    3d cube of images for doing the KLIP reduction on the science target, using the reference cube, with 
    fake signal injection provided by the templatecube.  If not provided, the pipeline will construct the necessary 
    cubes from the images listed in CleanList and the specified science star.  The template star by default is the
    same as the science star if not provided by user.

    Parameters
    -----------
    path : str
        dataset folder
    Star : 'A' or 'B'
        which star had the signal injection
    K_klip : int
        number of KLIP modes for the reduction
    sep : flt
        separation to test
    pa : flt
        position angle in degrees from North
    C : flt
        contrast to test
    sepformat : str
        format of provided separations, either `lambda/D` or `pixels`.  Default = `lambda/D`
    returnsnrs : bool
        if True, return the SNRs array as well as the mean.  Default = False
    writeklip : bool
        if True, a fits file of the first injected signal KLIP-reduced image at each contrast and \
        separation will be written to disk.  Default = False
    update_prog : bool
        if True, display a progress bar for computing the SNR in the ring. Default = False, progress \
        bar within RunContrastCurveCalculation is the priority when running a full calculation
    sciencecube : 3d arr
        optional user-provided cube of psf images for doing the KLIP reduction. 
    refcube : 3d arr
        optional user-provided cube of reference psfs.
    templatecube : 3d arr
        optional user-provded cube of images to use as injected signal psf template.
    mask_core : bool
        if True, set all pixels within mask_radius to value of 0 in all images.  Default = True
    mask_radius : flt
        radius of inner mask in lambda/D or pixels.  Default = 1. lamdba/D
    mask_outer_annulus: bool
        if True, set all pixels outside the specified radius to zero.  Default = False
    outer_mask_radius : flt
        Set all pixels outside this radius to 0 if mask_outer_aunnulus set to True.
    subtract_radial_profile : bool
        If True, subtract the median radial profile from each image in the cubes.  Default = True.
    wavelength : flt
        central wavelength of filter band in microns.  Default = 3.9

    Returns
    -------
    flt
        SNR for injected signal at that sep, PA, contrast
    object
       SyntheticSignal object, cube with injected signal
    object
        BDI object, KLIP reduced object with injected signal
    '''
    k = pd.read_csv(path+'CleanList', comment='#')
    
    SynthCubeObject2 = SyntheticSignal(k, Star, sep, pa, C, verbose = False, 
                                  sciencecube = sciencecube,
                                  refcube = refcube,
                                  templatecube = templatecube
                                 )

    if Star == 'A':
        acube = SynthCubeObject2.synthcube
        bcube = SynthCubeObject2.refcube
    elif Star == 'B':
        acube = SynthCubeObject2.refcube
        bcube = SynthCubeObject2.synthcube
    
    from cliotools.bdi import BDI
    # create BDI object with injected signal:
    SynthCubeObjectBDI2 = BDI(k, path, K_klip = K_klip, 
                    boxsize = boxsize,         
                    normalize = normalize, 
                    normalizebymask = normalizebymask, 
                    normalizing_radius = normalizing_radius,       
                    inner_mask_core = mask_core,        
                    inner_radius_format = 'pixels',
                    inner_mask_radius = mask_radius,        
                    outer_mask_annulus = mask_outer_annulus,     
                    outer_radius_format = 'pixels',
                    outer_mask_radius = outer_mask_radius,       
                    mask_cval = 0,       
                    subtract_radial_profile = subtract_radial_profile,          
                    verbose = False,               
                    acube = acube,    
                    bcube = bcube   
                   )
    # Do klip reduction:
    SynthCubeObjectBDI2.Reduce(interp='bicubic',
                 rot_cval=0.,
                 mask_interp_overlapped_pixels = True
                ) 
    if Star == 'A':
        kliped = SynthCubeObjectBDI2.A_Reduced
    elif Star == 'B':
        kliped = SynthCubeObjectBDI2.B_Reduced
        
    xc, yc = (0.5*((kliped.shape[1])-1),0.5*((kliped.shape[0])-1))
    snr = getsnr(kliped, sep, pa, xc, yc, wavelength = wavelength)
    if writeklip:
        from astropy.io import fits
        name = path+'/injectedsignal_star'+Star+'_sep'+'{:.0f}'.format(sep)+'_C'+'{:.1f}'.format(C)+'.fit'
        fits.writeto(name,kliped,overwrite=True)
    
    return snr, SynthCubeObject2, SynthCubeObjectBDI2

def DoSNR(path, Star, K_klip, sep, C, 
                sepformat = 'lambda/D',
                sep_cutout_region = [0,0], pa_cutout_region = [0,0],
                returnsnrs = False, writeklip = False, update_prog = False, 
                sciencecube = [],
                refcube = [],
                templatecube = [], 
                mask_core = True, mask_outer_annulus = True, mask_radius = 5., outer_mask_radius = 50.,
                normalize = True, normalizebymask = False, normalizing_radius = [],
                subtract_radial_profile = True, wavelength = 3.9
                ):
    ''' For a single value of separation and contrast, compute the SNR at that separation by computing mean and 
    std deviation of SNRs in apertures in a ring at that sep, a la Mawet 2014 (see Fig 4).

    sciencecube, refcube, and templatecube are optional varaibles for supplying a previously constructed
    3d cube of images for doing the KLIP reduction on the science target, using the reference cube, with 
    fake signal injection provided by the templatecube.  If not provided, the pipeline will construct the necessary 
    cubes from the images listed in CleanList and the specified science star.  The template star by default is the
    same as the science star if not provided by user.

    Parameters
    -----------
    path : str
        dataset folder
    Star : 'A' or 'B'
        which star had the signal injection
    K_klip : int
        number of KLIP modes for the reduction
    sep : flt
        separation to compute SNR in ring at that sep.
    C : flt
        contrast of inject signal
    sepformat : str
        format of provided separations, either `lambda/D` or `pixels`.  Default = `lambda/D`
    sep_cutout_region, pa_cutout_region : tuple, tuple
        do not include apertures that fall within this sep/pa box in SNR calc.  Sep in lambda/D, pa in degrees
    returnsnrs : bool
        if True, return the SNRs array as well as the mean.  Default = False
    writeklip : bool
        if True, a fits file of the first injected signal KLIP-reduced image at each contrast and \
        separation will be written to disk.  Default = False
    update_prog : bool
        if True, display a progress bar for computing the SNR in the ring. Default = False, progress \
        bar within RunContrastCurveCalculation is the priority when running a full calculation
    sciencecube : 3d arr
        optional user-provided cube of psf images for doing the KLIP reduction. 
    refcube : 3d arr
        optional user-provided cube of reference psfs.
    templatecube : 3d arr
        optional user-provded cube of images to use as injected signal psf template.
    mask_core : bool
        if True, set all pixels within mask_radius to value of 0 in all images.  Default = True
    mask_radius : flt
        radius of inner mask in lambda/D or pixels.  Default = 1. lamdba/D
    mask_outer_annulus: bool
        if True, set all pixels outside the specified radius to zero.  Default = False
    outer_mask_radius : flt
        Set all pixels outside this radius to 0 if mask_outer_aunnulus set to True.
    subtract_radial_profile : bool
        If True, subtract the median radial profile from each image in the cubes.  Default = True.
    wavelength : flt
        central wavelength of filter band in microns.  Default = 3.9

    Returns
    -------
    flt
        mean SNR for that ring
    arr
        if returnsnrs = True, returns array of SNRs in apertures in the ring
    '''
    from cliotools.pca_skysub import update_progress
    # Define starting point pa:
    pa = 270.
    # Number of 1L/D apertures that can fit on the circumference at separation:
    Napers = np.floor(sep*2*np.pi)
    # Change in angle from one aper to the next:
    dTheta = 360/Napers
    # Create array around circumference, excluding the ones immediately before and after
    # where the planet is:
    pas = np.arange(pa+2*dTheta,pa+360-2*dTheta,dTheta)%360
    # Exclude cutout region if applicable:
    if sep >= sep_cutout_region[0] and sep <= sep_cutout_region[1]:
        cutouts = np.where(pas > pa_cutout_region[1])[0]
        cutouts = np.append(cutouts,np.where(pas < pa_cutout_region[0])[0])
        pas = pas[cutouts]
    # create empty container to store results:
    snrs = np.zeros(len(pas))
    # create synth cube with injected signal:
    boxsize = sciencecube.shape[1] * 0.5
    for i in range(len(pas)):
        if i == 0 and writeklip:
            do_writeklip = True
        else:
            do_writeklip = False
        snr, SynthCubeObject, SynthCubeObjectBDI = GetSNR(path, Star, K_klip, sep, pas[i], C, 
                boxsize = boxsize,
                sepformat = sepformat,
                returnsnrs = returnsnrs, writeklip = do_writeklip, update_prog = False, 
                sciencecube = sciencecube,
                refcube = refcube,
                templatecube = templatecube, 
                mask_core = mask_core, mask_outer_annulus = mask_outer_annulus, 
                mask_radius = mask_radius, outer_mask_radius = outer_mask_radius,
                normalize = normalize, normalizebymask = normalizebymask, normalizing_radius = normalizing_radius,
                subtract_radial_profile = subtract_radial_profile, wavelength = wavelength
                )
        snrs[i] = snr
        if update_prog:
            update_progress(i+1,len(pas))
        
    if returnsnrs:
        return np.mean(snrs), snrs
    
    return np.mean(snrs)


def getsnr(image, sep, pa, xc, yc, wavelength = 3.9, radius = 0.5, radius_format = 'lambda/D', return_signal_noise = False):
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
    return_signal_noise : bool
        if True, return SNR, signal with background subtracted, noise level, background level
    
    Returns:
    --------
    flt
        Signal-to-Noise ratio for given injected signal
            
    '''
    from cliotools.bditools import lod_to_pixels
    from photutils import CircularAperture, aperture_photometry
    radius = lod_to_pixels(radius, wavelength)
    lod = lod_to_pixels(1., wavelength)
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
    signal_without_bkgd = signal.copy()
    # compute mean background:
    bkgd = np.mean(noisesums)
    # Eqn 9 in Mawet 2014:
    signal = signal - bkgd
    snr = signal / ( noise * np.sqrt(1+ (1/np.size(pas))) )
    if return_signal_noise:
        return snr, signal_without_bkgd, noise, bkgd
    return snr


def mag(image, x, y, radius = 3.89245, returnflux = False, returntable = False):
    ''' Compute instrument magnitudes of one object.  Defaults are set to CLIO 3.9um optimal.

    Parameters:
    -----------
    image : 2d array
        science image
    x,y : flt
        x and y pixel location of center of star
    radius : flt
        pixel radius for aperture.  Default = 3.89, approx 1/2 L/D for 
        CLIO 3.9um 
    r_in, r_out : flt
        inside and outside radius for background annulus.  Default = 10,12
    returnflux : bool
        if true, return the instrument mag and the raw flux value.
    returntable : bool
        if true, return the entire photometry table.
    Returns:
    --------
    flt
        instrument magnitudes of source
    flt
        signal to noise ratio
    '''
    from photutils import CircularAperture, aperture_photometry
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
    flt
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
    2d arr
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
                 box = 70,
                 inject_negative_signal = False
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
    2d arr
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
        from cliotools.bditools import lod_to_pixels
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
    if inject_negative_signal:
        Planet = Planet * (-1)
    # account for integer pixel positions:
    if synth[ymin:ymax,xmin:xmax].shape != Planet.shape:
        try:
            synth[ymin:ymax+1,xmin:xmax] = synth[ymin:ymax+1,xmin:xmax] + (Planet)
        except:
            synth[ymin:ymax,xmin:xmax+1] = synth[ymin:ymax,xmin:xmax+1] + (Planet)
    else:
        synth[ymin:ymax,xmin:xmax] = synth[ymin:ymax,xmin:xmax] + (Planet)
    return synth

def injectplanets(image, imhdr, template, sep, pa, contrast, TC, xc, yc, inject_negative_signal = False, **kwargs):
    ''' Wrapper for injectplanet() that allows for multiple fake planets in one image.
        Parameters are same as injectplanet() except sep, pa, and contrast must all be
        arrays of the same length.  **kwargs are passed to injectplanet().
    '''
    from cliotools.bditools import injectplanet
    synth = image.copy()
    try:
        for i in range(len(sep)):
            synth1 = injectplanet(synth, imhdr, template, sep[i], pa[i], contrast[i], TC, xc, yc, 
                                      inject_negative_signal = inject_negative_signal,
                                      **kwargs)
            synth = synth1.copy()
    except:
        synth = injectplanet(synth, imhdr, template, sep, pa, contrast, TC, xc, yc, 
                                      inject_negative_signal = inject_negative_signal,
                                      **kwargs)
    return synth

class SyntheticSignal(object):
    def __init__(self, k, Star, sep, pa, C, sepformat = 'lambda/D', boxsize = 50,
                sciencecube = [], refcube = [], templatecube = [],
                template = [], TC = None, use_same = True, verbose = True,
                inject_negative_signal = False, wavelength = 3.9
                ):
        ''' Class for creating and controling images with synthetic point source signals ("planet") injected.

        Written by Logan A. Pearce, 2020
        Dependencies: numpy, scipy, pandas

        Attributes:
        -----------
        k : str
            pandas datafrom made from importing "CleanList"
        Star : 'A' or 'B'
            star to put the fake signal around
        sep : flt
            separation of planet placement in either arcsec, mas, pixels, or lambda/D [prefered]
        pa : flt
            position angle for planet placement in degrees from North
        C : flt
            desired contrast of planet with central object
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'lambda/D'
        boxsize : int
            size of box of size "2box x 2box" for image stamps, if cubes aren't supplied by user
        sciencecube : 3d arr
            optional user input the base images to use in injection for science images.  If not provided \
            script will generate them from files in CleanList and specified science star
        refcube : 3d arr
            optional user input the base images to use in injection for reference images in KLIP reduction.  If not provided \
            script will generate them from files in CleanList and specified science star
        templatecube : 3d arr
            user input the base images to use as psf template in creating the fake signal. If not provided \
            script will generate them from files in CleanList and specified science star.
        template : 2d arr
            optional user input for a psf template not built from the BDI dataset.
        TC : flt
            if external template provided, you must specify the contrast of the template relative to the science star
        use_same : bool
            If True, use the same star as a template for a synthetic psf signal around itself.  If false, use the opposite star. \
            Default = True
        verbose : bool
            If True, print status of things.  Default = True
        inject_negative_signal : bool
            If True, inject a negative planet signal instead of positive.  Default = False.

        '''
        from cliotools.bditools import contrast
        self.k = k
        self.Star = Star
        self.sep = sep
        self.pa = pa
        self.C = C
        self.sepformat = sepformat
        self.verbose = verbose
        # If no image cubes provided:
        if np.size(sciencecube) == 1:
            # Make image cubes without normalizing or masking:
            self.astamp, self.bstamp = PrepareCubes(self.k, 
                                                    boxsize = boxsize, 
                                                    normalize = False,
                                                    inner_mask_core = False,         
                                                    outer_mask_annulus = False,
                                                    verbose = self.verbose
                                                    )
            # If using the same star as the psf template:
            if use_same:
                if Star == 'A':
                    self.sciencecube = self.astamp.copy()
                    self.templatecube = self.astamp.copy()
                elif Star == 'B':
                    self.sciencecube = self.bstamp.copy()
                    self.templatecube = self.bstamp.copy()
            # else use the other star as psf template:
            else:
                if Star == 'A':
                    self.sciencecube = self.astamp.copy()
                    self.templatecube = self.bstamp.copy()
                elif Star == 'B':
                    self.sciencecube = self.bstamp.copy()
                    self.templatecube = self.astamp.copy()
            # Assign the opposite star as the reference set for KLIP reduction:
            if Star == 'A':
                self.refcube = self.bstamp.copy()
            elif Star == 'B':
                self.refcube = self.astamp.copy()
            box = boxsize
        # Else assign user supplied cubes:
        else:
            self.sciencecube = sciencecube
            self.refcube = refcube
            self.templatecube = templatecube
            box = templatecube.shape[1] / 2
        
        # Inject planet signal into science target star:
        from cliotools.bditools import injectplanets
        synthcube = np.zeros(np.shape(self.sciencecube))
        if len(templatecube) == 0:
            #print('not template provided')
            # If template PSF is not provided by user (this is most common):
            from cliotools.bditools import contrast
            # for each image in science cube:
            for i in range(self.sciencecube.shape[0]):
                # Get template constrast of refcube to sciencecube
                center = (0.5*((self.sciencecube.shape[2])-1),0.5*((self.sciencecube.shape[1])-1))
                TC = contrast(self.sciencecube[i],self.templatecube[i],center,center)
                # image header must be provided to 
                # accomodate rotation from north up reference got PA to image reference:
                imhdr = fits.getheader(self.k['filename'][i]) 
                # Inject the desired signal into the science cube:
                synth = injectplanets(self.sciencecube[i], imhdr, self.templatecube[i], sep, pa, C, TC, 
                                      center[0], center[1], 
                                      sepformat = self.sepformat, wavelength = wavelength, box = box, 
                                      inject_negative_signal = inject_negative_signal)
                # place signal-injected image into stack of images:
                synthcube[i,:,:] = synth
                
        else:
            center = (0.5*((self.sciencecube.shape[2])-1),0.5*((self.sciencecube.shape[1])-1))
            # If external template is provided: (this might happen if other star is saturated, etc)
            #if TC == None:
                # Known contrast of template to science star must be provided
                #raise ValueError('template contrast needed')
            # inject signal:
            for i in range(self.sciencecube.shape[0]):
                if TC == None:
                    from cliotools.bditools import contrast
                    # Get template constrast of refcube to sciencecube
                    TC = contrast(self.sciencecube[i],self.templatecube[i],center,center)
                imhdr = fits.getheader(k['filename'][i])
                synth = injectplanets(self.sciencecube[i], imhdr, self.templatecube[i], sep, pa, C, TC, box, box, 
                                              sepformat = sepformat, wavelength = wavelength, box = box, 
                                              inject_negative_signal = inject_negative_signal)
                synthcube[i,:,:] = synth
                
        self.synthcube = synthcube.copy()

################## Tools for estimating noise floor contrast ###################################

def makeskycube(path,x,y,k,box,lim_lod = 10, write_skycube = False):
    from cliotools.bditools import circle_mask
    from cliotools.bditools import lod_to_pixels
    xx,yy = np.meshgrid(np.arange(x-box,x+box+1,1),np.arange(y-box,y+box+1,1))
    lim = lod_to_pixels(lim_lod, 3.9)
    im = fits.getdata(k['filename'][0])
    count = 0
    keep = np.array([])
    im = fits.getdata(k['filename'][0])
    ndim = len(im.shape)
    if ndim == 2:
        skycube = np.zeros([len(k),box*2,box*2])
    if ndim == 3:
        s = im.shape[0]
        skycube = np.zeros([(len(k)+1)*s,box*2,box*2])
    for i in range(len(k)):
        if ndim == 2:
            tooclose = False
            im = fits.getdata(k['filename'][i])
            sky = im[y-box:y+box,x-box:x+box]
            xca,yca = k['xca'][i],k['yca'][i]
            xcb,ycb = k['xcb'][i],k['ycb'][i]
            ra = circle_mask(lim,im.shape[1],im.shape[0], xca, yca)
            rb = circle_mask(lim,im.shape[1],im.shape[0], xcb, ycb)
            skycube[count,:,:] = sky
            count += 1
        if ndim == 3:
            im = fits.getdata(k['filename'][i])
            for j in range(im.shape[0]):
                sky = im[j,y-box:y+box,x-box:x+box]
                xca,yca = k['xca'][i],k['yca'][i]
                xcb,ycb = k['xcb'][i],k['ycb'][i]
                ra = circle_mask(lim,im.shape[2],im.shape[1], xca, yca)
                rb = circle_mask(lim,im.shape[2],im.shape[1], xcb, ycb)
                skycube[count+j,:,:] = sky
            count += 1   
    #print(keep)
    #skycube = skycube[np.int_(keep)]
    if write_skycube:
        fits.writeto(path+'skycube.fits',skycube,overwrite = True)
    return skycube

def GetSingleSNRForNoiseFloor(path, k, x1, y1, x2, y2, K_klip, box, templatecube, C ,TC = 0,sep = 4, pa = 270., write_skycube = False,
                                skycube1 = None, skycube2 = None):
    from cliotools.bdi import BDI
    from cliotools.bditools import getsnr,injectplanets
    # Make two skycubes:
    if np.size(skycube1) == 1:
        skycube1 = makeskycube(path, x1,y1,k,box, write_skycube=write_skycube)
    if np.size(skycube2) == 1:
        skycube2 = makeskycube(path, x2,y2,k,box, write_skycube=write_skycube)
    
    smallest = np.min([skycube1.shape[0],templatecube.shape[0]])
    # inject fake signal into skycube1:
    synthcube = skycube1.copy()
    center = (0.5*((skycube1.shape[2])-1),0.5*((skycube1.shape[1])-1))
    for i in range(smallest):
        imhdr = fits.getheader(k['filename'][i])
        synthcube[i,:,:] = injectplanets(skycube1[i], imhdr, templatecube[i], sep, pa, C, TC, 
                                          center[0], center[1], box = box, wavelength = 3.9)
    # Make BDIObject and reduce:
    k = pd.read_csv(path+'CleanList', 
                         delim_whitespace = False,  # Spaces separate items in the list
                         comment = '#',             # Ignore commented rows
                        )
    BDIobject = BDI(k, path, K_klip = K_klip,
                    boxsize = box,
                    path_prefix = '',
                    normalize = False,
                    inner_mask_core = False,
                    outer_mask_annulus = False,    
                    subtract_radial_profile = False,
                    verbose = False,
                    acube = synthcube,
                    bcube = skycube2
                   )               
    BDIobject.Reduce(interp='bicubic',
                     rot_cval=0.,
                     mask_interp_overlapped_pixels = False
                    ) 
    # compute SNR:
    snr = getsnr(BDIobject.A_Reduced, sep, pa, center[0], center[1] , wavelength = 3.9)
    return snr, BDIobject

def GetSingleContrastSNRForNoiseFloor(path, k, x1, y1, x2, y2, K_klip, box, templatecube, C ,TC = 0,sep = 4, write_skycube = False,
                                        skycube1 = None, skycube2 = None):
    # Define starting point pa:
    pa = 270.
    # Number of 1L/D apertures that can fit on the circumference at separation:
    Napers = np.floor(sep*2*np.pi)
    # Change in angle from one aper to the next:
    dTheta = 360/Napers
    # Create array around circumference, excluding the ones immediately before and after
    # where the planet is:
    pas = np.arange(pa,pa+360-dTheta,dTheta)%360
    # create empty container to store results:
    snrs = np.zeros(len(pas))
    for i in range(len(pas)):
        snr, BDIobject2 = GetSingleSNRForNoiseFloor(path, k, x1, y1, x2, y2, K_klip, box, 
                                            templatecube, C ,TC = 0,sep = 4, pa = pas[i], write_skycube = write_skycube, skycube1 = skycube1, skycube2 = skycube2)
        snrs[i] = snr
    return np.mean(snrs)

def GetNoiseFloor(path, k, x1, y1, x2, y2, K_klip, box, templatecube, C, TC = 0,sep = 4, write_skycube = False, skycube1 = None, skycube2 = None):
    from cliotools.pca_skysub import update_progress
    snrs = np.zeros(len(C))
    for i in range(len(C)):
        snr1 = GetSingleContrastSNRForNoiseFloor(path, k, x1, y1, x2, y2, K_klip, box, templatecube, C[i], TC = 0,sep = 4, write_skycube = write_skycube,
                            skycube1 = skycube1, skycube2 = skycube2)
        snrs[i] = snr1
        update_progress(i+1,len(C))
    return snrs

def GetNoiseFloors(path, k, x1, y1, x2, y2, K_klip, box, templatecube, C, TC = 0,sep = 4, write_skycube = False, overwrite = False, 
                    skycube1 = None, skycube2 = None, filesuffix=''):
    from scipy import interpolate
    n = {}
    for Klip in K_klip:
        print('Testing KLIP modes:',Klip)
        snrs = GetNoiseFloor(path, k, x1, y1, x2, y2, np.array([Klip]), box, templatecube, C, TC = 0,sep = 4, write_skycube = write_skycube, 
                            skycube1 = skycube1, skycube2 = skycube2)
        pickle.dump(snrs,open(path+'NoiseFloorSNRS_Kklip'+str(Klip)+filesuffix+'.pkl','wb'))
        newC = np.linspace(np.min(C),np.max(C),100)
        f = interpolate.interp1d(C, snrs, fill_value='extrapolate')
        contrast = f(newC)
        ind = np.where(contrast <= 5.0)
        try:
            noise_floor = newC[ind][0]
        except:
            print('didnt fall below 5, skipping')
            continue
        n.update({Klip:noise_floor})
        pickle.dump(n,open(path+'NoiseFloors'+filesuffix+'.pkl','wb'))

def get_phoenix_model(model, wavelength_lim = None, DF = -8.0):
    ''' Open *.7 spectral file from https://phoenix.ens-lyon.fr/Grids/.  Explanataion of file
    at https://phoenix.ens-lyon.fr/Grids/FORMAT
    
    Args:
        model (str): path to model file
        wavelength_lim (flt): cut off wavelength in Ang at red end
        DF (flt): DF value for converting model to Ergs/sec/cm**2/A, from the "format" page.  DF=-8.0 for
            modern models
    Returns:
        pd datafram with columns 'Wavelength','Flux','BBFlux'; flux, BBflux in Ergs/sec/cm**2/A, wavelength in Ang
        
    '''
    t = pd.read_table(model, delim_whitespace=True, usecols=[0,1,2], skiprows=50000,
                     names=['Wavelength','Flux','BBFlux'])
    # convert from IDL double float precision to Python-ese:
    t['BBFlux'] = t['BBFlux'].str.replace('D','e')
    t['Flux'] = t['Flux'].str.replace('D','e')
    # Convert string to float and add DF:
    t['Flux'] = t['Flux'].astype(float) + DF
    t['BBFlux'] = t['BBFlux'].astype(float) + DF
    # sort dataframe by wavelength in case it is not sorted:
    t = t.sort_values(by=['Wavelength'])
    # convert wavelength to microns:
    #angstroms_to_um = 1/10000
    #t['Wavelength'] = t['Wavelength'] * angstroms_to_um
    # limit the output to desired wavelength range, because the model 
    # goes out to 100's of microns:
    if wavelength_lim:
        lim = np.where(t['Wavelength'] < wavelength_lim)[0][-1]
        t = t.loc[0:lim]
    return t

def GetMassLimits(path,reloadA,reloadB,m,models,spt,k,distance,age, interpflux = [], filesuffix = ''):
    d = distance
    ############# Filter zero point fluxes: ###################
    # the wavelengths of the filter bands:
    wavelengths = np.array([1.24,1.65,2.2,3.35,4.6]) #microns
    # Filter zero points:
    # 2MASS:
    f0J = ( 1594*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(1.235 * u.um)),
           27.8*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(1.235 * u.um)) )
    f0H = ( 1024*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(1.662 * u.um)),
           20.0*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(1.662 * u.um)) )
    f0K = ( 666.7*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(2.159 * u.um)),
           12.6*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(2.159 * u.um)) )
    # WISE:
    f0335 = 309.540 #Jy
    f0335 = f0335*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(3.35 * u.um))
    f046 = 171.787 # Jy
    f046 = f046*u.Jy.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(4.6 * u.um))
    #put in array:
    f0 = np.array([f0J[0],f0H[0],f0K[0],f0335,f046])
    
    ############# Convert app mags to fluxes: ###################
    # convert m to fluxes in those filters:
    fluxes = f0*10**(-m/2.5)
    
    ############# Interpolate to 3.9 microns flux using models: ###################
    if np.size(interpflux) == 0:
        interpflux = np.zeros(len(models))
        for i in range(len(models)):
            r = fits.open('../model_spectra/'+models[i])
            data = r[1].data
            ind = np.where((data['WAVELENGTH'] < 33500.1) & (data['WAVELENGTH'] > 33400) )[0]
            try:
                ind = ind[-1]
            except:
                pass
            scale_factor = fluxes[3]/(data['FLUX'][ind])
            scaledflux = data['FLUX']*scale_factor
            ind2 = np.where((data['WAVELENGTH'] < 39000.1) & (data['WAVELENGTH'] > 38950) )[0]
            try:
                ind2 = ind2[-1]
            except:
                pass
            interpflux[i] = scaledflux[ind2]
    
    ############# Compute primary's true flux: ##################
    primary_true_flux = np.mean(interpflux) # in physical units ergs s^-1 cm^-2 Ang^-1
    primary_true_flux_err = np.std(interpflux)
    
    ############# Convert to apparent magnitude: #############
    # open Vega's model:
    model = 'alpha_lyr_mod_004.fits'
    r = fits.open('../model_spectra/'+model)
    data = r[1].data
    ind = np.where((data['WAVELENGTH'] < 39000.1) & (data['WAVELENGTH'] > 38950) )[0]
    f_vega = data['FLUX'][ind]
    primary_app_mag = -2.5*np.log10(primary_true_flux/f_vega)[0]
    
    ############# Compute contrast of A relative to B in images: ############
    fivesigmacontrast = reloadA.fivesigmacontrast
    sep = reloadA.resep_au
    from cliotools.bditools import contrast
    cont = np.zeros(len(k))
    for i in range(len(k)):
        image = fits.getdata(k['filename'][i])
        if len(image.shape) == 2:
            cont[i] = contrast(image,image,[k['xca'][i],k['yca'][i]],[k['xcb'][i],k['ycb'][i]])
        elif len(image.shape) == 3:
            cont[i] = contrast(image[0],image[0],[k['xca'][i],k['yca'][i]],[k['xcb'][i],k['ycb'][i]])

    ABcontrast = np.mean(cont)
    
    ############# Apparent mag os object at the 5 sigma contrast limit for star A: ################
    fivesigmacontrastA = fivesigmacontrast
    fivesigma_app_mag = primary_app_mag + fivesigmacontrastA
    
    ########## Convert to absolute mag: ################
    fivesigma_abs_Mag = fivesigma_app_mag - 5*np.log10(d[0]) + 5
    
    ########## load BT Settl grids #####################
    # Load BT Settl grids:
    f = pd.read_table("../isochrones/model.BT-Settl.MKO.txt",header=3,delim_whitespace=True)
    # we want to find a mass by interpolating from our literature age value and
    # out just computed L' magnitudes:
    BTmass = f['M/Ms'].values
    BTage = f['t(Gyr)'].values
    BTL = f["L'"].values
    
    ########### Interpolate mass for age and L' mag: ###################
    from scipy.interpolate import griddata
    import pickle

    BTagearray = np.random.normal(age[0],age[1],100000)

    # For each abs magnitude value:
    fivesigma_mass_limit = np.zeros(len(fivesigma_abs_Mag))
    for i in range(len(fivesigma_abs_Mag)):
        # Generate an array of L' values around the mag value with error
        # (a placeholder for now, I haven't computed error accurately)
        BTLarray = np.random.normal(fivesigma_abs_Mag[i],0.1,100000)
        # interpolate masses from a grid of ages and L' mag:
        BTmassarray = griddata((BTage, BTL),BTmass, (BTagearray, BTLarray), method='linear')
        fivesigma_mass_limit[i] = np.nanmedian(BTmassarray)

    pickle.dump(fivesigma_mass_limit, open(path+'StarA_fivesigma_mass_limit_Kklip'+str(reloadA.K_klip)+filesuffix+'.pkl','wb'))
    
    ############## Repeat for B ##########################
    fivesigmacontrast = reloadB.fivesigmacontrast
    fivesigmacontrastB = fivesigmacontrast + ABcontrast
    # The apparent magnitude of an object at the 5 sigma constrast limit around B:
    fivesigma_app_mag = primary_app_mag + fivesigmacontrastB
    # distance modulus:
    fivesigma_abs_Mag = fivesigma_app_mag - 5*np.log10(d[0]) + 5
    
    # Interpolate masses:
    BTagearray = np.random.normal(age[0],age[1],100000)
    # For each abs magnitude value:
    fivesigma_mass_limit = np.zeros(len(fivesigma_abs_Mag))
    for i in range(len(fivesigma_abs_Mag)):
        # Generate an array of L' values around the mag value with error
        # (a placeholder for now, I haven't computed error accurately)
        BTLarray = np.random.normal(fivesigma_abs_Mag[i],0.1,100000)
        # interpolate masses from a grid of ages and L' mag:
        BTmassarray = griddata((BTage, BTL),BTmass, (BTagearray, BTLarray), method='linear')
        fivesigma_mass_limit[i] = np.nanmedian(BTmassarray)
    pickle.dump(fivesigma_mass_limit, open(path+'StarB_fivesigma_mass_limit_Kklip'+str(reloadB.K_klip)+filesuffix+'.pkl','wb'))
