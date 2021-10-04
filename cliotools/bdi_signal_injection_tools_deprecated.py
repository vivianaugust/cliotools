#from cliotools.bditools import *
#from cliotools.bdi import *

from cliotools.bditools import PrepareCubes
import numpy as np
from astropy.io import fits
import pandas as pd


def GetSNR(path, Star, K_klip, sep, pa, C, boxsize = 50,
                sepformat = 'lambda/D',
                returnsnrs = False, writeklip = False, update_prog = False, 
                sciencecube = [],
                refcube = [],
                templatecube = [], 
                mask_core = True, mask_outer_annulus = True, mask_radius = 5., outer_mask_radius = 50., subtract_radial_profile = True,
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
                    normalize = True,              
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
    from cliotools.cliotools import lod_to_pixels
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
    from cliotools.bdi_signal_injection_tools import mag
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
    from cliotools.bdi_signal_injection_tools import makeplanet
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
    from cliotools.bdi_signal_injection_tools import injectplanet
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
        from cliotools.bdi_signal_injection_tools import contrast
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
        from cliotools.bdi_signal_injection_tools import injectplanets
        synthcube = np.zeros(np.shape(self.sciencecube))
        if len(templatecube) == 0:
            #print('not template provided')
            # If template PSF is not provided by user (this is most common):
            from cliotools.bdi_signal_injection_tools import contrast
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
                    from cliotools.bdi_signal_injection_tools import contrast
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
    from cliotools.cliotools import lod_to_pixels
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
    from cliotools.bdi_signal_injection_tools import getsnr,injectplanets
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
