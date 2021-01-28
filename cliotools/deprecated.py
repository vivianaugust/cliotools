##########################################################################
#              DEPRECATED                                                #

def DoInjection_deprecated(path, Star, K_klip, sep, pa, C,
                sepformat = 'lambda/D', box = 100, 
                template = [], TC = None, use_same = True, verbose = True,
                # Params for handling KLIP bases:
                returnZ = False, Z = [], immean = [],
                # Params for user supplied image cubes:
                returnstamps = False, sciencecube = [], refcube = [], templatecube = [],
                return_synthcube = False,
                # Params for preparing images for KLIP:
                normalize = True, normalizebymask = False,  normalize_radius = [],
                mask_core = True, mask_radius = 10.,
                mask_outer_annulus = True, outer_mask_radius = 50., outer_radius_format = 'pixels',cval = 0.0,
                interp = 'bicubic'
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
    
    box = templatecube.shape[1] / 2

    # Inject planet signal into science target star
    synthcube = np.zeros(np.shape(sciencecube))
    from cliotools.bditools import injectplanets
    if not len(template):
        # If template PSF is not provided by user (this is most common):
        from cliotools.bditools import contrast
        for i in range(sciencecube.shape[0]):
            # Get template constrast of refcube to sciencecube
            center = (0.5*((sciencecube.shape[2])-1),0.5*((sciencecube.shape[1])-1))
            TC = contrast(sciencecube[i],templatecube[i],center,center)
            # image header must be provided to 
            # accomodate rotation from north up reference got PA to image reference:
            imhdr = fits.getheader(k['filename'][i]) 
            # Inject the desired signal into the science cube:
            synth = injectplanets(sciencecube[i], imhdr, templatecube[i], sep, pa, C, TC, center[0], center[1], 
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
            synth = injectplanets(sciencecube[i], imhdr, template, sep, pa, C, TC, center[0], center[1], 
                                          sepformat = sepformat, wavelength = 3.9, box = box)
            synthcube[i,:,:] = synth
    
    
    # Normalize and mask both science target with injected signal, and reference star basis set:    
    from cliotools.bditools import normalize_cubes, mask_star_core
    if normalize:
        synthcube1,refcube1 = normalize_cubes(synthcube.copy(),refcube.copy(), normalizebymask = False)
    else:
        synthcube1,refcube1 = synthcube.copy(),refcube.copy()
    if mask_core:
        synthcube2,refcube2 = mask_star_core(synthcube1.copy(),refcube1.copy(), mask_radius, center[0], \
                                       center[1])
    else:
        synthcube2,refcube2 = synthcube1.copy(),refcube1.copy()
    if mask_outer_annulus:
        from cliotools.bditools import mask_outer
        # Set all pixels exterior to a radius of the center to cval:
        if not outer_mask_radius:
            raise ValueError('Outer radius must be specified if mask_outer_annulus == True')
        synthcube,refcube = mask_outer(synthcube2.copy(),refcube2.copy(), outer_mask_radius, center[0], center[1], radius_format = outer_radius_format, cval = cval)
    else:
        synthcube,refcube = synthcube2.copy(),refcube2.copy()

            
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
        Frot = rotate_clio(F[0], imhdr, center = None, interp = 'bicubic', bordermode = 'constant', cval = 0, scale = 1)
        klipcube[i,:,:] = Frot
    from astropy.stats import sigma_clip
    # Take sigma-clipped mean:
    kliped = np.mean(sigma_clip(klipcube, sigma = 3, axis = 0), axis = 0)
    
    if returnstamps and returnZ:
        return kliped, Z, immean, astamp, bstamp
    if returnstamps:
        return kliped, astamp, bstamp
    if returnZ:
        return kliped, Z, immean
    if return_synthcube:
        return kliped, synthcube, refcube
    return kliped


def DoSNR_deprecated(path, Star, K_klip, sep, C, sepformat = 'lambda/D', 
            box = 100, returnsnrs = False, writeklip = True, 
            verbose = False,  update_prog = True, use_same = True,
            sciencecube = [], refcube = [], templatecube = [],
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
    # Number of 1L/D apertures that can fit on the circumference at separation:
    Napers = np.floor(sep*2*np.pi)
    # Change in angle from one aper to the next:
    dTheta = 360/Napers
    # Create array around circumference, excluding the ones immediately before and after
    # where the planet is:
    pas = np.arange(pa+2*dTheta,pa+360-dTheta,dTheta)%360
    # create empty container to store results:
    snrs = np.zeros(len(pas))
    
    if np.size(sciencecube) == 1:
        # Do initial run on 1st pa location to create stamps and KLIP basis set:
        kliped, Z, immean, astamp, bstamp = DoInjection(path, Star, K_klip, sep, pa, C,  
                        sepformat = sepformat, box = box, 
                        verbose = verbose, returnZ = True, returnstamps = True, use_same = use_same, **kwargs)
    else:
        kliped, Z, immean = DoInjection(path, Star, K_klip, sep, pa, C, return_synthcube = False,
                                     returnZ = True,
                                     sciencecube = sciencecube,
                                     refcube = refcube,
                                     templatecube = templatecube,
                                     returnstamps = False,
                                     use_same = use_same, **kwargs)

        if Star == 'A':
            astamp = sciencecube.copy()
            bstamp = refcube.copy()
        elif Star == 'B':
            bstamp = sciencecube.copy()
            astamp = refcube.copy()

    # define center of box (location of star):
    xc,yc = (0.5*((astamp.shape[2])-1),0.5*((astamp.shape[1])-1))


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


def rotate_clio_ndimage(image, imhdr, **kwargs):
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


def contrast_curve_deprecated(path, Star, sep = np.arange(1,7,1), C = np.arange(3,7,0.2), curves_file = [],
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

def radial_subtraction_of_cube_deprecated(cube, exclude_r = 5., exclude_outer = 50., update_prog = True):
    ''' Subtract radial profile (median value in an annulus extending from the center) from each image
    in a cube of images.  Uses the radial profile script by Ian Crossfield found here:
    https://www.astrobetter.com/wiki/python_radial_profiles

    Parameters:
    ----------
    cube : 3d array
        cube of images
    exclude_r : flt
        Exclude pixels interior to this radius.  Default = 5
    exclude_outer : flt
        Exclude pixels exterior to this radius.  Default = 50
    update_prog : bool
        If True, show progress bar of applying subtraction to images in cube.  Default = True
        
    Returns:
    --------
    3d arr
        cube of radial profile subtracted images

    Written by Logan Pearce 2020, with functions adapted from Ian Crossfield 2010
    '''
    from cliotools.bditools import update_progress
    from cliotools.miscellany import radial_data
    Nimages = cube.shape[0]
    radsub = cube.copy()
    center = (cube[0].shape[1]/2, cube[0].shape[0]/2)
    for N in range(Nimages):
        RadialProfile = radial_data(cube[N]).median
        inds = np.arange(len(RadialProfile))
        # for each pixel:
        for i in range(cube[0].shape[0]):
            for j in range(cube[0].shape[1]):
                # compute distance of pixel from center:
                d = np.hypot(i - center[0], j - center[1])
                # if the pixel is outside the masked region, and 
                # within 50 pixels of center (the region we have radial prof for):
                if d > exclude_r and d < exclude_outer:
                    # Find the index of radial profile closest to that
                    # pixel's distance:
                    ind = (np.abs(d-inds)).argmin()
                    # get the value of median radial profile at that distance
                    # and subtract from the pixel's values
                    if not np.isnan(RadialProfile[ind]):
                        radsub[N][j,i] = cube[N][j,i] - RadialProfile[ind]
        if update_prog:
            update_progress(N+1,Nimages)
    return radsub

##########################################################################
#              Old PCA skysub functions                                  #
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

############# Sorting functions ##################

def beam_count(ims):
    '''Count the number of images in a dataset in each dither'''
    from astropy.io import fits
    count0, count1 = 0,0
    for i in ims:
        imhdr = fits.getheader(i)
        if imhdr['BEAM'] == 0:
            count0 += 1
        if imhdr['BEAM'] == 1:
            count1 += 1
    return count0,count1

def build_reference_stack(path, skip_list=False, K_klip = 10):
    '''Stack reference images into Nx512x1024 array for Nod 0 and Nod 1.
       Written by Logan A. Pearce, 2019
    
        Parameters:
        -----------
            path : str
                path to directory containing all the images of a single system on a single observing night
            skip_list : bool 
                Set to true to skip making a list of all fits files in the directory.  
                 Default = False
            K_klip : int
                use up to the Kth number of modes to reconstruct the image.  Default = 10
            
        Returns:
        --------
            sky0_stack, sky1_stack : Nx512x1024 array
                stack of reference images.
            K_klip : int
                if N < requested number of modes, return new value of K_klip where K_klip = min(sky0_stack.shape[0],sky1_stack.shape[0])
                otherwise returns requested number of modes.
    '''
    # Make a list of all images in the dataset:
    if skip_list == False:
        os.system('ls '+path+'*.fit > list')
    # Open the list:
    with open('list') as f:
        ims = f.read().splitlines()
    count0,count1 = beam_count(ims)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(ims[0]).shape
    
    print('Stacking reference images for',path.split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(ims)):
            # open the image
            image = fits.getdata(ims[i])
            imhdr = fits.getheader(ims[i])
            
            # Divide by number of coadds:
            if imhdr['COADDS'] != 1:
                image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
                
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
        for j in ims:
            # Open the image data:
            image = fits.getdata(j)
            imhdr = fits.getheader(j)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                sky0_stack[count0,:,:] = image
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                count1 += 1
    print('I will use ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    
    if sky0_stack.shape[0] < K_klip or sky1_stack.shape[0] < K_klip:
        print('Oops, there are fewer images than your requested number of modes.  Using all the images \
             in the reference set')
        K_klip = np.min([sky1_stack.shape[0],sky0_stack.shape[0]])
        print('K_klip =',K_klip)
        
    return sky0_stack, sky1_stack, K_klip

############### PCA fucntions ###################

def find_eigenimages(array, K_klip = 10):
    ''' Build a set (of size K_klip) of basis modes from the inputted reference images.
    Based on math in Soummer+ 2012 section 2.
       Written by Logan A. Pearce, 2019
    
    Parameters:
    -----------
        array : dxmxn array
            stack of reference images to build the eigenimage basis set.  
            Array must be dxmxn, where d = number of reference images, mxn = image shape.
        K_klip : int
            use up to the Kth number of modes to reconstruct the image.  Default = 10
        
    Returns:
    ---------
        Z (p x K_klip array): column vector array of K_klip number of basis modes, ordered
                by decreasing eigenvalue
    '''
    from scipy.linalg import eigh
    import numpy as np
    import time
    print('Building eigenimages...')
    start = time.time()
    shape = array.shape
    
    # Reshape references images to a set of d column vectors (Nxp) array, 
    # where p = width*height of images
    p = shape[1]*shape[2]
    R = np.reshape(array,(shape[0],p))
    # Compute the mean pixel value of each reference image:
    immean = np.nanmean(R, axis=1)
    # subtract mean:
    R = R - immean[:, None] #<- makes an empty first dimension to make
    # the vector math work out
    
    # compute covariance matrix of reference images:
    cov = np.cov(R)
    
    # compute eigenvalues (lambda) and corresponding eigenvectors (c)
    # of covariance matrix.  Compute only the eigenvalues/vectors up to the
    # desired number of bases K_klip.
    N = shape[0]
    lamb,c = eigh(cov, eigvals = (N-K_klip,N-1))
    
    # np.cov returns eigenvalues/vectors in increasing order, so
    # we need to reverse the order:
    index = np.flip(np.argsort(lamb))
    # sort corresponding eigenvalues:
    lamb = lamb[index]
    # and eigenvectors in order of descending eigenvalues:
    c = c.T
    c = c[index]
    
    # np.cov normalizes the covariance matrix by N-1.  We have to correct
    # for that because it's not in the Soummer 2012 equation:
    lamb = lamb * (p-1)
    
    # Take the dot product of the reference image with corresponding eigenvector:
    Z = np.dot(R.T, c.T)
    # Multiply by 1/sqrt(eigenvalue):
    Z = Z * np.sqrt(1/lamb)
    
    #print('time to compute Z',time.time()-time3)
    return Z

def build_estimator(T, Z,  K_klip = 10, numbasis = None):
    """ Build the estimated psf/sky by projecting the science
       target onto the basis modes.
       Based on math in Soummer 2012 section 4.
       Written by Logan A. Pearce, 2019
    
    Parameters:
    -----------
        imagepath : str
            path to science target image (fits file)
        Z : p x K_klip array
            column vector array of K_klip number of basis modes, ordered
            by decreasing eigenvalue; output from find_eigenimages
        K_klip : int
            max number of basis modes
        numbasis : int
            number of basis modes to use in building the estimator
            if numbasis = None, numbasis = K_klip and all computed modes
            will be used in sky reconstruction
        
    Returns:
    --------
        sky : mxn array
            reconstructed sky/psf estimator image.
    
    """
    from astropy.io import fits
    import numpy as np
    if numbasis == None:
        numbasis = K_klip
        
    # Import science target:
    shape=T.shape
    # Reshape science target into 1xp array:
    T_reshape = np.reshape(T,(shape[0]*shape[1]))
    # Subtract mean from science image:
    T_meansub = T_reshape - np.mean(T_reshape)
    # Make K_klip number of copies of science image
    # to use fast vectorized math:
    T_meansub = np.tile(T_meansub, (K_klip, 1))
    
    # Project the science images onto the basis modes:
    projection_sci_onto_basis = np.dot(T_meansub,Z)
    # This produces a (K_klip,K_klip) sized array of identical
    # rows of the projected science target.  We only need one row:
    projection_sci_onto_basis = projection_sci_onto_basis[0]

    # This is fancy math that let's you use fewer modes to subtract:
    lower_triangular = np.tril(np.ones([K_klip, K_klip]))
    projection_sci_onto_basis_tril = projection_sci_onto_basis * lower_triangular
    
    # Create the final estimator by multiplying by the basis modes:
    #sky = np.dot(projection_sci_onto_basis, Z.T)
    sky = np.dot(projection_sci_onto_basis_tril[numbasis-1,:], Z.T)
    sky = np.reshape(sky, (shape))
    return sky

################ Driver functions ##############################

def sky_subtract(rawimage, sky):
    """ Subtract estimated sky from one dither position
    from science image of opposite dither position.
    """
    from astropy.io import fits
    import numpy as np
    rawimage = rawimage - np.mean(rawimage)
    skysub_image = rawimage - sky
    return skysub_image

def skysub_single_image(filename, Z, K_klip):
    """Skysubtract a single coadded clio image (m x n)
       Parameters:
       -----------
       Filename : str
           path to target image
       Z : (K_klip x p) array
           stack of KLIP basis modes
       K_klip : int
           number of klip modes to use to build sky estimator
           
       Returns:
       --------
       skysub : (m x n) array
           skysubtracted image, mxn = original image dimensions
       sky : (m x n) array
           reconstructed sky estimator
    """
    from astropy.io import fits
    from cliotools.pcaskysub import build_estimator, sky_subtract
    T = fits.getdata(filename)
    sky = build_estimator(T, Z, K_klip = K_klip)
    skysub = sky_subtract(T, sky)
    return skysub, sky

def skysub_single_imagestack(filename, Z, K_klip, numstack):
    """Skysubtract a single clio imagestack (numstack x m x n)
       Written by Logan A. Pearce, 2019

       Parameters:
       -----------
       Filename : str
           path to target image
       Z : (K_klip x p) array
           stack of KLIP basis modes
       K_klip : int
           number of klip modes to use to build sky estimator
       numstack : int
           number of images in the image cube
           
       Returns:
       --------
       skysub : (numstack x m x n) array
           skysubtracted image, mxn = original image dimensions
       sky : (numstack x m x n) array
           reconstructed sky estimator
    """
    from astropy.io import fits
    from cliotools.pcaskysub import build_estimator, sky_subtract
    import numpy as np
    T = fits.getdata(filename)
    sky = np.array([build_estimator(T[i], Z, K_klip = K_klip) for i in range(numstack)])
    skysub = np.array([sky_subtract(T[i], sky[i]) for i in range(numstack)])
    return skysub, sky


def badpixelsub(image, imhdr, nan = False, dx = 1):
    """Replace known bad pixels with 0 or nan
       Written by Logan A. Pearce, 2019

       Parameters:
       -----------
       image : 2d array
           science image
       imhdr : fits header object
           fits header for the science image
       nan : bool
           Set to True to replace bad pixels with NaNs.  Otherwise they 
           are replaced with 0 values.
           
       Returns:
       --------
       im_badpixfix : 2d array
           science image

    """
    from astropy.io import fits
    import numpy as np
    import cliotools
    # Open correct bad pixel mask for the image size:
    if imhdr['NAXIS2'] == 512:
        # Open full frame pixel mask:
        badpixname = 'badpix_fullframe.fit'
    elif imhdr['NAXIS2'] == 300:
        # strip mode:
        badpixname = 'badpix_strip.fit'
    elif imhdr['NAXIS2'] == 200:
        # stamp mode:
        badpixname = 'badpix_stamp.fit'
    elif imhdr['NAXIS2'] == 50:
        # substamp mode:
        badpixname = 'badpix_substamp.fit'
    else:
        print("'Scuse me what?")

    try:
        # Get the bad pixel map:
        badpixels = fits.getdata(str(cliotools.__path__).replace('[','').replace(']','').replace("'",'')+'/CLIO_badpix/'+badpixname)
    except:
        # If the bad pixel maps aren't where expected, enter the correct path or retrieve the maps
        # from the zero wiki:
        print("Hmm I can't find the necessary bad pixel map. Shall I retrive it from the zero wiki?")
        yn = input("Type the path to the pixel masks, or type y to retrieve it.")
        if str(yn) == 'y':
            print('retrieving bad pixel map...')
            import requests
            url = 'https://magao-clio.github.io/zero-wiki/6d927/attachments/1242c/'+badpixname
            r = requests.get(url)
            with open(badpixname, 'wb') as f:
                f.write(r.content)
            print('Done.  Importing bad pixel map... done')
            badpixels = fits.getdata(badpixname)
        else:
            badpixels = fits.getdata(str(yn))
    im_badpixfix = image.copy()
    # Pull out the bad pixels in badpixel map:
    indicies = np.where(badpixels > 0)
    #for i in range(len(indicies[0])):
    #    x, y = indicies[1][i], indicies[0][i]
    #    im_badpixfix[y,x] = interp_pix(im_badpixfix, (x,y))
    #im_badpixfix[np.where(badpixels > 0)] = 0
    #im_badpixfix[indicies] = [np.nanmedian(im_badpixfix[indicies[0][i],np.arange(indicies[1][i]-dx,indicies[1][i]+dx+1,2)]) \
    #   for i in range(len(indicies[1]))]
    # Replace the bad pixels with interpolations from nearest neighbors:
    for i in range(len(indicies[1])):
        x, y = indicies[1][i], indicies[0][i]
        xarray = np.arange(x-dx,x+dx+1,2)
        xarray = xarray[np.where(xarray > 0)[0]]
        xarray = xarray[np.where(xarray < 1024)[0]]
        im_badpixfix[y,x] = np.nanmedian(image[y,xarray])

    return im_badpixfix

def clio_skysubtract(path, K_klip=5, skip_list = False, write_file = True, badpixelreplace = True, dx = 1):
    """Skysubtract an entire dataset
       Written by Logan A. Pearce, 2019
       Dependencies: numpy, astropy

       Parameters:
       -----------
       path : str
           path to science images including image prefixes and underscores.  
           ex: An image set of target BDI0933 with filenames of the form BDI0933__00xxx.fit
               would take as input a path string of 'BDI0933/BDI0933__'
       K_klip : int
           Number of basis modes to use in subtraction 
       skip_list : bool
           Set to True if a list of paths to science files has already been made.  List
           must be named "list"
       write_file : bool
           Set to False to skip writing the sky subtracted file to disk.  
       badpixelreplace : bool
           Set to False to skip replacing bad pixels
           
       Returns:
       --------
       If write_file = True, writes skysubtracted images to file with filename appended with 'skysub'
       ex: skysubtracted images of BDI0933__00xxx.fit are written to same directory
           as original with filename BDI0933__00xxx_skysub.fit
       If write_file = False, returns
          skysub : 2d array
              sky subtracted image
          imhdr : fits image header object
              original header object plus added comment noting sky subtraction

    """
    from astropy.io import fits
    import time
    import os
    import sys
    from cliotools.pcaskysub import update_progress, skysub_single_image, skysub_single_imagestack, \
        find_eigenimages, build_reference_stack, badpixelsub
    # Make a list of all images in the dataset, excluding any "cal" images:
    if skip_list == False:
        os.system('ls '+path+'0*.fit > list')
    # Open the list:
    with open('list') as f:
        ims = f.read().splitlines()
    # Build reference stack for each dither position:
    sky0_stack, sky1_stack, K_klip = build_reference_stack(path, K_klip = K_klip)
    Z0 = find_eigenimages(sky0_stack, K_klip = K_klip)
    Z1 = find_eigenimages(sky1_stack, K_klip = K_klip)
    # Open each image:
    count = 0
    print('Subtracting...')
    for i in range(len(ims)):
        #print('Subtracting ',ims[i])
        image = fits.getdata(ims[i])
        imhdr = fits.getheader(ims[i])
        shape = image.shape
        # Determine image dither:
        if imhdr['BEAM'] == 0:
            # Subtract opposite dither sky from image:
            if len(shape) == 3:
                skysub, sky = skysub_single_imagestack(ims[i], Z1, K_klip, shape[0])
            elif len(shape) == 2:
                skysub, sky = skysub_single_image(ims[i], Z1, K_klip)
        if imhdr['BEAM'] == 1:
            if len(shape) == 3:
                skysub, sky = skysub_single_imagestack(ims[i], Z0, K_klip, shape[0])
            elif len(shape) == 2:
                skysub, sky = skysub_single_image(ims[i], Z0, K_klip)
        # Replace known bad pixels with 0:
        if len(shape) == 2:
            skysub = badpixelsub(skysub, imhdr, dx=dx)
        elif len(shape) == 3:
            skysub = [badpixelsub(skysub[j], imhdr, dx=dx) for j in range(shape[0])]
        # Append the image header.
        imhdr['COMMENT'] = '    Sky subtracted and bad pixel corrected on '+time.strftime("%m/%d/%Y")+ ' By Logan A. Pearce'
        if write_file == True:
            # Write sky subtracted image to file.
            fits.writeto(str(ims[i]).split('.')[0]+'_skysub.fit',skysub,imhdr,overwrite=True)
        else:
            return skysub, imhdr
        count+=1
        update_progress(count,len(ims))
    os.system('rm list')
    print('Done.')

def clio_skysubtract_wskyframes(path, skyframepath, K_klip=5, skip_list = False, write_file = True, badpixelreplace = True):
    """Skysubtract an entire dataset using sky frames rather than opposite nods.
       Written by Logan A. Pearce, 2019
       Dependencies: numpy, astropy

       Parameters:
       -----------
       path : str
           path to science images including image prefixes and underscores.  
           ex: An image set of target BDI0933 with filenames of the form BDI0933__00xxx.fit
               would take as input a path string of 'BDI0933/BDI0933__'
       skyframepath : str
           path to sky frames including image prefixes and underscores.  
           ex: An image set of target BDI1350 with sky frame names of the form BDI1350sky_00xxx.fit
               would take as input a path string of 'BDI1350/BDI1350sky_'
       K_klip : int
           Number of basis modes to use in subtraction 
       skip_list : bool
           Set to True if a list of paths to science files has already been made.  List
           must be named "list"
       write_file : bool
           Set to False to skip writing the sky subtracted file to disk.  
       badpixelreplace : bool
           Set to False to skip replacing bad pixels
           
       Returns:
       --------
       If write_file = True, writes skysubtracted images to file with filename appended with 'skysub'
       ex: skysubtracted images of BDI0933__00xxx.fit are written to same directory
           as original with filename BDI0933__00xxx_skysub.fit
       If write_file = False, returns
          skysub : 2d array
              sky subtracted image
          imhdr : fits image header object
              original header object plus added comment noting sky subtraction

    """
    from astropy.io import fits
    import time
    import os
    import sys
    from cliotools.pcaskysub import update_progress, skysub_single_image, skysub_single_imagestack, \
        find_eigenimages, build_reference_stack
    # Make a list of all images in the dataset, excluding any "cal" images:
    if skip_list == False:
        os.system('ls '+path+'0*.fit > list')
    # Open the list:
    with open('list') as f:
        ims = f.read().splitlines()
    # Build reference stack for each dither position:
    # Skyframes have "Beam" = 0
    sky0_stack = build_reference_stack(skyframepath)
    Z0 = find_eigenimages(sky0_stack[0], K_klip = K_klip)
    # Open each image:
    count = 0
    print('Subtracting...')
    for i in range(len(ims)):
        #print('Subtracting ',ims[i])
        image = fits.getdata(ims[i])
        imhdr = fits.getheader(ims[i])
        shape = image.shape
        # Subtract opposite dither sky from image:
        if len(shape) == 3:
            skysub, sky = skysub_single_imagestack(ims[i], Z0, K_klip, shape[0])
        elif len(shape) == 2:
            skysub, sky = skysub_single_image(ims[i], Z0, K_klip)
        # Replace known bad pixels with 0:
        if len(shape) == 2:
            skysub = badpixelsub(skysub, imhdr)
        elif len(shape) == 3:
            skysub = [badpixelsub(skysub[j], imhdr) for j in range(shape[0])]
        # Append the image header.
        imhdr['COMMENT'] = '    Sky subtracted and bad pixel corrected on '+time.strftime("%m/%d/%Y")+ ' By Logan A. Pearce'
        if write_file == True:
            # Write sky subtracted image to file.
            fits.writeto(str(ims[i]).split('.')[0]+'_skysub.fit',skysub,imhdr,overwrite=True)
        else:
            return skysub, imhdr
        count+=1
        update_progress(count,len(ims))
    print('Done.')
    os.system('rm list')
