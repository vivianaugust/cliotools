from cliotools.bditools import *
#from cliotools.bditools import *
import pickle
import time
import astropy.units as u
import numpy as np
from astropy.io import fits

###################################################################################
#  BDI and performing KLIP reductions 


class BDI(object):
    def __init__(self, k, path, K_klip = 10,
                                boxsize = 50,                  
                                path_prefix = '',
                                normalize = True,
                                normalizebymask = False,          # If True, normalize using pix within radius
                                normalizing_radius = [],
                                inner_mask_core = True,
                                inner_radius_format = 'pixels',
                                inner_mask_radius = 10.,
                                outer_mask_annulus = True,
                                outer_radius_format = 'pixels',
                                outer_mask_radius = 50.,
                                mask_cval = 0,
                                subtract_radial_profile = True,
                                verbose = False,
                                acube = None,
                                bcube = None
                                                ):
        ''' Class for preparing images for and performing BDI KLIP reduction.

        Assemble cubes of images and prepare them for KLIP reduction by: centering/subpixel-aligning images along
        vertical axis, normalizing images by dividing by sum of pixels in image, and masking the core of the central star.
        Written by Logan A. Pearce, 2020
        Dependencies: numpy, astropy.io.fits

        Attributes:
        -----------
        path : str
            path to dataset
        k : Pandas array
            Pandas array made from the output of bditools.findstars_in_dataset.  
            Assumes column names are ['filename', 'xca','yca', 'xcb', 'ycb'].  If
            acube,bcube are supplied, this will be a dummy variable.
        boxsize : int
            size of box to draw around star psfs for DAOStarFinder
        path_prefix : int
            string to put in front of filenames in input file in case the relative
            location of files has changed.
        normalize : bool
            if True, normalize each image science and reference image integrated flux by dividing by each image
            by the sum of all pixels in image.  If False do not normalize images. Default = True
        inner_mask_core : bool
            if True, set all pixels within mask_radius to value of 0 in all images.  Default = True
        inner_mask_radius : flt
            all pixels interior to this radius set to 0. In lambda/D or pixels.  Default = 10 pixels
        outer_mask_annulus : bool
            If True, Set all pixels extrerior to a radius value to 0.  Default = True
        outer_mask_radius : flt
            Set all pixels extrerior to this radius to 0 if outer_mask_annulus set to True.  In lambda/D 
            or pixels.  Default = 50 pixels
        mask_cval : int or nan
            Set all masked pixels to this value.  Default = 0.  Warning: mask_cval = np.nan will cause
            problems in the reduction step.
        subtract_radial_profile : bool
            If True, subtract the median radial profile from each image in the cubes.  Default = True.
        verbose : bool
            if True, print status updates
        acube : 3d array
            Optional user input of cube of postage stamps of Star A.  Pay attention to if cubes have already been \
            normalized and masked yet; if so set those keywords to false.
        bcube : 3d array
            Optional user input of cube of postage stamps of Star B
        A_Reduced : 2d arr
            After running Reduce function, the KLIP reduced images for Star A are stored as this attribute.
        B_Reduced : 2d arr
            After running Reduce function, the KLIP reduced images for Star B are stored as this attribute.
        '''
        self.path = path
        self.k = k
        self.boxsize = boxsize
        self.K_klip = K_klip
        self.path_prefix = path_prefix
        self.normalize = normalize
        self.normalizebymask = normalizebymask
        self.normalizing_radius = normalizing_radius
        # inner mask:
        self.inner_mask_core = inner_mask_core
        if inner_radius_format == 'lambda/D':
            # convert to pixels:
            from cliotools.bditools import lod_to_pixels
            radius = lod_to_pixels(inner_mask_radius, 3.9)
            self.inner_mask_radius = radius
        else:
            self.inner_mask_radius = inner_mask_radius
        # outer mask:
        self.outer_mask_annulus = outer_mask_annulus
        if outer_radius_format == 'lambda/D':
            # convert to pixels:
            from cliotools.bditools import lod_to_pixels
            radius = lod_to_pixels(outer_mask_radius, 3.9)
            self.outer_mask_radius = radius
        else:
            self.outer_mask_radius = outer_mask_radius
        # value to fill masked pixels:
        self.mask_cval = mask_cval
        if np.isnan(self.mask_cval):
            import warnings
            warnings.warn('NaNs in images will cause KLIP reduction step to fail.')
        self.verbose = verbose
        # center of postagestamps
        self.center = (0.5*((2*self.boxsize)-1),0.5*((2*self.boxsize)-1))

        self.subtract_radial_profile = subtract_radial_profile

        if np.size(acube) == 1:
            # Execute PrepareCubes on this object and store resulting cubes as attributes:
            self.acube, self.bcube = PrepareCubes(self.k, 
                                                    boxsize = self.boxsize,                         # Define postage stamp size
                                                    path_prefix = self.path_prefix,                 # Prepend path to images if necessary
                                                    normalize = self.normalize,                     # Toggle normalize (Default = True)
                                                    normalizebymask = self.normalizebymask,          # If True, normalize using pix within radius
                                                    normalizing_radius = self.normalizing_radius,
                                                    inner_mask_core = self.inner_mask_core,         # Toggle masking the star's core (Default = True)
                                                    inner_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                    inner_mask_radius = self.inner_mask_radius,     # Mask all pixels interior to this radius
                                                    outer_mask_annulus = self.outer_mask_annulus,   # Toggle masking outer annulus (Default = False)
                                                    outer_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                    outer_mask_radius = self.outer_mask_radius,     # Mask all pixels exterior to this radius
                                                    cval = self.mask_cval,                          # Value to fill masked pixels
                                                    subtract_radial_profile = self.subtract_radial_profile, # Toggle subtract radial profile
                                                    verbose = self.verbose                          # If True, print status updates
                                                    )
        else:
            # Check 
            if acube.shape[1]*0.5 != self.boxsize:
                self.boxsize = acube.shape[1]*0.5
                print('Boxsize must be the same as supplied image cubes, changing boxsize attribute to ',acube.shape[1]*0.5)
            # Execute PrepareCubes using user-supplied aligned cubes on this object and store resulting cubes as attributes,
            # skipping the cutout and align step:
            self.acube, self.bcube = PrepareCubes(self.k, 
                                                    boxsize = self.boxsize,                         # Define postage stamp size
                                                    path_prefix = self.path_prefix,                 # Prepend path to images if necessary
                                                    normalize = self.normalize,                     # Toggle normalize (Default = True)
                                                    normalizebymask = self.normalizebymask,          # If True, normalize using pix within radius
                                                    normalizing_radius = self.normalizing_radius,
                                                    inner_mask_core = self.inner_mask_core,         # Toggle masking the star's core (Default = True)
                                                    inner_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                    inner_mask_radius = self.inner_mask_radius,     # Mask all pixels interior to this radius
                                                    outer_mask_annulus = self.outer_mask_annulus,   # Toggle masking outer annulus (Default = False)
                                                    outer_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                    outer_mask_radius = self.outer_mask_radius,     # Mask all pixels exterior to this radius
                                                    cval = self.mask_cval,                          # Value to fill masked pixels
                                                    subtract_radial_profile = self.subtract_radial_profile, # Toggle subtract radial profile
                                                    verbose = self.verbose,                          # If True, print status updates
                                                    acube = acube,                                  # Supply already cut out and aligned cube of images
                                                    bcube = bcube                                   # for A and B
                                                    )
        

    def Reduce(self, interp = 'bicubic', rot_cval = np.nan, mask_interp_overlapped_pixels = True):
        '''
        Wrapper function for performing KLIP reduction on BDI object.  Uses bditools.SubtractCubes

        SubtractCubes performs the BDI KLIP math.  For the cube of Star A, it builds a PCA basis set
        using the cube of Star B, then steps through the stack of images on Star A, projecting each image
        onto the basis set to build a PSF model, and then subtracting the reconstructed model from the
        original image.  The subtracted image is derotated to North Up East Left using OpenCV2 rotation function.  
        When all images in the cube have been subtracted, it takes a sigma-clipped mean along the vertical 
        axis as the final reduced image.  Then repeats for Star B, using Star A as basis set.

        If using "bicubic" or "lanczos4" interpolations, then pixels along the edge of the
        masked regions will be interpolated with the 0 values in the mask.  For bicubic, this
        interpolation extends ~2 pixels into the data region, for lanczos4, ~3 pixels.  Thus, the pixels
        that have been interpolated with the value 0 masked pixels will be corrupted, and we want to
        mask them further.  The mask_interp_overlapped_pixels keyword toggles this additinal masking. 
        
        Does not return anything; Stores reduced image (or image cube if size(K_klip) > 1) as attributes A_Reduced and B_Reduced

        Written by Logan A. Pearce, 2020
        Dependencies: numpy, astropy.io.fits, OpenCV

        Parameters:
        -----------
        interp : str
            Interpolation mode for OpenCV.  Either nearest, bilinear, bicubic, or lanczos4.
            Default = bicubic
        rot_cval : flt or nan
            fill value for rotated images.  Default = np.nan
        mask_interp_overlapped_pixels : bool
            If True, extend the masked region in the reduced image to mask these interpolated data regions. Default = True
        '''

        self.interp = interp
        # Perform KLIP reduction:
        self.A_Reduced,self.B_Reduced = SubtractCubes(self.acube,self.bcube,    # cubes from PrepareCubes
                                                     self.K_klip,               # number of KLIP modes
                                                     self.k,                    # list of images in order, need their headers for rotate
                                                     interp = self.interp,      # interpolation mode for rotate
                                                     rot_cval = rot_cval,       # fill value for extrapolated pixels in OpenCV
                                                     verbose = self.verbose     # if True print status updates
                                                     )

        # If bicubic or lanczos4 interp, need to mask pixels around edge of previous mask that have been
        # interpolated with the zeroes in the masked pixels.
        if mask_interp_overlapped_pixels:
            # store non-remasked images in case we want them again for some reason:
            self.A_Reduced_notremasked = self.A_Reduced.copy()
            self.B_Reduced_notremasked = self.B_Reduced.copy()
            # If size(K_klip) = 1, need to make image into 3d array because mask_star_core
            # expects it.
            if np.size(self.K_klip) == 1:
                A = np.array([self.A_Reduced.copy()])
                B = np.array([self.B_Reduced.copy()])
            else:
                # otherwise just make a copy:
                A = self.A_Reduced.copy()
                B = self.B_Reduced.copy()
            # Remasking only needed for bicubic or lanczos4 interp algorithms:
            if self.interp == 'bicubic' or self.interp == 'lanczos4':
                if self.interp == 'bicubic':
                    # overlap is ~2 pix for bicubic:
                    radius_buffer = 2
                elif self.interp == 'lanczos4':
                    # about 3 pix for lanczos4:
                    radius_buffer = 3
                # Use the mask_star_core function to mask a slightly larger region:
                A_Reduced_remask,B_Reduced_remask = mask_star_core(A,B, 
                                                                self.inner_mask_radius+radius_buffer, 
                                                                self.center[0], self.center[1], 
                                                                radius_format = 'pixels', 
                                                                cval = self.mask_cval)
                A = A_Reduced_remask.copy()
                B = B_Reduced_remask.copy()
                # Use mask_outer function to mask a slightly smaller region:
                A_Reduced_remask,B_Reduced_remask = mask_outer(A,B, 
                                                                self.outer_mask_radius-radius_buffer, 
                                                                self.center[0], self.center[1], 
                                                                radius_format = 'pixels', 
                                                                cval = self.mask_cval)
                # if only one KLIP mode, return a 2d image, otherwise 3d cube:
                if np.size(self.K_klip) == 1:
                    self.A_Reduced = A_Reduced_remask[0].copy()
                    self.B_Reduced = B_Reduced_remask[0].copy()
                else:
                    self.A_Reduced = A_Reduced_remask.copy()
                    self.B_Reduced = B_Reduced_remask.copy()
                # change the mask radius attributes:
                self.inner_mask_radius_final = self.inner_mask_radius+radius_buffer
                self.outer_mask_radius_final = self.outer_mask_radius-radius_buffer
            else:
                # If using other interp modes skip remasking
                pass

    
    def WriteToDisk(self, headercomment = None, outfilesuffix = '', write_directory = ''):
        ''' Writes KLIP reduced images or image cubes to disk.

        Written by Logan A. Pearce, 2020
        Dependencies: numpy, astropy.io.fits, OpenCV

        Parameters
        ----------
        headercomment : str
            Add comment to header of saved fits file.  Default = None
        outfilesuffix : str
            String to append to written filename.  Files written out have the filename 
            "[BDI prefix]_klipcube_a[outfilesuffix].fit" and "[BDI prefix]_klipcube_b[outfilesuffix].fit".
            Default = None
        write_directory : str
            Path to desired location for outfiles if other than current directory.  Default = None

        '''
        if np.size(self.K_klip) == 1:
            klipstring = str(self.K_klip)
        else:
            kk = [str(KK) for KK in self.K_klip]
            klipstring = '-'.join(kk)

        if self.verbose:
            print('Writing finished cubes to file... done!')
        newhdr = psfsub_cube_header(self.k['filename'][0].split('/')[0], self.K_klip, 'A', self.A_Reduced.shape, self.acube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(self.path+'/'+'A_klipcube_'+self.k['filename'][0].split('/')[0]+'_'+'box'+str(self.boxsize)+\
            '_Kklip'+klipstring+'_im'+str(int(self.inner_mask_radius))+'_om'+str(int(self.outer_mask_radius))+outfilesuffix+\
                '.fit',self.A_Reduced,newhdr,overwrite=True)

        newhdr = psfsub_cube_header(self.k['filename'][0].split('/')[0], self.K_klip, 'B', self.B_Reduced.shape, self.bcube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(self.path+'/'+'B_klipcube_'+self.k['filename'][0].split('/')[0]+'_'+'box'+str(self.boxsize)+\
            '_Kklip'+klipstring+'_im'+str(int(self.inner_mask_radius))+'_om'+str(int(self.outer_mask_radius))+outfilesuffix+\
                '.fit',self.B_Reduced,newhdr,overwrite=True)



###################################################################################
#  Computing Contrast Curves for a BDI data set

class ContrastCurve(object):
    def __init__(self, path, Star, K_klip, sep, C, box = 50, sepformat = 'lambda/D',
                 sciencecube = [], refcube = [], templatecube = [],
                 sep_cutout_region = [0,0], pa_cutout_region = [0,0],
                 normalize = True, normalizebymask = False, normalizing_radius = [],
                 mask_core = True, mask_outer_annulus = True,
                 mask_radius = 5., outer_mask_radius = 50.,
                 subtract_radial_profile = True,
                 save_results_filename = None, wavelength = 3.9
                ):
        ''' Class for computing SNR are a variety of separations and contrasts and plotting results

        Inject a vareity of fake signals from an array of separations and contrasts, and compute the signal-to-noise
        ratio at

        Written by Logan A. Pearce, 2020
        Dependencies: numpy, scipy, pandas

        Attributes:
        -----------
        path : str
            path to data set, directory must contain "CleanList"
        Star : 'A' or 'B'
            star to put the fake signal around
        K_klip : int
            number of KLIP modes to use in psf subtraction
        sep : flt
            separation of planet placement in either arcsec, mas, pixels, or lambda/D [prefered]
        C : flt
            desired contrast of planet with central object
        box : int
            size of box of size "2box x 2box" for image stamps, if cubes aren't supplied by user
        sepformat : str
            format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
            Default = 'lambda/D'
        sciencecub, refcube, templatecube : 3d arr
            user input the base images to use in injection for science images, psf templates, and KLIP reference basis sets. 
        snrs : 2d arr
            signal to noise ratios at the computed separation (columns) and contrasts (rows) locations
        runtime : flt
            time to perform computation
        wavelength : flt
            central wavelength of filter band in microns.  Default = 3.9
        '''
        
        self.path = path
        self.Star = Star
        self.K_klip = K_klip
        self.sep = sep
        self.C = C
        self.snrs = np.zeros((len(self.C),len(self.sep)))
        self.sciencecube = sciencecube
        self.refcube = refcube
        self.templatecube = templatecube
        self.sepformat = sepformat
        self.sep_cutout_region = sep_cutout_region
        self.pa_cutout_region = pa_cutout_region
        self.normalize = normalize
        self.normalizebymask = normalizebymask
        self.normalizing_radius = normalizing_radius
        self.inner_mask_radius = mask_radius
        self.outer_mask_radius = outer_mask_radius
        self.mask_core = mask_core
        self.mask_outer_annulus = mask_outer_annulus
        self.subtract_radial_profile = subtract_radial_profile
        self.save_results_filename = save_results_filename
        self.wavelength = wavelength
        if np.size(self.sciencecube) == 1:
            self.box = box
        else:
            self.box = self.sciencecube.shape[1]/2
        
        
    def RunContrastCurveCalculation(self, writeklip = False):
        ''' For the ContrastCurve object created, run the contrast curve calculation
        testing all the specified permutations.

        Parameters
        ----------
        writeklip : bool
            If True, for every sep/cont combo write the first synthetic image to disk. \
            Default = False

        '''
        from cliotools.bditools import DoSNR
        import time
        start = time.time()
        for j in range(len(self.sep)):
            print("Sep =",self.sep[j])
            for i in range(self.C.shape[0]):
                if np.size(self.sciencecube) == 1:
                    snr = DoSNR(self.path, self.Star, self.K_klip, self.sep[j], self.C[i], 
                                sepformat = self.sepformat, sep_cutout_region = self.sep_cutout_region, 
                                pa_cutout_region = self.pa_cutout_region,
                                box = self.box,
                                returnsnrs = False, writeklip = writeklip, update_prog = False,
                                mask_core = self.mask_core, 
                                mask_outer_annulus = self.mask_outer_annulus,
                                mask_radius = self.inner_mask_radius, outer_mask_radius = self.outer_mask_radius,
                                normalize = self.normalize, normalizebymask = self.normalizebymask, 
                                normalizing_radius = self.normalizing_radius,
                                subtract_radial_profile = self.subtract_radial_profile, wavelength = self.wavelength)
                else:
                    snr = DoSNR(self.path, self.Star, self.K_klip, self.sep[j], self.C[i], 
                                sepformat = self.sepformat,
                                returnsnrs = False, writeklip = writeklip, update_prog = False, 
                                sciencecube = self.sciencecube,
                                refcube = self.refcube,
                                templatecube = self.templatecube, mask_core = self.mask_core, 
                                mask_outer_annulus = self.mask_outer_annulus,
                                mask_radius = self.inner_mask_radius, outer_mask_radius = self.outer_mask_radius,
                                normalize = self.normalize, normalizebymask = self.normalizebymask, 
                                normalizing_radius = self.normalizing_radius,
                                subtract_radial_profile = self.subtract_radial_profile, wavelength = self.wavelength)
                self.snrs[i,j] = snr
                update_progress(i+1,len(self.C))
            # write out results every time a separation ring is completed:
            if self.save_results_filename:
                self.SaveResults(self.save_results_filename)
            else:
                self.SaveResults(self.path+'ContrastCurvesSNRs'+time.strftime("%Y.%m.%d.%H.%M.%S")+'.pkl')

        # write out final results:
        if self.save_results_filename:
            self.SaveResults(self.save_results_filename)
        else:
            self.SaveResults(self.path+'ContrastCurvesSNRs'+time.strftime("%Y.%m.%d.%H.%M.%S")+'.pkl')
        stop = time.time()
        self.runtime = (stop - start)*u.s
        
    def SaveResults(self, filename):
        ''' Save the orbits and orbital parameters attributes in a pickle file

        Parameters
        ----------
        filename : str 
            filename for pickle file

        Written by Logan Pearce, 2020
        '''
        pickle.dump(self.snrs, open( filename, "wb" ) )
        
    def LoadResults(self, filename, append = False):
        ''' Read in the snrs from a pickle file

        Parameters
        ----------
        filename : str
            filename of pickle file to load
        append : bool
            if True, append read in results to another ContrastCurves object.  Default = False.

        Written by Logan Pearce, 2020
        '''
        self.snrs = pickle.load( open( filename, "rb" ) )
        
    def ContrastCurvePlot(self, cmap = 'viridis', Ncontours=100, fontsize=15, plotstyle = 'default', 
                            load_snrs = False, filename = None, plot_cmap = True, sep_in_au = False, distance = None,
                            plot_mass_limits = False, planet_mass = []):
        ''' Plot constrast curves with colormap underneath

        Parameters
        ----------
        cmap : str 
            mpl colormap for shading
        Ncontours : int
            number of contours for colormap.  Default = 100
        fontsize : int 
            fontsize for plot labels. Default = 15
        plotstyle : str
            mpl plot style. Default = 'default'
        load_snrs : bool
            if no snrs already in object, load them here. Default = False
        filename :str 
            filename for snrs to load. Derfault = None
        plot_cmap : bool
            plot colormap under contours.  Default = True
        sep_in_au : bool
            if True, plot separation in AU instead of lambda/D
        distance : tuple, flt
            if sep_in_au = True, provide distance and error of system for the converstion from lambda/D to AU
        plot_mass_limits : bool
            if True, plot mass on Y-axis instead of magnitudes. Default = False
        planet_mass : arr
            if plotting mass limits, provide masses to plot on y-axis.

        Written by Logan Pearce, 2020
        '''
        if plot_mass_limits:
            self.planet_mass = planet_mass
        self.cmap = cmap
        if not hasattr(self, 'plotstyle'):
            self.plotstyle = plotstyle
        self.fontsize = fontsize
        self.plot_cmap = plot_cmap
        self.Ncontours = Ncontours
        from scipy import interpolate
        # load in snrs if they aren't already saved:
        if load_snrs:
            self.snrs = pickle.load( open( filename, "rb" ) )

        if not hasattr(self, 'newSNRs'):
            # regrid separation space to the same number of points as contrast
            # to make a square matrix:
            self.resep = np.linspace(np.min(self.sep),np.max(self.sep),len(self.C))
            # find which is the largest dimension:
            m = np.max([len(self.sep),len(self.C)])
            # create blank array for resampling SNRs:
            newSNRs = np.zeros((m,m))
            for i in range(m):
                # resample SNRs on new grid using scipy.interpolate:
                f = interpolate.interp1d(self.sep, self.snrs[i])
                newSNR = f(self.resep)
                # save in array:
                newSNRs[i] = newSNR
            self.newSNRs = newSNRs

        if sep_in_au: 
            if not hasattr(self, 'distance'):
                if distance == None:
                    raise ValueError('distance to system required to convert separation to AU')
                self.distance = distance
            from cliotools.bditools import lod_to_physical
            try:
                # if given as tuple:
                self.resep_au = lod_to_physical(self.resep, self.distance[0], 3.9)
            except:
                # if a single value:
                self.resep_au = lod_to_physical(self.resep, self.distance, 3.9)

        try:
            plt.style.use(self.plotstyle)
        except:
            plt.style.use('default')
        fig = plt.figure()

        if plot_mass_limits:
            if len(planet_mass) == 1:
                raise ValueError('Maasses for companions must be provided.')

        if sep_in_au:
            if plot_mass_limits:
                contours = plt.contour(self.resep_au,self.planet_mass,self.newSNRs, 5, colors='red',linestyles=(':',))
                # label contour line:
                plt.clabel(contours, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                contour = plt.contour(self.resep_au,self.planet_mass,self.newSNRs,levels = [5.0],
                            colors=('r',),linestyles=('-',),linewidths=(2,))
                # label these contour lines:
                plt.clabel(contour, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                if self.plot_cmap:
                    # plot a colormap underneath contour lines:
                    plt.contourf(self.resep_au,self.planet_mass,self.newSNRs,self.Ncontours,cmap=self.cmap)
                    plt.colorbar()
                plt.ylabel('Mass [M$\odot$]')
            else:
                # plot a thick contour line at SNR = 5:
                contours = plt.contour(self.resep_au,self.C,self.newSNRs, 5, colors='red',linestyles=(':',))
                plt.clabel(contours, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                # plot contour lines at intervals of 5:
                contour = plt.contour(self.resep_au,self.C,self.newSNRs,levels = [5.0],
                            colors=('r',),linestyles=('-',),linewidths=(2,))
                plt.clabel(contour, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                if self.plot_cmap:
                    # plot a colormap underneath contour lines:
                    plt.contourf(self.resep_au,self.C,self.newSNRs,self.Ncontours,cmap=self.cmap)
                    plt.colorbar()
                plt.ylabel('Contrast [mags]')
            plt.xlabel('Sep [AU]')
        else:
            if plot_mass_limits:
                contours = plt.contour(self.resep,planet_mass,self.newSNRs, 5, colors='red',linestyles=(':',))
                plt.clabel(contours, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                contour = plt.contour(self.resep,planet_mass,self.newSNRs,levels = [5.0],
                            colors=('r',),linestyles=('-',),linewidths=(2,))
                plt.clabel(contour, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                plt.ylabel('Mass [M$\odot$]')
            else:
                # plot a thick contour line at SNR = 5:
                contours = plt.contour(self.resep,self.C,self.newSNRs, 5, colors='red',linestyles=(':',))
                plt.clabel(contours, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                # plot contour lines at intervals of 5:
                contour = plt.contour(self.resep,self.C,self.newSNRs,levels = [5.0],
                            colors=('r',),linestyles=('-',),linewidths=(2,))
                plt.clabel(contour, inline=True, fontsize=self.fontsize,fmt='%1.0f')
                plt.ylabel('Contrast [mags]')

            if self.plot_cmap:
                # plot a colormap underneath contour lines:
                plt.contourf(self.resep,self.C,self.newSNRs,self.Ncontours,cmap=self.cmap)
                plt.colorbar()
            plt.xlabel(r'Sep [$\frac{\lambda}{D}$]')

        
        if not plot_mass_limits:
            plt.gca().invert_yaxis()
        plt.title(self.path.split('/')[0]+' '+self.Star)
        return fig

    def Compute5SigmaContrast(self, sep_in_au = True, distance = None):
        from scipy import interpolate
        if not hasattr(self, 'newSNRs'):
            self.resep = np.linspace(np.min(self.sep),np.max(self.sep),len(self.C))
            # find which is the largest dimension:
            m = np.max([len(self.sep),len(self.C)])
            # create blank array for resampling SNRs:
            newSNRs = np.zeros((m,m))
            for i in range(m):
                # resample SNRs on new grid using scipy.interpolate:
                f = interpolate.interp1d(self.sep, self.snrs[i])
                newSNR = f(self.resep)
                # save in array:
                newSNRs[i] = newSNR
            self.newSNRs = newSNRs

        if sep_in_au:
            if not hasattr(self, 'distance'):
                if distance == None:
                    raise ValueError('distance to system required to convert separation to AU')
                self.distance = distance
            from cliotools.bditools import lod_to_physical
            if distance == None:
                raise ValueError('distance to system required to convert separation to AU')
            self.distance = distance
            try:
                # if given as tuple:
                self.resep_au = lod_to_physical(self.resep, self.distance[0], 3.9)
            except:
                # if a single value:
                self.resep_au = lod_to_physical(self.resep, self.distance, 3.9)

        sigmaupper, sigmalower = np.zeros(len(self.resep)),np.zeros(len(self.resep))
        Cabove5sigmalimit,Cbelow5sigmalimit = np.zeros(len(self.resep)),np.zeros(len(self.resep))
        for i in range(len(self.resep)):
            # Collect sigma values above and below 5 sigma:
            sigmaupper[i] = self.newSNRs[:,i][[np.where(self.newSNRs[:,i] < 5.0)][0][0][0]-1]
            sigmalower[i] = self.newSNRs[:,i][[np.where(self.newSNRs[:,i] < 5.0)][0][0][0]]
            # Collect the contrast values at those locations:
            Cabove5sigmalimit[i] = self.C[[np.where(self.newSNRs[:,i] < 5.0)][0][0][0]-1]
            Cbelow5sigmalimit[i] = self.C[[np.where(self.newSNRs[:,i] < 5.0)][0][0][0]]
        # Now interpolate to get contrast at 5 sigma:
        fivesigmacontrast = np.zeros(len(self.resep))
        for i in range(len(self.resep)):
            f = interpolate.interp1d([sigmalower[i],sigmaupper[i]], [Cbelow5sigmalimit[i],Cabove5sigmalimit[i]],fill_value='extrapolate')
            fivesigmacontrast[i] = f(5.0)
        self.fivesigmacontrast = fivesigmacontrast

    def ContrastCurve5SigmaPlot(self, fontsize=15, plotstyle = 'default', 
                            yaxis_left = 'mag contrast', yaxis_right = 'mass limits',
                            xaxis_bottom = 'lambda/D', xaxis_top = 'AU',
                            sep_in_au = True, distance = None,
                            plot_mass_limits = False, fivesigma_mass_limit = [], legend = False,
                            plot_all = True, plot_noise_floor = False, noise_floor = [],
                            color='#8531E6', label=[], alpha = 1, save_plot = False, filename = []
                            ):
        ''' Plot just the 5-sigma contrast curve limit.

        Parameters
        ----------
        fontsize : int 
            fontsize for plot labels. Default = 15
        plotstyle : str
            mpl plot style. Default = 'default'
        sep_in_au : bool
            if True, plot separation in AU instead of lambda/D if not plotting them both on one plot.  Only used
            when plot_all = False
        distance : tuple, flt
            if sep_in_au = True, provide distance and error of system for the converstion from lambda/D to AU
        plot_mass_limits : bool
            if True, plot mass on Y-axis instead of magnitudes. Default = False
        fivesigma_mass_limit : arr
            if plotting mass limits, provide masses to plot on y-axis.
        xaxis_bottom, x_axis_top : str
            specify what to plot on top and bottom x axis. Either 'lambda/D','AU', or 'arcsec'
        yaxis_left, yaxis_right : str
            specify waht to plot on left and right y axis. Either 'mag contrast', 'flux contrast', or 'mass limits'.
        color : str
            color hexcode for line
        label : str
            label for plot legend.
        alpha : flt
            visibility of plotted curve line.
        '''
        if plot_mass_limits:
            self.fivesigma_mass_limit = fivesigma_mass_limit
        
        self.Compute5SigmaContrast(sep_in_au = sep_in_au, distance = distance)

        if not hasattr(self, 'plotstyle'):
            self.plotstyle = plotstyle
        try:
            plt.style.use(self.plotstyle)
        except:
            plt.style.use('default')

        fig, ax = plt.subplots()
        
        if plot_all:
            if yaxis_left == 'mag contrast':
                Y1 = self.fivesigmacontrast
                alpha1 = 0
                Y1_label = 'Contrast [3.9 mags]'
                ax.invert_yaxis()
            elif yaxis_left == 'flux contrast':
                Y1 = -self.fivesigmacontrast/2.5
                alpha1 = 0
                Y1_label = 'log(Contrast)'
                ax.invert_yaxis()
            elif yaxis_left == 'mass limits':
                Y1 = self.fivesigma_mass_limit
                alpha1 = alpha
                Y1_label = 'Mass [M$\odot$]'
            else:
                raise ValueError('Set "yaxis_bottom" and "yaxis_top" must be either "mass_limits", "flux contrast",\
                    or "mag constrast"')
                
            if yaxis_right == 'mag contrast':
                Y2 = self.fivesigmacontrast
                alpha2 = 0
                Y2_label = 'Contrast [3.9 mags]'
                ax2.invert_yaxis()
            elif yaxis_right == 'flux contrast':
                Y2 = -self.fivesigmacontrast/2.5
                alpha2 = 0
                Y2_label = 'log(Contrast)'
                ax2.invert_yaxis()
            elif yaxis_right == 'mass limits':
                Y2 = self.fivesigma_mass_limit
                alpha2 = alpha
                Y2_label = 'Mass [M$\odot$]'
            else:
                raise ValueError('Set "yaxis_bottom" and "yaxis_top" must be either "mass_limits", "flux contrast",\
                    or "mag constrast"')
                
            if xaxis_bottom == 'lambda/D':
                X1 = self.resep
                X1_label = r'Sep [$\frac{\lambda}{D}$]'
            elif xaxis_bottom == 'AU':
                X1 = self.resep_au
                X1_label = r'Sep [AU]'
            elif xaxis_bottom == 'arcsec':
                from cliotools.bditools import lod_to_arcsec
                X1 = lod_to_arcsec(self.resep)
                X1_label = r'Sep [arcsec]'
            else:
                raise ValueError('Set "xaxis_bottom" and "xaxis_top" must be either "lambda/D", "arcsec" or "AU"')
                
            if xaxis_top == 'lambda/D':
                X3 = self.resep
                X3_label = r'Sep [$\frac{\lambda}{D}$]'
            elif xaxis_top == 'AU':
                X3 = self.resep_au
                X3_label = r'Sep [AU]'
            elif xaxis_top == 'arcsec':
                from cliotools.bditools import lod_to_arcsec
                X3 = lod_to_arcsec(self.resep) 
                X3_label = r'Sep [arcsec]'
            else:
                raise ValueError('Set "xaxis_bottom" and "xaxis_top" must be either "lambda/D", "arcsec" or "AU"')
                
        if not plot_all:
            if plot_mass_limits:
                if len(self.fivesigma_mass_limit) == 1:
                    raise ValueError('5 sigma mass limit must be provided.')

            if sep_in_au:
                if plot_mass_limits:
                    ax.plot(self.resep_au,self.fivesigma_mass_limit,label=label)
                    ax.set_ylabel('Mass [M$\odot$]')
                else:
                    ax.plot(self.resep_au,self.fivesigmacontrast,label=label)
                    ax.set_ylabel('Contrast [mags]')
                ax.set_xlabel('Sep [AU]')
            else:
                if plot_mass_limits:
                    ax.plot(self.resep,self.fivesigma_mass_limit,label=label)
                    ax.set_ylabel('Mass [M$\odot$]')
                else:
                    ax.plot(self.resep,self.fivesigmacontrast,label=label)
                    ax.set_ylabel('Contrast [mags]')
                ax.set_xlabel(r'Sep [$\frac{\lambda}{D}$]')
            if not plot_mass_limits:
                ax.invert_yaxis()

        elif plot_all:
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            if legend:
                if len(label) == 0:
                    label = str(self.K_klip)
            if plot_noise_floor:
                if len(self.fivesigma_mass_limit) == 0:
                    raise ValueError('Noise floor must be provided.')
            if len(fivesigma_mass_limit) == 0:
                raise ValueError('5 sigma mass limit must be provided.')
            
            
            # make the plot:
            handle1, = ax.plot(X1,Y1,alpha = alpha1, color=color)
            ax.set_xlabel(X1_label)
            ax.set_ylabel(Y1_label)
            ax.grid(ls=':')
            #if yaxis_left == 'mag contrast' or yaxis_left == 'flux contrast':
                #print('invert yaxis left')
                #ax.invert_yaxis()

            handle2, = ax2.plot(X1,Y2,alpha = alpha2, color=color)
            ax2.set_ylabel(Y2_label)
            #if yaxis_right == 'mag contrast' or yaxis_right == 'flux contrast':
            #    print('invert yaxis right')
            #    ax2.invert_yaxis()

            ax3.plot(X3,Y1,alpha = 0, color=color)
            ax3.set_xlabel(X3_label)
            ax3.xaxis.labelpad = 10
            if plot_noise_floor:
                ax.plot(self.resep, [noise_floor[int(self.K_klip)]]*len(self.resep_au), color=color, alpha = 0.5, ls=':')
            if legend:
                if alpha1 == 0:
                    ax_handle = handle2
                if alpha2 == 0:
                    ax_handle = handle1
                plt.legend(handles = [ax_handle], labels=[label], handletextpad=0.2, fontsize = fontsize)
                
        ax.grid(ls=':')
        plt.tight_layout()

        if save_plot:
            plt.savefig(filename)

        return fig


    def ContrastCurve5SigmaPlotMassOrContrastOnly(self, fontsize=15, plotstyle = 'default', 
                            yaxis_left = 'mass limits',
                            xaxis_bottom = 'arcsec', xaxis_top = 'AU',
                            distance = None,  fivesigma_mass_limit = [], legend = False,
                            plot_noise_floor = False, noise_floor = [],
                            color='#8531E6', label=[], alpha = 1, save_plot = False, filename = [],
                            sep_in_au = True, legend_loc = 'upper right'
                            ):
        ''' Plot just the 5-sigma contrast curve limit.

        Parameters
        ----------
        fontsize : int 
            fontsize for plot labels. Default = 15
        plotstyle : str
            mpl plot style. Default = 'default'
        sep_in_au : bool
            if True, plot separation in AU instead of lambda/D if not plotting them both on one plot.  Only used
            when plot_all = False
        distance : tuple, flt
            if sep_in_au = True, provide distance and error of system for the converstion from lambda/D to AU
        plot_mass_limits : bool
            if True, plot mass on Y-axis instead of magnitudes. Default = False
        fivesigma_mass_limit : arr
            if plotting mass limits, provide masses to plot on y-axis.
        xaxis_bottom, x_axis_top : str
            specify what to plot on top and bottom x axis. Either 'lambda/D','AU', or 'arcsec'
        yaxis_left, yaxis_right : str
            specify waht to plot on left and right y axis. Either 'mag contrast', 'flux contrast', or 'mass limits'.
        color : str
            color hexcode for line
        label : str
            label for plot legend.
        alpha : flt
            visibility of plotted curve line.
        '''
        self.fivesigma_mass_limit = fivesigma_mass_limit
        
        self.Compute5SigmaContrast(sep_in_au = sep_in_au, distance = distance)

        if not hasattr(self, 'plotstyle'):
            self.plotstyle = plotstyle
        try:
            plt.style.use(self.plotstyle)
        except:
            plt.style.use('default')

        fig, ax = plt.subplots()

        if yaxis_left == 'mag contrast':
            Y1 = self.fivesigmacontrast
            Y1_label = 'Contrast [3.9 mags]'
            ax.invert_yaxis()
        elif yaxis_left == 'flux contrast':
            Y1 = -self.fivesigmacontrast/2.5
            Y1_label = r"log$_{10}$(L$^{'}$\, Flux\,Contrast)"
            #ax.invert_yaxis()
        elif yaxis_left == 'mass limits':
            Y1 = self.fivesigma_mass_limit
            Y1_label = 'Mass [M$\odot$]'
        else:
            raise ValueError('Set "yaxis_bottom" and "yaxis_top" must be either "mass_limits", "flux contrast",\
                or "mag constrast"')
            
        if xaxis_bottom == 'lambda/D':
            X1 = self.resep
            X1_label = r'Sep [$\frac{\lambda}{D}$]'
        elif xaxis_bottom == 'AU':
            X1 = self.resep_au
            X1_label = r'Sep [AU]'
        elif xaxis_bottom == 'arcsec':
            from cliotools.bditools import lod_to_arcsec
            X1 = lod_to_arcsec(self.resep)
            X1_label = r'Sep [arcsec]'
        else:
            raise ValueError('Set "xaxis_bottom" and "xaxis_top" must be either "lambda/D", "arcsec" or "AU"')
            
        if xaxis_top == 'lambda/D':
            X3 = self.resep
            X3_label = r'Sep [$\frac{\lambda}{D}$]'
        elif xaxis_top == 'AU':
            X3 = self.resep_au
            X3_label = r'Sep [AU]'
        elif xaxis_top == 'arcsec':
            from cliotools.bditools import lod_to_arcsec
            X3 = lod_to_arcsec(self.resep) 
            X3_label = r'Sep [arcsec]'
        else:
            raise ValueError('Set "xaxis_bottom" and "xaxis_top" must be either "lambda/D", "arcsec" or "AU"')

        ax3 = ax.twiny()
        if legend:
            if len(label) == 0:
                label = str(self.K_klip)+" KLIP Modes"
        if plot_noise_floor:
            if len(self.fivesigma_mass_limit) == 0:
                raise ValueError('Noise floor must be provided.')
        if len(fivesigma_mass_limit) == 0:
            raise ValueError('5 sigma mass limit must be provided.')
        
        # make the plot:
        handle1, = ax.plot(X1,Y1,alpha = alpha, color=color)
        ax.set_xlabel(X1_label)
        ax.set_ylabel(Y1_label)
        ax.grid(ls=':')

        ax3.plot(X3,Y1,alpha = 0, color=color)
        ax3.set_xlabel(X3_label)
        ax3.xaxis.labelpad = 10

        #ax3.xaxis.labelpad = 10
        if plot_noise_floor:
            ax.plot(self.resep, [noise_floor[int(self.K_klip)]]*len(self.resep_au), color=color, alpha = 0.5, ls=':')
        if legend:
            ax_handle = handle1
            plt.legend(handles = [ax_handle], labels=[label], handletextpad=0.2, fontsize = fontsize, loc = legend_loc)
                
        ax.grid(ls=':')
        plt.tight_layout()

        if save_plot:
            plt.savefig(filename)

        return fig