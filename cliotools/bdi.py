from cliotools.bditools import *

class BDI(object):
    '''
    Class for preparing images for and performing BDI KLIP reduction.

    Assemble cubes of images and prepare them for KLIP reduction by: centering/subpixel-aligning images along
    vertical axis, normalizing images by dividing by sum of pixels in image, and masking the core of the central star.
    Written by Logan A. Pearce, 2020
    Dependencies: numpy, astropy.io.fits

    Attributes:
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
    verbose : bool
        if True, print status updates
    acube, bcube : 3d array
        Cube of aligned, normalized, and masked postage stamps of Star A and Star B
    '''

    def __init__(self, k, K_klip = 10,
                                boxsize = 50,                  
                                path_prefix = '',
                                normalize = True,
                                inner_mask_core = True,
                                inner_radius_format = 'pixels',
                                inner_mask_radius = 10.,
                                outer_mask_annulus = True,
                                outer_radius_format = 'pixels',
                                outer_mask_radius = 50.,
                                mask_cval = 0,
                                verbose = False
                                                ):
        self.k = k
        self.boxsize = boxsize
        self.K_klip = K_klip
        self.path_prefix = path_prefix
        self.normalize = normalize
        # inner mask:
        self.inner_mask_core = inner_mask_core
        if inner_radius_format == 'lambda/D':
            # convert to pixels:
            from cliotools.cliotools import lod_to_pixels
            radius = lod_to_pixels(inner_mask_radius, 3.9)
            self.inner_mask_radius = radius
        else:
            self.inner_mask_radius = inner_mask_radius
        # outer mask:
        self.outer_mask_annulus = outer_mask_annulus
        if outer_radius_format == 'lambda/D':
            # convert to pixels:
            from cliotools.cliotools import lod_to_pixels
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

        # Execute PrepareCubes on this object and store resulting cubes as attributes:
        self.acube, self.bcube = PrepareCubes(self.k, 
                                                boxsize = self.boxsize,                         # Define postage stamp size
                                                path_prefix = self.path_prefix,                 # Prepend path to images if necessary
                                                normalize = self.normalize,                     # Toggle normalize (Default = True)
                                                inner_mask_core = self.inner_mask_core,         # Toggle masking the star's core (Default = True)
                                                inner_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                inner_mask_radius = self.inner_mask_radius,     # Mask all pixels interior to this radius
                                                outer_mask_annulus = self.outer_mask_annulus,   # Toggle masking outer annulus (Default = False)
                                                outer_radius_format = 'pixels',                 # Mask radius defined in pixels (Default='pixels')
                                                outer_mask_radius = self.outer_mask_radius,     # Mask all pixels exterior to this radius
                                                cval = self.mask_cval,                          # Value to fill masked pixels
                                                verbose = self.verbose                          # If True, print status updates
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
            If using "bicubic" or "lanczos4" interpolations, then pixels along the edge of the
            masked regions will be interpolated with the 0 values in the mask.  For bicubic, this
            interpolation extends ~2 pixels into the data region, for lanczos4, ~3 pixels.  If set to
            True, extend the masked region in the reduced image to mask these interpolated data regions.
            Default = True
            
        Returns:
        --------
        Stores reduced image (or image cube if size(K_klip) > 1) as attributes A_Reduced and B_Reduced
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
            else:
                # If using other interp modes skip remasking
                pass

    
    def WriteToDisk(self, headercomment = None, outfilesuffix = '', write_directory = ''):
        if self.verbose:
            print('Writing finished cubes to file... done!')
        newhdr = psfsub_cube_header(self.k['filename'][0].split('/')[0], self.K_klip, 'A', self.A_Reduced.shape, self.acube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(write_directory+self.k['filename'][0].split('_')[0]+'_klipcube_a'+outfilesuffix+'.fit',self.A_Reduced,newhdr,overwrite=True)

        newhdr = psfsub_cube_header(self.k['filename'][0].split('/')[0], self.K_klip, 'B', self.B_Reduced.shape, self.bcube.shape[1])
        if headercomment:
            newhdr['COMMENT'] = headercomment
        fits.writeto(write_directory+self.k['filename'][0].split('_')[0]+'_klipcube_b'+outfilesuffix+'.fit',self.B_Reduced,newhdr,overwrite=True)