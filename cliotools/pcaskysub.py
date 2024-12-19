import numpy as np
from astropy.io import fits
import os
import pandas as pd

def update_progress(n,max_value):
    import sys
    import time
    import numpy as np
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(np.float64(n/max_value),decimals=2)
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

def circle_mask(radius, xsize, ysize, xc, yc):
    ''' For an image of size xsize x ysize, make a mask centered at (xc,yc) with radius = radius

    Args:
        radius (flt): radius of circle mask in pixels
        xsize (flt): x dimension of image
        ysize (flt): y dimension of image
        xc, yc (flt): x and y coordinates of center of circle mask
    Returns:
        arr: array of indicies in image that are within the desired circle
    '''
    xx,yy = np.meshgrid(np.arange(xsize)-xc,np.arange(ysize)-yc)
    r=np.hypot(xx,yy)
    return np.where(r<radius)

def diffraction_spike_mask(x,y,mask_image,rotation_angle = 37, xoffsets = [160,160,-160,-160], 
                           yoffsets = [-30,30,30,-30]):
    ''' Mask MagAO/CLIO diffraction spikes

    Args:
        x,y (flt): x,y position of center of star psf
        mask_image (2d arr): image to add the mask to
        rotation_angle (flt): rotation in deg from x-axis for diffraction spikes. Default = 37 deg
        xoffsets (1d list): x-positions of the four verticies of the unrotated rectangle to draw.
        yoffsets (1d list): y-positions of four verticies. 

    Returns:
        2d arr: mask containing diffraction spike mask. Masked pixels have value 1, unmasked 0

    '''
    import cv2
    # put into array:
    pos = np.array([xoffsets,yoffsets]).T
    # establish rotation matrix:
    t = np.radians(rotation_angle)
    R = np.array([[np.cos(t), np.sin(t)],[-np.sin(t), np.cos(t)]])
    # rotate each rectangle point:
    newpos = [np.matmul(pos[i],R) for i in range(len(pos))]
    newpos = np.stack(newpos,axis=0)
    # center at x,y point:
    newpos[:,0] = newpos[:,0]+x
    newpos[:,1] = newpos[:,1]+y
    # fill rectangle with ones:
    cv2.fillPoly(mask_image, np.array([newpos], dtype=np.int32), 1)
    return mask_image

def make_global_masks(path, shape, radius = 10, diffraction_spike_rotation_angle = 37, xoffsets = [160,160,-160,-160], 
                           yoffsets = [-30,30,30,-30], imlist = [], user_supplied_k = []):
    ''' For a complete raw dataset, make a mask of every star position in Nod 0 and Nod 1.  Must be run after Fidestars 
    and depends on the existence of the file ABLocations in the file path, and that the file path directory only contains
    the good images to skysubbed (i.e. all images commented out in ABLocations are removed from the directory.)

    Args:
        path (str): path to the dataset
        shape (2d arr): 2d image dimensions
        radius (int): radius in lambda/D for star circle masks. Defail = 10
        diffraction_spike_rotation_angle (flt): rotation angle in deg for diffraction spot masks. Default = 37
        xoffsets, yoffsets (lists): location of verticies of unrotated diffraction spike box. Changing location of \
            x verticies makes box longer/shorter, y vertices makes box thicker/thinner. Default: xoffsets = [160,160,-160,-160]\
            yoffsets = [-30,30,30,-30]
        imlist (str): if performing action on subset of images in a dataset, supply a list of paths to \
            image files as this variable
        user_supplied_k (pd dataframe): if performing action on subset of images in a dataset, supply the
            dataframe of images with the corresponding x/y star locations with this variable
    
    Returns:
        2d arr: mask for Nod 0 and mask for Nod 1

    '''
    from cliotools.cliotools import lod_to_pixels
    radius = lod_to_pixels(radius, 3.9)
    if len(imlist) == 0:
        # Make list of all fits images in folder:
        os.system('ls '+path+'*0*.fit > list')
        # Open the list:
        with open('list') as f:
            ims = f.read().splitlines()
    else:
        ims = imlist

    if len(user_supplied_k) == 0:
        # open the star locations list:
        k = pd.read_csv(path+'ABLocations.txt', 
                         delim_whitespace = True, # Spaces separate items in the list
                         comment = '#', # Ignore commented rows
                         names=['filename', 'xca','yca', 'xcb', 'ycb'] # specify column names
                        )
    else:
        k = user_supplied_k
    
    # Now make a global mask of all Nod 0 and Nod 1 star locations:
    mask0,mask1 = np.empty(shape),np.empty(shape)
    mask0[:],mask1[:] = np.nan, np.nan

    # for each image:
    for i in range(len(ims)):
        # open image and header:
        image = fits.getdata(ims[i])
        imhdr = fits.getheader(ims[i])
        if imhdr['BEAM'] == 1:
            # For Nod 1:
            # at star A, make a circle mask over the star:
            maskinda = circle_mask(radius, shape[1], shape[0],k['xca'][i],k['yca'][i])
            mask1[maskinda] = 1
            # And a mask over the diffraction spikes:
            mask1 = diffraction_spike_mask(k['xca'][i],k['yca'][i],mask1, 
                                           rotation_angle = diffraction_spike_rotation_angle,
                                          xoffsets = xoffsets, yoffsets = yoffsets)
            # repeat for star B:
            maskindb = circle_mask(radius, shape[1], shape[0],k['xcb'][i],k['ycb'][i])
            mask1[maskindb] = 1
            mask1 = diffraction_spike_mask(k['xcb'][i],k['ycb'][i],mask1,
                                           rotation_angle = diffraction_spike_rotation_angle,
                                          xoffsets = xoffsets, yoffsets = yoffsets)
        elif imhdr['BEAM'] == 0:
            # repeat for Nod 0:
            maskinda = circle_mask(radius, shape[1], shape[0],k['xca'][i],k['yca'][i])
            mask0[maskinda] = 1
            mask0 = diffraction_spike_mask(k['xca'][i],k['yca'][i],mask0,
                                           rotation_angle = diffraction_spike_rotation_angle,
                                          xoffsets = xoffsets, yoffsets = yoffsets)
            maskindb = circle_mask(radius, shape[1], shape[0],k['xcb'][i],k['ycb'][i])
            mask0[maskindb] = 1
            mask0 = diffraction_spike_mask(k['xcb'][i],k['ycb'][i],mask0,
                                           rotation_angle = diffraction_spike_rotation_angle,
                                          xoffsets = xoffsets, yoffsets = yoffsets)
            
        badpixname = 'badpix_fullframe.fit'
        badpixels = fits.getdata(badpixname)
        badpixels[np.where(badpixels==0)] = np.nan
        mask0[~np.isnan(badpixels)] = 1
        mask1[~np.isnan(badpixels)] = 1
    return mask0, mask1


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

def build_reference_stack(path, mask0, mask1, skip_list=False, K_klip = 5, imlist=[]):
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
        imlist : str
            if performing action on subset of images in a dataset, supply a list of paths to
            image files as this variable
        
    Returns:
    --------
        sky0_stack, sky1_stack : Nx512x1024 array
            stack of reference images.
        K_klip : int
            if N < requested number of modes, return new value of K_klip where 
            K_klip = min(sky0_stack.shape[0],sky1_stack.shape[0])
            otherwise returns requested number of modes.
    '''
    # Make a list of all images in the dataset:
    if skip_list == False:
        os.system('ls '+path+'*.fit > list')

    if len(imlist) == 0:
        # Make list of all fits images in folder:
        os.system('ls '+path+'*0*.fit > list')
        # Open the list:
        with open('list') as f:
            ims = f.read().splitlines()
    else:
        ims = imlist

    count0,count1 = beam_count(ims)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(ims[0]).shape
    
    print('Stacking reference images for',path.split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        sky0_stack_masked = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack_masked = np.zeros((count1*shape[0],shape[1],shape[2]))
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
                # Apply mask:
                image_masked = image.copy()
                for j in range(image.shape[0]):
                    image_masked[j,~np.isnan(mask0)] = 0
                # add to masked stack:
                sky0_stack_masked[count0:count0+image.shape[0],:,:] = image_masked
                # iterate counter
                count0 += shape[0]
            # If its nod 1:
            if imhdr['BEAM'] == 1:
                # Stack the image cube into one array
                sky1_stack[count1:count1+image.shape[0],:,:] = image
                image_masked = image.copy()
                for j in range(image.shape[0]):
                    image_masked[j,~np.isnan(mask1)] = 0
                sky1_stack_masked[count1:count1+image.shape[0],:,:] = image_masked
                count1 += shape[0]  
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        sky1_stack = np.zeros((count1,shape[0],shape[1]))
        sky0_stack = np.zeros((count0,shape[0],shape[1]))
        sky1_stack_masked = np.zeros((count1,shape[0],shape[1]))
        sky0_stack_masked = np.zeros((count0,shape[0],shape[1]))
        count0, count1 = 0,0
        for j in ims:
            # Open the image data:
            image = fits.getdata(j)
            imhdr = fits.getheader(j)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                # add to unmasked stack:
                sky0_stack[count0,:,:] = image
                # Apply mask:
                image[~np.isnan(mask0)] = 0
                # add to masked stack:
                sky0_stack_masked[count0,:,:] = image
                # iterate counter
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                # Apply mask:
                image[~np.isnan(mask1)] = 0
                # add to masked stack:
                sky1_stack_masked[count1,:,:] = image
                count1 += 1
    print('I will use ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    
    if sky0_stack.shape[0] < K_klip or sky1_stack.shape[0] < K_klip:
        print('Oops, there are fewer images than your requested number of modes.  Using all the images \
             in the reference set')
        K_klip = np.min([sky1_stack.shape[0],sky0_stack.shape[0]])
        print('K_klip =',K_klip)
        
    return sky0_stack, sky0_stack_masked, sky1_stack, sky1_stack_masked, K_klip

def build_reference_skyframe_stack(path, skip_list=False, K_klip = 5):
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
                if N < requested number of modes, return new value of K_klip where 
                K_klip = min(sky0_stack.shape[0],sky1_stack.shape[0])
                otherwise returns requested number of modes.
    '''
    # Make a list of all images in the dataset:
    if skip_list == False:
        os.system('ls '+path+'*.fit > list')
        os.system('ls '+path+'skyframes/*.fit > skylist')
    # Open the list:
    with open('list') as f:
        ims = f.read().splitlines()
    with open('skylist') as f:
        skyims = f.read().splitlines()
    count0,count1 = beam_count(ims)
    skycount0,skycount1 = beam_count(skyims)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(ims[0]).shape
    
    print('Stacking reference images for',path.split('/')[0],'...')
    # Each image is a single frame composite of several coadds.
    sky0_stack = np.zeros((count0,shape[0],shape[1]))
    sky_stack = np.zeros((skycount0,shape[0],shape[1]))
    count0, skycount0 = 0,0
    for j in ims:
        # Open the image data:
        image = fits.getdata(j)
        imhdr = fits.getheader(j)
        # Divide by number of coadds:
        image = image/imhdr['COADDS']
        # Stack the image in the appropriate stack:
        # add to unmasked stack:
        sky0_stack[count0,:,:] = image
        # iterate counter
        count0 += 1
        
    for j in skyims:
        # Open the image data:
        image = fits.getdata(j)
        imhdr = fits.getheader(j)
        # Divide by number of coadds:
        image = image/imhdr['COADDS']
        # Stack the image in the appropriate stack:
        # add to unmasked stack:
        sky_stack[skycount0,:,:] = image
        # iterate counter
        skycount0 += 1
    print('I will use ',sky0_stack.shape[0],' images for Nod 0, and ',sky_stack.shape[0],'skyframes')
    
    if sky0_stack.shape[0] < K_klip:
        print('Oops, there are fewer images than your requested number of modes.  Using all the images \
             in the reference set')
        K_klip = sky0_stack.shape[0]
        print('K_klip =',K_klip)
        
    return sky0_stack, sky_stack, K_klip

def find_eigenimages(array, K_klip = 5):
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
    # Make the mean image:
    immean = np.nanmean(R, axis=0)
    # Subtract mean image from each reference image:
    R_meansub = R - immean[None,:]#<- makes an empty first dimension to make
    # the vector math work out
    
    # compute covariance matrix of reference images:
    cov = np.cov(R_meansub)
    
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
    return Z, np.reshape(immean,(shape[1],shape[2]))

def build_estimator(T, Z, immean, mask, K_klip = 5, numbasis = None):
    """ Build the estimated psf/sky by projecting the science
    target onto the basis modes.
    Based on math in Soummer 2012 section 4.
    Written by Logan A. Pearce, 2019

    Parameters:
    -----------
        T : arr
            science target image (fits file)
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
    # Apply mask:
    T_masked = T.copy()
    T_masked[~np.isnan(mask)] = 0
    # Subtract mean from science image:
    T_meansub = T_masked - immean + 230
    T_meansub[~np.isnan(mask)] = 0
    # Reshape science target into 1xp array:
    T_reshape = np.reshape(T_meansub,(shape[0]*shape[1]))
    # Make K_klip number of copies of science image
    # to use fast vectorized math:
    T_tiled = np.tile(T_reshape, (K_klip, 1))
    
    # Project the science images onto the basis modes:
    projection_sci_onto_basis = np.dot(T_tiled,Z)
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

def fix_pixel(imagesr, pixel, dx = 5):
    ''' Replace bad pixels with median of dx nearest neighbors
    along x direction

    Parameters:
    -----------
    imagesr : 2d array
        stack of images to fix
    badpixr : 2d array
        list of bad pixels, output of findbadpix
    dx : int
        how many pixels to look to right and left to
        get median replacement value.  Should be odd.

    Returns:
    --------
    imfix : 3d array
        stack of bad pixel corrected images.
    '''
    x, y = pixel[0],pixel[1]
    xarray = np.arange(x-dx,x+dx+1,2)
    xarray = xarray[np.where(xarray > 0)[0]]
    xarray = xarray[np.where(xarray < 1024)[0]]
    new_pix_value = np.nanmedian(imagesr[y,xarray])
    return new_pix_value

def badpixfix(image, badpixels, xca, yca, xcb, ycb, dx=5):
    ''' Fix all the bad pixels in the provided bad pixels map that are within the specified
    area from the stars of interest. Fixes pixels by interpolating over pixels to the right 
    and left (along x-axis) from pixel of interest.

    Args:
        image (2d arr): image to be fixed
        badpixels (2d arr): map of bad pixels where "good" pixels are marked as nan
        xca, yca (flt): x, y location of star A
        xcb, ycb (flt): x, y location of star B
        dx (int): how many pixels to the side should be used for interpolation. Should be odd.
    '''
    imfix = image.copy()

    # Create a box that covers the entire image
    boxsize_y, boxsize_x = image.shape
    boxa = np.array([0, boxsize_y, 0, boxsize_x])
    boxb = np.array([0, boxsize_y, 0, boxsize_x])
    
    badpix_list = np.where(~np.isnan(badpixels))
    
    for j in range(len(badpix_list[0])):
        x = badpix_list[1][j]
        y = badpix_list[0][j]
        
        if 0 <= y < boxsize_y and 0 <= x < boxsize_x:
            if (boxa[0] <= y <= boxa[1] and boxa[2] <= x <= boxa[3]) or (boxb[0] <= y <= boxb[1] and boxb[2] <= x <= boxb[3]):
                imfix[y, x] = fix_pixel(image, (x, y))
    
    return imfix


def do_reduction(imagefile, Z, Z_mean, ndim, mask, star_locations, badpixels, K_klip = 5, verbose = False):
    # Open target image:
    if verbose:
        print('Opening image')
    target_image = fits.getdata(imagefile)
    target_image_hdr = fits.getheader(imagefile)
    # Divide by number of coadds:
    target_image = target_image/target_image_hdr['COADDS']
    target_image_masked = target_image.copy()
    
    if ndim == 3:
        skysubbed = target_image.copy()
        for j in range(target_image.shape[0]):
            if verbose:
                print('Building estimator')
            # Build sky estimator:
            estimated_sky = build_estimator(target_image[j], Z, Z_mean, mask, K_klip = K_klip, numbasis = None)
            # Mask opposite nod stars in target image:
            if verbose:
                print('Subtracting sky')
            target_image_masked[j][~np.isnan(mask)] = 0
            delta_mean = np.median(target_image_masked[j]) - np.median(Z_mean)
            meansubbed = target_image_masked[j] - (Z_mean)
            meansubbed[~np.isnan(mask)] = 0
            skysub = meansubbed - estimated_sky
            skysub = skysub - np.median(skysub[np.isnan(mask)])
            skysub[~np.isnan(mask)] = 0
            skysubbed[j] = skysub
            
        badpixfixed = skysubbed.copy()
        for j in range(target_image.shape[0]):
            badpixfixed[j] = badpixfix(skysubbed[j],badpixels,*star_locations, dx = 5)
            

    elif ndim == 2:
        # Build sky estimator:
        if verbose:
            print('Building estimator')
        estimated_sky = build_estimator(target_image, Z, Z_mean, mask, K_klip = K_klip, numbasis = None)
        if verbose:
            print('Subtracting sky')
        target_image_masked = target_image.copy()
        target_image_masked[~np.isnan(mask)] = 0
        delta_mean = np.median(target_image_masked) - np.median(Z_mean)
        meansubbed = target_image_masked - (Z_mean)
        meansubbed[~np.isnan(mask)] = 0
        skysubbed = meansubbed - estimated_sky
        skysubbed = skysubbed - np.median(skysubbed[np.isnan(mask)])
        skysubbed[~np.isnan(mask)] = 0
        badpixfixed = skysubbed.copy()
        badpixfixed = badpixfix(badpixfixed,badpixels,*star_locations, dx = 5)
        
    return badpixfixed, skysubbed, estimated_sky

#################### 4 More thorough bad pix fix #########################

##################### Bad pixel find and fix functions ###################
# pca_skysub uses the established CLIO bad pixel maps.  These functions #
# do an independent identification of bad pixels and replaces them with  #
# the median of their n neighbors along the x-direction, because CLIO    #
# seems to show patterns in columns.                                     #
# I am using the output of bditools.findstars because that file contains #
# a list of the good images in the dataset, the ones with useable star   #
# images, and won't waste time on images that won't be used in the       #
# analysis later 

def raw_beam_count(k):
    '''Count the number of images in a dataset in each dither'''
    from astropy.io import fits
    count0, count1 = 0,0
  
    for i in range(len(k)):
        sp = k['filename'][i].split(',')
        filename = sp[0]
        imhdr = fits.getheader(filename)
        if imhdr['BEAM'] == 0:
            count0 += 1
        if imhdr['BEAM'] == 1:
            count1 += 1
    return count0,count1

def build_raw_stack(k):
    '''Stack all raw images into Nx512x1024 array for Nod 0 and Nod 1.
       Requires the pandas dataframe made from the file "ABLocations" 
       from bditools.findstars output
       Written by Logan A. Pearce, 2020
    
        Parameters:
        -----------
            k : pandas dataframe
                list of skysubbed files and x/y star locations
                output from bditools.findstars
            
        Returns:
        --------
            stack0, stack1 : Nx512x1024 array
                stack of raw images in nod 0 and nod 1.
            xca0, yca0, xcb0, ycb0, xca1, yca1, xcb1, ycb1 : 1d arrays
                list of star A and B pixel locations in the stack of
                nod 0 and nod 1

    '''
    from cliotools.pcaskysub import raw_beam_count
    count0,count1 = raw_beam_count(k)

    sp = k['filename'][0].split(',')
    filename = sp[0]
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(filename).shape
    xca0, yca0, xcb0, ycb0 = [], [], [], []
    xca1, yca1, xcb1, ycb1 = [], [], [], []

    sp = k['filename'][0].split(',')
    filename = sp[0]
    
    print('Stacking reference images for',filename.split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
    
        for i in range(len(k)):
            sp = k['filename'][i].split(',')
            filename = sp[0]
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            
            # Divide by number of coadds:
            #if imhdr['COADDS'] != 1:
             #   image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
                
            # If its nod 0:
            if imhdr['BEAM'] == 0:
                # Stack the image cube into one array
                sky0_stack[count0:count0+image.shape[0],:,:] = image
                xca0.append(k['xca'][i])
                yca0.append(k['yca'][i]) 
                xcb0.append(k['xcb'][i]) 
                ycb0.append(k['ycb'][i])
                count0 += shape[0]
            # If its nod 1:
            if imhdr['BEAM'] == 1:
                # Stack the image cube into one array
                sky1_stack[count1:count1+image.shape[0],:,:] = image
                xca1.append(k['xca'][i])
                yca1.append(k['yca'][i]) 
                xcb1.append(k['xcb'][i]) 
                ycb1.append(k['ycb'][i])
                count1 += shape[0]  
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        sky1_stack = np.zeros((count1,shape[0],shape[1]))
        sky0_stack = np.zeros((count0,shape[0],shape[1]))
        count0, count1 = 0,0
        for i in range(len(k)):
            sp = k['filename'][i].split(',')
            filename = sp[0]            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                sky0_stack[count0,:,:] = image
                xca0.append(k['xca'][i])
                yca0.append(k['yca'][i]) 
                xcb0.append(k['xcb'][i]) 
                ycb0.append(k['ycb'][i])
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                xca1.append(k['xca'][i])
                yca1.append(k['yca'][i]) 
                xcb1.append(k['xcb'][i]) 
                ycb1.append(k['ycb'][i])
                count1 += 1
    print('I found ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    return sky0_stack, sky1_stack, xca0, yca0, xcb0, ycb0, xca1, yca1, xcb1, ycb1

def build_skyframe_stack(path, skip_list=False, K_klip = 10):
    import os
    os.system('ls '+path+'*sky_0* > list')
    print('Collecting sky images for',path.split('/')[0],'...')
    with open('list') as f:
        z = f.read().splitlines()
    image = fits.getdata(z[0])
    shape = image.shape
    sky_stack = np.zeros((len(z),*shape))
    count0 = 0
    xca0, yca0, xcb0, ycb0 = [], [], [], []
    for i in range(len(z)):
        # open the image
        image = fits.getdata(z[i])
        imhdr = fits.getheader(z[i])
        # Divide by number of coadds:
        image = image/imhdr['COADDS']
        # Stack the image:
        sky_stack[count0,:,:] = image
        count0 += 1
    print('I found ',sky_stack.shape[0],' sky frames')
    os.system('rm list')
    return sky_stack 

def build_skysubbed_stack(k):
    '''Stack skysubbed images into Nx512x1024 array for Nod 0 and Nod 1.
       Written by Logan A. Pearce, 2020

        Parameters:
        -----------
            k : pandas dataframe
                list of skysubbed files and x/y star locations
                output from bditools.findstars
            
        Returns:
        --------
            stack0, stack1 : Nx512x1024 array
                stack of skysubbed images in nod 0 and nod 1.
    '''
    from cliotools.pcaskysub import raw_beam_count
    count0,count1 = raw_beam_count(k)

    sp = k['filename'][0].split(',')
    filename = sp[0]

    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(filename).shape

    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(k)):
            sp = k['filename'][i].split(',')
            filename = sp[0]
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            
            # Divide by number of coadds:
            # if imhdr['COADDS'] != 1:
            #   image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
            # ^ Don't divide by number of coadds here because we already did it in
            # the pcaskysub step.
                
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
        for i in range(len(k)):
            sp = k['filename'][i].split(',')
            filename = sp[0]
            # open the image
            image = fits.getdata(filename)
            imhdr = fits.getheader(filename)
            # Divide by number of coadds:
            # image = image/imhdr['COADDS'] #<- not needed because it was
            # done in the pcaskysub step.
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                sky0_stack[count0,:,:] = image
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                count1 += 1
    #print('I found ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
    return sky0_stack, sky1_stack

def plot(image):
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize
    from matplotlib.colors import LogNorm
    #%matplotlib notebook
    plt.imshow(image, origin='lower', cmap='gray',norm = ImageNormalize(image, interval=ZScaleInterval(),))
    plt.show()
    
def mask(image, xc, yc, radius = 20):
    """ Mask the stars in an image
        Parameters:
        -----------
           image : 2d image array
               image to be masked
           xc, yc : flt
               x/y pixel location of the center of the star
           radius : int
               radius of circular mask to apply to star
        Returns:
        --------
            image_masked : 2d array
                masked image
    """
    import numpy as np
    # copy image:
    image_masked = image.copy()
    for i in range(len(xc)):
        # Make a meshgrid of the image centered at star locations:
        xx,yy = np.meshgrid(np.arange(image.shape[1])-xc[i],np.arange(image.shape[0])-yc[i])
        # Make an array of the distances of each pixel from that center:
        rA=np.hypot(xx,yy)
        #image0_masked = np.ma.masked_where(rA < radius, image0)
        image_masked[np.where(rA < radius)] = np.nan
    return image_masked

def highpassfilter(stack, size = 11):
    """ High pass filter (unsharp mask) of image

        Parameters:
        -----------
           stack : 3d array
               stack of images to apply hpf to
           size : int
               extent of hpf
        Returns:
        --------
            infilt : 3d array
                stack of images with hpf applied
    """
    from scipy import ndimage
    from cliotools.pcaskysub import update_progress
    # copy image stack:
    imfilt = stack.copy()
    # for each image:
    for i in range(imfilt.shape[0]):
        # make hpf filter:
        imhpf = ndimage.median_filter(stack[i,:,:], size)
        # subtract hpf from image:
        im = stack[i]-imhpf
        # place filter image in output stack:
        imfilt[i,:,:] = im
        update_progress(i+1,imfilt.shape[0])
    return imfilt

def findbadpix(std, chunk, n=5):
    ''' Identify bad pixels in a map of the standard deviation
        in the time domain.
        Parameters:
        -----------
        std : 2d array
            map of std deviation in the time domain of masked
            stack of images from a single nod
        chunk : 2d array
            chunk of the std dev map that does not contain any
            visible bad pixels
        n : int
            threshold = median(chunk) + n*sigma, where sigma is std dev in the 
            the chunk
        Returns:
        --------
        badpix : 2d array
            list of indicies of bad pixels
        badpixmap : 2d array
            bad pixel map: identifies pix = 1, else pix = 0
    '''
    import numpy as np
    # Select a patch of sky with no obsvious highly-variable pixels
    # and take the standard deviation in that chunk to
    # sample the "correct" noise level:
    sigma = np.std(chunk)
    # set upper threshold as some n*sigma
    threshold = np.median(chunk) + n*sigma
    # find pixels exceeding threshold
    badpix = np.where(std>threshold)
    # make a bad pixel map:
    badpixmap = np.zeros(std.shape)
    badpixmap[badpix] = 1
    return badpix, badpixmap

def stack_em_up(path, skip_list = False, imlist=[]):
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
        imlist : str
            if performing action on subset of images in a dataset, supply a list of paths to
            image files as this variable
        
    Returns:
    --------
        sky0_stack, sky1_stack : Nx512x1024 array
            stack of reference images.
        K_klip : int
            if N < requested number of modes, return new value of K_klip where 
            K_klip = min(sky0_stack.shape[0],sky1_stack.shape[0])
            otherwise returns requested number of modes.
    '''
    g = open(path+'skysubbed0.txt','w')
    h = open(path+'skysubbed1.txt','w')
    g.close()
    h.close()
    # Make a list of all images in the dataset:
    if skip_list == False:
        os.system('ls '+path+'*skysub.fit > list')

    if len(imlist) == 0:
        # Make list of all fits images in folder:
        os.system('ls '+path+'*0*skysub.fit > list')
        # Open the list:
        with open('list') as f:
            ims = f.read().splitlines()
    else:
        ims = imlist

    count0,count1 = beam_count(ims)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(ims[0]).shape
    
    print('Stacking reference images for',path.split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        #sky0_stack_masked = np.zeros((count0*shape[0],shape[1],shape[2]))
        #sky1_stack_masked = np.zeros((count1*shape[0],shape[1],shape[2]))
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
                # Apply mask:
                #image_masked = image.copy()
                #for j in range(image.shape[0]):
                #    image_masked[j,~np.isnan(mask0)] = 0
                # add to masked stack:
                #sky0_stack_masked[count0:count0+image.shape[0],:,:] = image_masked
                g = open(path+'skysubbed0.txt','a')
                g.write(str(count0)+' '+str(ims[i])+' '+str(imhdr['BEAM'])+'\n')
                g.close()
                # iterate counter
                count0 += shape[0]
            # If its nod 1:
            if imhdr['BEAM'] == 1:
                # Stack the image cube into one array
                sky1_stack[count1:count1+image.shape[0],:,:] = image
                #image_masked = image.copy()
                #for j in range(image.shape[0]):
                #    image_masked[j,~np.isnan(mask1)] = 0
                #sky1_stack_masked[count1:count1+image.shape[0],:,:] = image_masked
                g = open(path+'skysubbed1.txt','a')
                g.write(str(count0)+' '+str(ims[i])+' '+str(imhdr['BEAM'])+'\n')
                g.close()
                count1 += shape[0]  
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        sky1_stack = np.zeros((count1,shape[0],shape[1]))
        sky0_stack = np.zeros((count0,shape[0],shape[1]))
        #sky1_stack_masked = np.zeros((count1,shape[0],shape[1]))
        #sky0_stack_masked = np.zeros((count0,shape[0],shape[1]))
        count0, count1 = 0,0
        for j in ims:
            # Open the image data:
            image = fits.getdata(j)
            imhdr = fits.getheader(j)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            if imhdr['BEAM'] == 0:
                # add to unmasked stack:
                sky0_stack[count0,:,:] = image
                # Apply mask:
                #image[~np.isnan(mask0)] = 0
                # add to masked stack:
                #sky0_stack_masked[count0,:,:] = image
                # iterate counter
                g = open(path+'skysubbed0.txt','a')
                g.write(str(count0)+' '+str(j)+' '+str(imhdr['BEAM'])+'\n')
                g.close()
                count0 += 1
            if imhdr['BEAM'] == 1:
                sky1_stack[count1,:,:] = image
                # Apply mask:
                #image[~np.isnan(mask1)] = 0
                # add to masked stack:
                #sky1_stack_masked[count1,:,:] = image
                g = open(path+'skysubbed1.txt','a')
                g.write(str(count1)+' '+str(j)+' '+str(imhdr['BEAM'])+'\n')
                g.close()
                count1 += 1
    print('I will use ',sky0_stack.shape[0],' images for Nod 0, and ',sky1_stack.shape[0],'images for Nod 1')
        
    return sky0_stack, sky1_stack

def stack_em_all_up(path, skip_list = False, imlist=[], filesuffix='ssbp'):
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
        imlist : str
            if performing action on subset of images in a dataset, supply a list of paths to
            image files as this variable
        
    Returns:
    --------
        stack : Nx512x1024 array
            stack of reference images.
        K_klip : int
            if N < requested number of modes, return new value of K_klip where 
            K_klip = min(sky0_stack.shape[0],sky1_stack.shape[0])
            otherwise returns requested number of modes.
    '''
    g = open(path+'skysubbed.txt','w')
    g.close()
    # Make a list of all images in the dataset:
    if skip_list == False:
        os.system('ls '+path+'*'+filesuffix+'.fit > list')

    if len(imlist) == 0:
        # Make list of all fits images in folder:
        os.system('ls '+path+'*'+filesuffix+'.fit > list')
        # Open the list:
        with open('list') as f:
            ims = f.read().splitlines()
    else:
        ims = imlist
    count = len(ims)
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(ims[0]).shape
    
    print('Stacking reference images for',path.split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        stack = np.zeros((count*shape[0],shape[1],shape[2]))
        count = 0
        # For each image in the dataset:
        for i in range(len(ims)):
            # open the image
            image = fits.getdata(ims[i])
            imhdr = fits.getheader(ims[i])
            
            # Divide by number of coadds:
            if imhdr['COADDS'] != 1:
                image = [image[i]/imhdr['COADDS'] for i in range(shape[0])]
                
            # Stack the image cube into one array
            stack[count:count+image.shape[0],:,:] = image
            # Apply mask:
            #image_masked = image.copy()
            #for j in range(image.shape[0]):
            #    image_masked[j,~np.isnan(mask0)] = 0
            # add to masked stack:
            #sky0_stack_masked[count0:count0+image.shape[0],:,:] = image_masked
            g = open(path+'skysubbed.txt','a')
            g.write(str(count+1)+' '+str(ims[i])+' '+str(imhdr['BEAM'])+'\n')
            g.close()
            # iterate counter
            count += shape[0]
        
    elif len(shape) == 2:
        # Each image is a single frame composite of several coadds.
        stack = np.zeros((count,shape[0],shape[1]))
        count = 0
        for j in ims:
            # Open the image data:
            image = fits.getdata(j)
            imhdr = fits.getheader(j)
            # Divide by number of coadds:
            image = image/imhdr['COADDS']
            # Stack the image in the appropriate stack:
            # add to unmasked stack:
            stack[count,:,:] = image
            # Apply mask:
            #image[~np.isnan(mask0)] = 0
            # add to masked stack:
            #sky0_stack_masked[count0,:,:] = image
            # iterate counter
            g = open(path+'skysubbed.txt','a')
            g.write(str(count+1)+' '+str(j)+' '+str(imhdr['BEAM'])+'\n')
            g.close()
            count += 1
        
    return stack
