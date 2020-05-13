import numpy as np
from astropy.io import fits
import os

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

##################### Bad pixel find and fix functions ###################
# badpixelsub uses the established CLIO bad pixel maps.  These functions #
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
        sp = k['filename'][i].split('_')
        filename = sp[0]+'_'+sp[1]+'.fit'
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
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(k['filename'][0]).shape
    xca0, yca0, xcb0, ycb0 = [], [], [], []
    xca1, yca1, xcb1, ycb1 = [], [], [], []
    
    print('Stacking reference images for',k['filename'][0].split('/')[0],'...')
    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(k)):
            sp = k['filename'][i].split('_')
            filename = sp[0]+'_'+sp[1]+'.fit'
            # open the image
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
            sp = k['filename'][i].split('_')
            filename = sp[0]+'_'+sp[1]+'.fit'
            # open the image
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
    
    # Distinguish between fits files that are single images vs data cubes:
    shape = fits.getdata(k['filename'][0]).shape

    # Make a stack of images to put into PCA analysis:
    if len(shape) == 3:
        # Each image is a cube of individual coadds.
        sky0_stack = np.zeros((count0*shape[0],shape[1],shape[2]))
        sky1_stack = np.zeros((count1*shape[0],shape[1],shape[2]))
        # Reset count
        count0, count1 = 0,0
        # For each image in the dataset:
        for i in range(len(k)):
            filename = k['filename'][i]
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
            filename = k['filename'][i]
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

def badpixfix(imagesr, badpixr, dx = 15):
    ''' Replace bad pixels with median of dx nearest neighbors
        along x direction

        Parameters:
        -----------
        imagesr : 3d array
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
    import numpy as np
    from cliotools.pcaskysub import update_progress
    # Interpolate bad pixels:
    imfix = imagesr.copy()
    for i in range(imagesr.shape[0]):
        for j in range(len(badpixr[1])):
            dx = dx # <- must be odd
            x, y = badpixr[1][j], badpixr[0][j]
            xarray = np.arange(x-dx,x+dx+1,2)
            xarray = xarray[np.where(xarray > 0)[0]]
            xarray = xarray[np.where(xarray < 1024)[0]]
            imfix[i,y,x] = np.nanmedian(imagesr[i,y,xarray])
        update_progress(i+1,imagesr.shape[0])
    return imfix

