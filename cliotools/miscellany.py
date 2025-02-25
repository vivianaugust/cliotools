import numpy as np

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    From : https://www.astrobetter.com/wiki/python_radial_profiles
    By Ian J Crossfield
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = ny.nan
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.nanmedian(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata

def radial_data_median_only(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    ''' Pared down version of radial_data that computes only the median radial profile
    '''
    import numpy as np
    data = np.array(data)
    
    if working_mask==None:
        working_mask = np.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = np.arange(-npix/2.,npix/2.)
        y1 = np.arange(-npiy/2.,npiy/2.)
        x,y = np.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    dr = np.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = np.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    #radialdata = radialDat()
    radialdata_median = np.zeros(nrad)

    for irad in range(nrad): #= 1:numel(radial)
        minrad = irad*dr
        maxrad = minrad + dr
        thisindex = (r>=minrad) * (r<maxrad) * working_mask
        if not thisindex.ravel().any():
            radialdata_median[irad] = np.nan
        else:
            radialdata_median[irad] = np.nanmedian(data[thisindex])
    
    return radialdata_median


def CenteredDistanceMatrix(n, ny = None):
    ''' Creates 2d array of the distance of each element from the center

    Parameters
    ----------
        n : flt
            x-dimension of 2d array
        ny : flt (optional)
            optional y-dimension of 2d array.  If not provided, array is square of dimension nxn
    
    Returns
    -------
        2d matrix of distance from center
    '''
    nx = n
    if ny:
        pass
    else:
        ny = nx
    center = ((nx-1)*0.5,(ny-1)*0.5)
    xx,yy = np.meshgrid(np.arange(nx)-center[0],np.arange(ny)-center[1])
    r=np.hypot(xx,yy)
    return r

from matplotlib import pyplot as plt

class PixelCollect:
    def __init__(self, pixels):
        self.pixels = pixels
        self.xs = list(pixels.get_xdata())
        self.ys = list(pixels.get_ydata())
        self.cid = pixels.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.pixels.axes: return
        self.xs.append(np.int(np.round(event.xdata)))
        self.ys.append(np.int(np.round(event.ydata)))
        
def GetBadPix(image, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Image '+str(i)+': Hover over each bad pixel and click to record location')
    im_fig = ax.imshow(image, origin='lower', cmap='gray',
                   norm = ImageNormalize(im, interval=MinMaxInterval(),
                          stretch=SqrtStretch(),))
    pixels, = ax.plot([], [])
    pixelcollection = PixelCollect(pixels)

    plt.show()
    
    return pixelcollection.xs,pixelcollection.ys

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
    

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    import numpy as np
    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma


def get_distance(source_ids, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source id to return the distance and error in parsecs'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    try:
        d,e = np.array([]),np.array([])
        for source_id in source_ids:
            job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id))
            j = job.get_results()
            di,ei = distance(np.array(j['parallax']),np.array(j['parallax_error']))
            d = np.append(d,di)
            e = np.append(e,ei)
            print('For',source_id,'d=',[di,ei])
    except:
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_ids))
        j = job.get_results()
        d,e = distance(np.array(j['parallax']),np.array(j['parallax_error']))
    return d,e

