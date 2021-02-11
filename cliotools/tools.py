def circle_mask(radius, xsize, ysize, xc, yc, radius_format = 'pixels', cval = 0)):
    xx,yy = np.meshgrid(np.arange(xsize)-xc,np.arange(ysize)-yc)
    r=np.hypot(xx,yy)
    return np.where(r<radius)