from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='cliotools',
      version='0.1',
      description='python tools for working with clio data',
      url='https://github.com/logan-pearce/cliotools',
      author='Logan Pearce',
      author_email='loganpearce55@gmail.com',
      license='MIT',
      packages=['cliotools'],
      install_requires=['numpy','scipy','astropy','matplotlib','opencv-python','photutils'],
      #include_package_data=True,
      zip_safe=False)
