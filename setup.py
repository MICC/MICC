"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup
'''
from sys import setrecursionlimit
setrecursionlimit(10000)
APP = ['micc/cli.py']
DATA_FILES = []
OPTIONS = {'argv_emulation': True}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
'''

setup(name='MICC',
      version='0.1.3',
      author='Matt Morse, Paul Glenn',
      author_email='mjmorse@buffalo.edu, paulglen@berkeley.edu',
      packages=['micc', 'micc.tests'],
      url='http://micc.github.io',
      download_url='https://github.com/micc/micc',
      license='LICENSE',
      description='An implementation of the Birman-Margalit-Menasco Theorem, to be used in a experimental, exploratory manner.',
      long_description=open('README.rst').read(),
      install_requires=[
          "numpy== 1.21.0",
          "nose >= 1.3.1",
          "rednose"
      ],
      test_suite="nose.collector")
