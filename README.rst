Metric in the Curve Complex: MICC
=================================
.. image:: https://travis-ci.org/MICC/MICC.svg?branch=master
    :target: https://travis-ci.org/MICC/MICC

The curve complex is a simplicial complex composed of vertices representing equivalency classes of isotopic 
simple closed curves on a surface of fixed genus and of edges drawn between vertices if classes contain a disjoint 
representative. MICC is a tool designed to compute short distances between these disjoint representatives, based 
on an intuitive disk-with-handles represntation of a surface.

Installation
------------
Installing through pip is recommended to use the programmatic interface:
::
    $ pip install micc

Otherwise, the command line interface for MICC is available at <http://micc.github.io/>

Getting Started
---------------
Example useage of MICC:

.. code-block:: python
    from curves import curvePair

    top    = [21,7,8,9,10,11,22,23,24,0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20]
    bottom = [9,10,11,12,13,14,15,1,2,3,4,5,16,17,18,19,20,21,22,23,24,0,6,7,8]
    test = curvePair(top, bottom)
    print test.distance

Documentation
-------------
TODO

License
-------
Copyright 2014 Matt Morse and Paul Glenn.

MICC is licensed under the `MIT License <https://github.com/micc/micc/LICENSE>`_.