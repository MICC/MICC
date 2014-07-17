MICC [![Build Status](https://travis-ci.org/MICC/MICC.svg?branch=master)](https://travis-ci.org/MICC/MICC)
====

Metric in the Curve Complex

The curve complex is a simplicial complex composed of vertices representing equivalency classes of isotopic 
simple closed curves on a surface of fixed genus and of edges drawn between vertices if classes contain a disjoint 
representative. MICC is a tool designed to compute short distances between these disjoint representatives, based 
on an intuitive disk-with-handles represntation of a surface.

Example useage of MICC:

<pre><code>
from curves import curvePair

top    = [21,7,8,9,10,11,22,23,24,0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20]
bottom = [9,10,11,12,13,14,15,1,2,3,4,5,16,17,18,19,20,21,22,23,24,0,6,7,8]
test = curvePair(top, bottom)
print test.distance

</pre></code>
