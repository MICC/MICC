import numpy as np
from micc import utils


class RigidGraph(object):
    """
    This class is the generic data structure describing a graphical
    representation of a given curve pair. The constructor is written to populate
    member variables using different algorithms for differing inputs, which
    allows for a reasonable level of backwards compatibility.
    """

    def __init__(self, *args, **kwargs):
        """
        The main purpose of this constructor is to filter out the representation
        that was sent into MICC and call the appropriate function once that has
        been determined. Please be reasonably cautious; only kwargs['ladder'] OR
        kwargs['cycle'] should be true at a given time (only use one rep.
        at a time!). Also, class variables are initialized prior to the argument
        parsing.
        """


        if kwargs.get('ladder', False):
            self.ladder_init(*args, **kwargs)
        elif kwargs.get('cycle', False):
            self.cycle_init(*args, **kwargs)

    def ladder_init(self, *args, **kwargs):
        pass

    def cycle_init(self, *args, **kwargs):
        pass


class CurvePair(object):
    """
    This class encapsulates all of the information required to compute and study
    distance in the curve complex. It functions as an interface between the user
    and the MICC internal API. This means that a user wishing to purely compute
    distance should do by instantiating and manipulating this object, not by
    calling MICC API functions.

    Attributes:
        ladder: a list of lists:
        cycle: a string of alternating integers and +/- signs that characterize
        the curve pair.
    """
    def __init__(self, compute=False, *args, **kwargs):

        if compute:
            pass
        pass
