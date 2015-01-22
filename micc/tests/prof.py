import cProfile
from micc.curves import CurvePair

ladder = [[1, 7, 18, 24, 5, 16, 12, 8, 19, 25, 6, 17, 23, 4, 15, 11, 22, 3, 14, 10, 21, 2, 13, 9, 20],
          [25, 6, 17, 23, 4, 15, 11, 7, 18, 24, 5, 16, 22, 3, 14, 10, 21, 2, 13, 9, 20, 1, 12, 8, 19]]
curvepair = CurvePair(ladder, compute=False)

