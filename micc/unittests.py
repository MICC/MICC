import unittest
from curves import curvePair
import numpy as np

class DistanceTests(unittest.TestCase):
	'''
	def test_1(self):
		test = curvePair([1,2,3,4,12,10,14,15,9,13,11,12,4,5,6,7,21,22,23,24,16,17,18,19],[24,1,2,3,13,9,15,16,8,14,10,11,5,6,7,8,20,21,22,23,17,18,19,20])
		self.assertEqual(test.distance,3)

	def test_2(self):
		test = curvePair([6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 1, 2, 3, 4, 17, 5], [1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4])
		self.assertEqual(test.distance,'at least 4!')

	def test_3(self):
		test = curvePair([17, 1, 2, 3, 4, 17, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],[1, 2, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4])
		self.assertEqual(test.distance,'at least 4!')

	def test_4(self):
		test = curvePair([23, 22, 1, 2, 3, 4, 5, 22, 23, 24, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24],[1, 2, 3, 4, 19, 20, 21, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 19, 5])
		self.assertEqual(test.distance,'at least 4!')
	def test_georgia_d4(self):
		top = [5,6,7,8,3,4,11,0,1,10,11,6]
		bot = [7,8,9,4,5,0,1,2,3,2,9,10]
		test = curvePair(top,bot)
		self.assertEqual(test.distance,'at least 4!')
	def test_hempel(self):
		top = [21,7,8,9,10,11,22,23,24,0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20]
		bot = [9,10,11,12,13,14,15,1,2,3,4,5,16,17,18,19,20,21,22,23,24,0,6,7,8]
		test = curvePair(top,bot)
		self.assertEqual(test.distance,'at least 4!')
	def test_birman(self):
		top = [4,3,3,1,0,0]
		bot = [5,5,4,2,2,1]
		test = curvePair(top,bot)
		self.assertEqual(test.distance,3)
	'''	
	def test_georgia_d3(self):
		top = [2,3,7,9,10,7,8,2,4,5,0,1]
		bot = [10,11,0,1,8,9,11,5,6,3,4,6]
		test = curvePair(top,bot)
		self.assertEqual(test.distance,3)
if __name__ == '__main__':
	unittest.main()
