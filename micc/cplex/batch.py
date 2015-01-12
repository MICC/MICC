from pycplex import weights_to_ladder
import re
import multiprocessing as mp
import micc.curves as c
import sys

def test_curve_pair(original_ladder):
	top_length = len(original_ladder[0])
	bottom_length = len(original_ladder[1])
	if top_length == bottom_length:
		d3,d4 = c.test_perms(original_ladder)
		if d4: return d4
		else: return [None]
	else:
		return [None]

def test_ladder_list(sols_and_ladders):
	
	d4 = []
	i = 0
	for solution, ladder in sols_and_ladders:
		print i, mp.current_process().pid
		i += 1
		d4s = test_curve_pair(ladder)
		d4.append([solution, d4s])
	return d4

''' 
w1   w2   w6   w3   w5   w6'
-|----|----|----|----|----|-
w1   w4   w5   w3   w4'  w2
top_order =    [1, 2, 6, 3, 5, -6] 
bottom_order = [1, 4, 5, 3, -4, 2]
'''
if __name__ == '__main__':

	#genus 3
	top_order =    [1, 2,  12, 3, 6, 4, 5, -6, 7, 9, 11, -12] 
	bottom_order = [1, 10, 11, 3, 7, 8, 5, 4, -8, 9,-10, 2]
	
	file = sys.argv[1]
	f = open(file, 'r')
	cpus = mp.cpu_count()
	subdivs = { i : [] for i in range(cpus) }
	div = 0 
	itr = 0
	for s in f.readlines():
		#w_3, w_8, w_12
		if itr == 30:
			break
		itr +=1
		str_sol = s.split(',')
		solution = [int(float(re.sub('[\s\[\]]', '',str(i)))) for i in str_sol]

		l = weights_to_ladder(solution, top_order, bottom_order)
		'''
		print 'In:'
		print solution
		print top_order
		print bottom_order
		print 'Out:'
		print l[0]
		print l[1],'\n'
		'''
		subdivs[div].append([solution,l])
		div  = (div + 1) % cpus
	
	print 'Beginning....'
	pool = mp.Pool(processes = cpus)
	#print div
	#print len(subdivs.values())
	results = pool.map(test_ladder_list, subdivs.values())
	print '----------------------------------------------------------------'
	print '----------------------------------------------------------------'
	print '----------------------------------------------------------------'
	d4_count = 0
	for result in results:
		for solution, d4s in result:
			print 'solution:', solution, '-  intersection:', sum(solution)
			for curve in d4s:
				if curve:
					print 'Distance 4 Ladder:'
					print curve.ladder[0]
					print curve.ladder[1]
					d4_count += 1
					print '##################################################'
			print '----------------------------------------------------------------'
	print 'We have found ', d4_count, 'curves that are at least distance 4.'
	
'''
f = open('genus3lp.out', 'r')
outfiles = [ open('genus31.out', 'w'), open('genus32.out', 'w'), open('genus33.out', 'w'), open('genus34.out', 'w')]

ladders = []
i = 0
for s in f.readlines():
	outfiles[i].write(s)
	i += 1
	i %= len(outfiles)
'''
