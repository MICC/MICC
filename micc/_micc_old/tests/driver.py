from micc.curves import CurvePair

top = [4,3,3,1,0,0]
bot = [5,5,4,2,2,1]

top = [5,6,7,8,3,4,11,0,1,10,11,6]
bot = [7,8,9,4,5,0,1,2,3,2,9,10]


top = [2,3,7,9,10,7,8,2,4,5,0,1]
bot = [10,11,0,1,8,9,11,5,6,3,4,6]


top = [21,7,8,9,10,11,22,23,24,0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20]
bot = [9,10,11,12,13,14,15,1,2,3,4,5,16,17,18,19,20,21,22,23,24,0,6,7,8]
test = CurvePair(top,bot)
print test.distance
for loop in test.loop_matrices.values():
    cc = CurvePair(loop[0, :, 1], loop[0, :, 3])

    print 'dist:',cc.distance
    print 'genus:',cc.genus
    print '\n'
