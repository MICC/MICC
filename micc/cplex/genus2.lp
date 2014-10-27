MINIMIZE
obj: w1 + w2 + w3 + w4 + w5 + w6 

SUBJECT TO
r1: w1 + w4 + w5 + w6 >= 4
r2: w2 + w4 + w5 + w6 >= 4
r3: w3 + w5 >= 4
r4: w1 + w2 >= 4
r5: w1 + w3 + w4 + w6 >= 4
r6: w2 + w3 + w4 + w6 >= 4
r7: w4 - w6 = 0
r8: w1 + w2 + w3 + w4 + w5 + w6 <= 25

bounds
0 <= w1
0 <= w2
0 <= w3
0 <= w4
0 <= w5
0 <= w6

integer
w1
w2
w3
w4
w5
w6
end
