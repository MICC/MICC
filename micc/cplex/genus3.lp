MINIMIZE
obj: w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12

SUBJECT TO
r1: w3 + w6 + w4 + w8 + w9 + w11 >= 5
r2: w4 + w5 >= 5
r3: w3 + w7 + w9 + w11 >= 5
r4: w8 + w7 + w6 + w5 >= 5
r5: w8 + w7 + w6 + w4 >= 5
r6: w2 + w10 + w9 + w8 + w4 + w6 + w3 + w12 >= 5
r7: w2 + w10 + w9 + w8 + w5 + w6 + w3 + w12 >= 5
r8: w1 + w10 + w9 + w8 + w4 + w6 + w3 + w12 >= 5
r9: w1 + w10 + w9 + w8 + w5 + w6 + w3 + w12 >= 5
r10: w1 + w10 + w9 + w7 + w3 + w12 >= 5
r11: w11 + w9 + w8 + w4 + w6 + w3 >= 5
r12: w11 + w9 + w8 + w5 + w6 + w3 >= 5
r13: w1 + w10 + w11 + w12 >= 5
r14: w2 + w10 + w11 + w12 >= 5
r15: w1 + w2 >= 5
r16: w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12  <= 32

bounds
0 <= w1
0 <= w2
0 <= w3
0 <= w4
0 <= w5
0 <= w6
0 <= w7
0 <= w8
0 <= w9
0 <= w10
0 <= w11
0 <= w12

integer
w1
w2
w3
w4
w5
w6
w7
w8
w9
w10
w11
w12
end



