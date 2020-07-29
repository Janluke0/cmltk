import matrix as mt

#check memory leak
for i in range(0,100000):
    m = mt.ones(1000,1000)
