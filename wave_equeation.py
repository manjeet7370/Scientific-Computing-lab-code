# uxx = 1/16 utt
import numpy as np

m = np.zeros((5,6))

l,k,h = 1,1,1

for j in range(5):
    for i in range(6):
        if(i==0 or i==5):
            m[j,i] = 0
        elif(j==0):
            m[j,i] = i**2*(i-5)
        else:
            m[j,i] = 2*(1-l)*m[j-1,i] + l*(m[j-1,i-1] + m[j-1,i+1])

            if(j==1):
                m[j,i] = m[j,i]/2
            else:
                m[j,i] = m[j,i] - m[j-2,i]

print(m)