from math import log as ln
import numpy as np
import matplotlib.pyplot as plt

h = [] #element sizes
E1 = [] #errors of u1
E2 = [] #errors of u2
r = []
for nx in [4,8,16,32,64]:
    h.append(ln(1.0/nx))

for ny in [0.0874614,0.0481824,0.0286971,0.0176732,0.0110335]:
    E1.append(ln(ny))

for nz in [0.134036,0.0783667,0.0479573,0.0298728,0.0187361]:
    E2.append(ln(nz))

r = 0
s = 0
for i in range(1,len(E1)):
   r +=(E1[i]-E1[i-1])/(h[i]-h[i-1])
   s +=(E2[i]-E2[i-1])/(h[i]-h[i-1])

x = np.array([-4,-4,-3.5,-4])
y = np.array([-2.5,-2,-2,-2.5])
plt.figure(figsize=(8,6))
plt.plot(x,y,color="black")
plt.plot(h,E1,"b--",label="$u_1$")
plt.plot(h,E1,"*")
plt.plot(h,E2,"r--",label="$u_2$")
plt.plot(h,E2,"*")
plt.xlabel('$\ln(h)$',fontsize=20)
plt.ylabel(r'$\ln(\frac{||u - u_h||_0}{||u||_0})$',fontsize=20)
plt.xlim(-4.5,-1)
plt.ylim(-5,-1.5)
plt.text(-3.7,-2.4,"k=1")
#plt.text(-2.5,-3.5,r/4)
#plt.text(-2.7,-3.1,s/4)
plt.title("Error rates of the u1 and u2 approximations")
plt.legend()
plt.show()
    
