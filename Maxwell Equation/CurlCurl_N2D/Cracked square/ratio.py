from math import log as ln
import numpy as np
import matplotlib.pyplot as plt

h = [] #element sizes
E1 = [] #errors of u1
E2 = [] #errors of u2
r = []
for nx in [8,16,32,64,128]:
    h.append(ln(1.0/nx))

for ny in [4.32192,1.73049,0.79694,0.39595,0.206631]:
    E1.append(ln(ny))

for nz in [4.3627,1.7712,0.830323,0.423606,0.230516]:
    E2.append(ln(nz))

r = 0
s = 0
for i in range(1,len(E1)):
   r +=(E1[i]-E1[i-1])/(h[i]-h[i-1])
   s +=(E2[i]-E2[i-1])/(h[i]-h[i-1])

x = np.array([-4.5,-4.5,-4.0,-4.5])
y = np.array([1,1.5,1.5,1])
plt.figure(figsize=(8,6))
plt.plot(x,y,color="black")
plt.plot(h,E1,"b--",label="$u_1$")
plt.plot(h,E1,"*")
plt.plot(h,E2,"r--",label="$u_2$")
plt.plot(h,E2,"*")
plt.xlabel('$\ln(h)$',fontsize=20)
plt.ylabel(r'$\ln(\frac{||u - u_h||_0}{||u||_0})$',fontsize=20)
plt.xlim(-5,-1.5)
plt.ylim(-2,2)
plt.text(-4.25,1.1,"k=1")
#plt.text(-2.5,-3.5,r/4)
#plt.text(-2.7,-3.1,s/4)
plt.title("Error rates of the u1 and u2 approximations")
plt.legend()
plt.show()
    
