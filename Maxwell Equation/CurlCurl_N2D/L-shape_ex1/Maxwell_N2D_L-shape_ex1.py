# _*_ coding:utf-8 _*_
"""
This demo program solves Maxwell L-shape bench-mark problem in 2D
    curl curl u - grad div u - omega^2 u = f    in  \Omega
    u x n = 0   on \partial\Omega
    div f = 0   in \Omega
    div u = 0   on \Partial\Omega
L-shape domain given as following 
    (-1,1) ______________________(1,1)
	   |                    |
	   |                    |
	   |                    |
	   |                    |
	   |          __________|(1.0)  
	   |          |(0,0)
	   |          |
	   |          |
	   |          |
   (-1,-1) |__________|(0,-1)   
  
and exact solution given by
    omega = 1
    psi  = r**(2.0/3.0)*cos((2.0/3.0)*theta)*phi(r)
    u = (u1,u2)' = curl psi
    r = r, theta = theta
    u1 = (2.0/3.0)*r**(-1.0/3.0)*sin((1.0/3.0)*theta)phi(r) + 
         r**(2.0/3.0)*cos((2.0/3.0)*theta)*sin(theta)*phi'(r)
    u2 = - (2.0/3.0)*r**(1.0/3.0)*cos((1.0/3.0)*theta)phi(r) - 
         r**(2.0/3.0)*cos((2.0/3.0)*theta)*cos(theta)*phi'(r)
    f = curl curl u  - u 
"""
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import dump
from xml.etree.ElementTree import Comment
from xml.etree.ElementTree import tostring
from dolfin import *
import numpy,sys
import numpy as np
import math #import pi,log
import scitools.BoxField
import scitools.easyviz as ev
import time

t0 = time.time()
# For example: python Maxwell_N2D_L-shape.py  8 8 2 
# Define mesh
nx = int(sys.argv[1]) 
ny = int(sys.argv[2])
nz = int(sys.argv[3])
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 2, 2)
editor.init_vertices(8)
editor.init_cells(6)
editor.add_vertex(0, 0.0, 0.0)
editor.add_vertex(1, 1.0, 0.0)
editor.add_vertex(2, 1.0, 1.0)
editor.add_vertex(3, 0.0, 1.0)
editor.add_vertex(4, -1.0, 1.0)
editor.add_vertex(5, -1.0, 0.0)
editor.add_vertex(6, -1.0, -1.0)
editor.add_vertex(7, 0.0, -1.0)
editor.add_cell(0, 0, 1, 3)
editor.add_cell(1, 1, 2, 3)
editor.add_cell(2, 0, 3, 5)
editor.add_cell(3, 3, 4, 5)
editor.add_cell(4, 0, 5, 7)
editor.add_cell(5, 5, 6, 7)
editor.close()
num_refine = int(math.log(int(sys.argv[1]),2))
h=0.5**num_refine
for i in range(num_refine):
	mesh=refine(mesh)
#define function space
U_h = VectorFunctionSpace(mesh, "Lagrange", nz) #nz次元

# Define trial and test function
u = TrialFunction(U_h)
v = TestFunction(U_h)
 

# Define boundary condition:u x n = u_1*n_2 - u_2*n1 = 0
def u0_boundary(x, on_boundary):
	tol=1.0e-14
        return on_boundary and ( abs(x[1]-1)<tol or abs(x[1]+1)<tol or (abs(x[1])<tol and x[0]>=0) )

def u1_boundary(x, on_boundary):
	tol=1.0e-14
        return on_boundary and ( abs(x[0]-1)<tol or abs(x[0]+1)<tol or (abs(x[0])<tol and x[1]<=0) )

u0 = Constant("0.0")
bc0 = DirichletBC(U_h.sub(0), u0, u0_boundary)
bc1 = DirichletBC(U_h.sub(1), u0, u1_boundary)
bc=[bc0,bc1]
print("bc is created!")
	
# Define parameter,f and exact solution.
t1 = time.time()
omega = Constant('1.0')
def phi(x,y):
     r = sqrt(x**2 + y**2)
     tol = 1.0e-6
     if r-0.25<tol:
       return 1
     elif r-0.75<tol:
       return -16*(r-0.75)**3*(5 + 15*(r - 0.75) + 12*(r - 0.75)**2)
     else:
       return 0

def phi_1(x,y):
     r = sqrt(x**2 + y**2)
     tol = 1.0e-6
     if r-0.75<tol and r - 0.25>tol:
       return -240*(r-0.75)**2*(1 + 4*(r - 0.75) + 4*(r - 0.75)**2)
     else:
       return 0

def phi_2(x,y):
     r = sqrt(x**2 + y**2)
     tol = 1.0e-6
     if r-0.75<tol and r - 0.25>tol:
       return -480*(r-0.75)*(1 + 6*(r - 0.75) + 8*(r - 0.75)**2)
     else:
       return 0

def phi_3(x,y):
     r = sqrt(x**2 + y**2)
     tol = 1.0e-6
     if r-0.75<tol and r - 0.25>tol:
       return -480*(1 + 12*(r - 0.75) + 24*(r - 0.75)**2)
     else:
       return 0

def theta(xx,yy):
   tol = 1.0e-6
   if xx > tol:
      return atan(yy/xx)
   elif xx < -tol:
      return  pi + atan(yy/xx)
   elif yy > tol:
      return pi/2.0
   elif yy < -tol:
      return 3.0*pi/2.0
   else:
      return 0
   
class f_Expression(Expression):
    def eval(self, value, x):
        tol = 1.0e-6
        if abs(x[0])<tol and abs(x[1])<tol:
          value[0] = 0
          value[1] = 0  
        else:
          value[0] = -(2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*phi(x[0],x[1]) - \
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*sin(theta(x[0],x[1]))*phi_1(x[0],x[1])+\
                     (7.0/9.0)*pow(x[0]**2 + x[1]**2,-2.0/3.0)*sin((5.0/3.0)*theta(x[0],x[1]))*phi_1(x[0],x[1]) +\
                     (7.0/9.0)*pow(x[0]**2 + x[1]**2,-2.0/3.0)*cos(theta(x[0],x[1]))*sin((2.0/3.0)*theta(x[0],x[1]))*phi_1(x[0],x[1]) - \
                     (7.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos((2.0/3.0)*theta(x[0],x[1]))*sin(theta(x[0],x[1]))*phi_2(x[0],x[1]) - \
                     (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*phi_2(x[0],x[1]) -\
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*sin(theta(x[0],x[1]))*phi_3(x[0],x[1]) 
          value[1] =  (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*phi(x[0],x[1]) +\
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*cos(theta(x[0],x[1]))*phi_1(x[0],x[1])-\
                     (7.0/9.0)*pow(x[0]**2 + x[1]**2,-2.0/3.0)*cos((1.0/3.0)*theta(x[0],x[1]))*phi_1(x[0],x[1]) +\
                     (7.0/9.0)*pow(x[0]**2 + x[1]**2,-2.0/3.0)*sin(theta(x[0],x[1]))*sin((2.0/3.0)*theta(x[0],x[1]))*phi_1(x[0],x[1]) + \
                     (7.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos((2.0/3.0)*theta(x[0],x[1]))*cos(theta(x[0],x[1]))*phi_2(x[0],x[1]) + \
                     (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*phi_2(x[0],x[1]) + \
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*cos(theta(x[0],x[1]))*phi_3(x[0],x[1])
    def value_shape(self):
        return (2,)
f = f_Expression()
#Ue=VectorFunctionSpace(mesh,"Lagrange",degree=5)
#ff = interpolate(f,Ue)

class u_exact_Expression(Expression):
    def eval(self, value, x):
        tol = 1.0e-6
        if abs(x[0])<tol and abs(x[1])<tol:        
          value[0] = 0
          value[1] = 0
	else: 
          value[0] = (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*phi(x[0],x[1]) +\
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*sin(theta(x[0],x[1]))*phi_1(x[0],x[1])
          value[1] = -(2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*phi(x[0],x[1]) -\
                     pow(x[0]**2 + x[1]**2,1.0/3.0)*cos((2.0/3.0)*theta(x[0],x[1]))*cos(theta(x[0],x[1]))*phi_1(x[0],x[1])
    def value_shape(self):
        return (2,)
u_exact = u_exact_Expression()

# Define variational form
n = FacetNormal(mesh) 
a = inner(curl(u),curl(v))*dx + inner(div(u),div(v))*dx - omega**2*inner(u,v)*dx
L = inner(f,v)*dx
# Solve
(A,b) = assemble_system(a,L,bc)
t2 = time.time()
print("A and b is assembled!")
u0 = Function(U_h)
solver = KrylovSolver("cg", "ilu")
solver.parameters["absolute_tolerance"] = 1E-14
solver.parameters["relative_tolerance"] = 1E-14
solver.parameters["maximum_iterations"] = 1000
set_log_level(DEBUG)
solve(A, u0.vector(), b)
	
# Save solution in VTK format
file = File("Data/Maxwell S2D_%gx%gP%g.pvd"%(nx,ny,nz))
file << u0
# Plot solution(mesh,surf,contour)
visual_mesh = plot(mesh,title = "Mesh")
visual_u = plot(u0,wireframe =  True,title = "the approximation of u",rescale = True , axes = True, basename = "deflection" ,legend ="u0")
visual_u1 = plot(u0[0],title="the approximation of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_u2 = plot(u0[1],wireframe =  True,title = "the approximation of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_mesh.write_png("Image/P%g_mesh_%gx%g.png"%(nz,nx,ny))
visual_u.write_png("Image/P%g_u_%gx%g"%(nz,nx,ny))
visual_u1.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_u2.elevate(-65)
visual_u1.write_png("Image/P%g_u1_%gx%g.png"%(nz,nx,ny))
visual_u2.write_png("Image/P%g_u2_%gx%g.png"%(nz,nx,ny))

# Define a higher-order approximation to the exact solution
t3 = time.time()
Ue=VectorFunctionSpace(mesh,"Lagrange",degree=5)
u_ex = interpolate(u_exact,Ue)
t4 = time.time()
# Plot exact solution
visual_ue1 = plot(u_ex[0],title="the exact solution of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_ue2 = plot(u_ex[1],wireframe = True,title="the exact solution of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_ue1.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_ue2.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_ue1.write_png("Image/P%gu1_exact_%gx%g.png"%(nz,nx,ny))
visual_ue2.write_png("Image/P%gu2_exact_%gx%g.png"%(nz,nx,ny))
interactive()

#Define L2 norm and H1 norm relative errors 
u1_error = (u0[0]-u_ex[0])**2*dx
u2_error = (u0[1]-u_ex[1])**2*dx
u1_ex_L2 = u_ex[0]**2*dx
u2_ex_L2 = u_ex[1]**2*dx

L2_error_u1 = sqrt(assemble(u1_error)/assemble(u1_ex_L2))
L2_error_u2 = sqrt(assemble(u2_error)/assemble(u2_ex_L2))

Curl_Div_error_u1 = sqrt(assemble(inner(curl(u0[0]-u_ex[0]),curl(u0[0]-u_ex[0]))*dx + inner(div(u0[0]-u_ex[0]),div(u0[0]-u_ex[0]))*dx)/assemble(inner(curl(u_ex[0]),curl(u_ex[0]))*dx + inner(div(u_ex[0]),div(u_ex[0]))*dx))
Curl_Div_error_u2 = sqrt(assemble(inner(curl(u0[1]-u_ex[1]),curl(u0[1]-u_ex[1]))*dx + inner(div(u0[1]-u_ex[1]),div(u0[1]-u_ex[1]))*dx)/assemble(inner(curl(u_ex[1]),curl(u_ex[1]))*dx + inner(div(u_ex[0]),div(u_ex[0]))*dx))


print("h=" ,CellSize(mesh))
print("The number of cells(triangle) in the mesh:" ,mesh.num_cells())
print("The number of vertices in the mesh:" ,mesh.num_vertices())
print("L2_error_u1=" ,L2_error_u1)
print("L2_error_u2=" ,L2_error_u2)
print("Curl_error_u1=" ,Curl_Div_error_u1)
print("Curl_error_u2=" ,Curl_Div_error_u2)
print("Total time is=",time.time() - t0)
print("Assemble time is=",t2 - t1)
print("Interpolate time is=",t4 - t3)

file_object = open('Error_P%g.txt'%(nz),'a')
file_object.writelines("The number of cells(triangle) in the mesh:%g" %(mesh.num_cells()))
file_object.writelines("The number of vertices in the mesh:%g"%(mesh.num_vertices()))
file_object.writelines("Mesh:%gx%gP%g\n"%(nx,ny,nz))
file_object.writelines("L2_error_u1= %g\n"%(L2_error_u1))
file_object.writelines("L2_error_u2= %g\n"%(L2_error_u2))
file_object.writelines("Curl_error_u1= %g\n" %(Curl_Div_error_u1))
file_object.writelines("Curl_error_u2= %g\n\n"%(Curl_Div_error_u2))
file_object.close()

file_output = open('Ratio_P%g.txt'%(nz),'a')
E = np.array([nx,L2_error_u1,L2_error_u2,Curl_Div_error_u1,Curl_Div_error_u2])
print(E)
file_output.writelines("%g %g %g %g %g\n"%(E[0],E[1],E[2],E[3],E[4]))


