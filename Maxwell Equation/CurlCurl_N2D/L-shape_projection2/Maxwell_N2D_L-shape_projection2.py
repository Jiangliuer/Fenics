# _*_ coding:utf-8 _*_
"""
We solved use mixed finite element method.
This demo program solves Maxwell L-shape bench-mark problem in 2D
    curl curl u  - omega^2 u = f    in  \Omega
    u x n = 0   on \partial\Omega
    div u = g   in \Omega
Classical variational form 
 (curl u,curl v) + (div u,div v) - omega^2*(u,v) = (f,v) + (g,div v)

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
    phi  = r**(2.0/3.0)*sin((2.0/3.0)*theta)*(x**2-1)*(y**2-1)
    u = (u1,u2)' = grad phi
    r = sqrt(x[0]**2 + x[1]**2), theta = atan(x[1]/x[0])
    u1 = -(2.0/3.0)*r**(-1.0/3.0)*sin((1.0/3.0)*theta)*(x**2 - 1)*(y**2 - 1) + 
         2*x*r**(2.0/3.0)*sin((2.0/3.0)*theta)*(y**2 - 1)
    u2 = (2.0/3.0)*r**(-1.0/3.0)*cos((1.0/3.0)*theta)*(x**2 - 1)*(y**2 - 1) + 
         2*y*r**(2.0/3.0)*sin((2.0/3.0)*theta)*(x**2 - 1)
    f =  - u 
    g = div u
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
import vtk
t0 = time.time()
# For example: python Maxwell_N2D_L-shape.py  8 8 1
# Define mesh
nx = int(sys.argv[1]) 
ny = int(sys.argv[2])
num = int(sys.argv[3])
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

# Define function space
U_h = VectorFunctionSpace(mesh, "Lagrange", num) #num次元
B = VectorFunctionSpace(mesh, "Bubble", 3)
Q_h = FunctionSpace(mesh, "CG", num)  #projection space of R(div)
W_h = FunctionSpace(mesh, "CG", num)  #projection space of R(curl)
Mix_h = MixedFunctionSpace([U_h, B, Q_h, W_h]) 
print("num_sub_spaces():",Mix_h.num_sub_spaces())

# Define boundary condition:u x n = u_1*n_2 - u_2*n1 = 0 and projection boundary: u = 0
def project_boundary(x, on_boundary):
        tol=1.0e-14
        return on_boundary
#abs(x[1]-1)<tol or abs(x[1]+1)<tol or (abs(x[1])<tol and x[0]>tol) or  abs(x[0]-1)<tol or abs(x[0]+1)<tol or (abs(x[0])<tol and x[1]<tol)

def u0_boundary(x, on_boundary):
	tol=1.0e-14
        return on_boundary and ( abs(x[1]-1)<tol or abs(x[1]+1)<tol or (abs(x[1])<tol and x[0]>tol) )

def u1_boundary(x, on_boundary):
	tol=1.0e-14
        return on_boundary and ( abs(x[0]-1)<tol or abs(x[0]+1)<tol or (abs(x[0])<tol and x[1]<tol) )

w0 = Constant("0.0")
u_init = Constant(('0.0','0.0'))
bc0 = DirichletBC(Mix_h.sub(0).sub(0), w0, u0_boundary)
bc1 = DirichletBC(Mix_h.sub(0).sub(1), w0, u1_boundary)
bc2 = DirichletBC(Mix_h.sub(1), u_init, project_boundary)
bc3 = DirichletBC(Mix_h.sub(2), w0, project_boundary)
bc = [bc0,bc1,bc2,bc3] 
print("bc is created!")

t1 = time.time()	
# Define parameter,f and exact solution.
omega = Constant('1.0')

def theta(xx,yy):
   tol = 1.0e-16
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
        tol = 1.0e-16
        if abs(x[0])<tol and abs(x[1])<tol:
          value[0] = 0
          value[1] = 0
        else:
          value[0] =  (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*(x[0]**2 - 1)*(x[1]**2 - 1) -\
                       2*pow(x[0]**2 + x[1]**2,1.0/3.0)*sin((2.0/3.0)*theta(x[0],x[1]))*x[0]*(x[1]**2 - 1)
          value[1] = -(2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*(x[0]**2 - 1)*(x[1]**2 - 1) -\
                       2*pow(x[0]**2 + x[1]**2,1.0/3.0)*sin((2.0/3.0)*theta(x[0],x[1]))*x[1]*(x[0]**2 - 1)
    def value_shape(self):
        return (2,)
f = f_Expression()
f = interpolate(f,VectorFunctionSpace(mesh,"Lagrange",degree= 3))

class g_Expression(Expression):
    def __init__(self, mesh):
        self._mesh = mesh
    def eval(self, value, x):
        tol = 1.0e-16
        if abs(x[0])<tol and abs(x[1])<tol:
          value[0] = 0
        else:
          value[0] = (8.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*x[1]*(x[0]**2 - 1) -\
                     (8.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*x[0]*(x[1]**2 - 1) +\
                     2*pow(x[0]**2 + x[1]**2,1.0/3.0)*sin((2.0/3.0)*theta(x[0],x[1]))*(x[0]**2 + x[1]**2 - 2) 
    def value_shap(shap):
        return(1,) 
g = g_Expression(mesh = mesh)
g = interpolate(g,FunctionSpace(mesh,"Lagrange",degree= 3))
#plot(g)

class u_exact_Expression(Expression):
    def eval(self, value, x):
        tol = 1.0e-16
        if abs(x[0])<tol and abs(x[1])<tol:
          value[0] = 0
          value[1] = 0
        else:
          value[0] = -(2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*sin(theta(x[0],x[1])/3.0)*(x[0]**2 - 1)*(x[1]**2 - 1) +\
                     2*pow(x[0]**2 + x[1]**2,1.0/3.0)*sin((2.0/3.0)*theta(x[0],x[1]))*x[0]*(x[1]**2 - 1)
          value[1] = (2.0/3.0)*pow(x[0]**2 + x[1]**2,-1.0/6.0)*cos(theta(x[0],x[1])/3.0)*(x[0]**2 - 1)*(x[1]**2 - 1) +\
                     2*pow(x[0]**2 + x[1]**2,1.0/3.0)*sin((2.0/3.0)*theta(x[0],x[1]))*x[1]*(x[0]**2 - 1)
    def value_shape(self):
        return (2,)
u_exact = u_exact_Expression()

# Projection to Mixed finite element varitional problem
U = TrialFunction(Mix_h)
V = TestFunction(Mix_h)
(u,u_b,DivRu,CurlRu) = split(U)
(v,v_b,q,w) = split(V)

# projection of g onto space of Q_h
s_g = TrialFunction(Q_h)
v_q = TestFunction(Q_h)
a_s = inner(s_g,v_q)*dx
L_s = inner(g,v_q)*dx
Sg = Function(Q_h)
bc_sg = DirichletBC(Q_h, w0, project_boundary)
(A_s, b_s) = assemble_system(a_s, L_s, bc_sg)
solve(A_s, Sg.vector(), b_s)
#plot(Sg)

a = inner(curl(CurlRu),v + v_b)*dx - inner(grad(DivRu),v + v_b)*dx - \
    omega**2*inner(u + u_b,v + v_b)*dx + inner(DivRu,q)*dx + \
    inner(u + u_b,grad(q))*dx+ inner(CurlRu,w)*dx - inner(u + u_b,curl(w))*dx
L = inner(f, v + v_b)*dx + inner(Sg, div(v + v_b))*dx 

# Compute solution
(A,b) = assemble_system(a,L,bc)
print "A and b is assembled!"
t2 = time.time()
solver = KrylovSolver("cg", "ilu")
solver.parameters["absolute_tolerance"] = 1E-14
solver.parameters["relative_tolerance"] = 1E-14
solver.parameters["maximum_iterations"] = 1000
U_0 = Function(Mix_h) 
set_log_level(DEBUG)
solve(A, U_0.vector(), b )
print "Solved!"
(u0,ub,Div_R_u,Curl_R_u) = split(U_0)  #extract components
u0=u0+ub
"""
u0 = interpolate(u0,Ue)
ub = interpolate(ub,Ue)
u0=u0+ub
"""
# Save solution in VTK format
#u1u2 = interpolate(u0,Ue)
fileu = File("Data/u_%gx%g.pvd"%(nx,ny))
#fileu << u1u2


# Define a higher-order approximation to the exact solution
t3 = time.time()
Ue=VectorFunctionSpace(mesh,"Lagrange",degree= 3,dim=2)
u_ex = interpolate(u_exact,Ue)
t4 = time.time()
# Save solution in VTK format
file_ue = File("Data/ue_%gx%g.pvd"%(nx,ny))
file_ue << u_ex


# Plot solution(mesh,surf,contour) and save data.
#import viper
visual_mesh = plot(mesh,
		   title = "Mesh")
visual_u = plot(0.0001*u0/sqrt(u0[0]**2+u0[1]**2),
                wireframe = True,
                title = "the approximation of u",
                rescale = True , 
                axes = False, 
                basename = "deflection")
visual_u1 = plot(u0[0],
		 mesh,
		 wireframe = False,
		 title="the approximation of u1",
		 rescale = True ,
		 axes = False,
		 basename = "deflection")
visual_u2 = plot(u0[1],
		 mesh,
		 wireframe = False,
		 title = "the approximation of u2",
		 rescale = True,
		 axes = False,
		 basename = "deflection")
#visual_u1.set_min_max(-5,5)
#visual_u1.show_scalarbar()
#visual_u1.set_viewangle(2)
#visual_u1.zoom(2)
min_w = U_0.vector().array().min()
max_w = U_0.vector().array().max()
visual_u1.set_min_max(0.5*min_w,0.5*max_w)
visual_u1.elevate(-20)
visual_mesh.write_png("Image/mesh:%gx%g"%(nx,ny))
visual_u.write_png("Image/P%g_u_%gx%g"%(num,nx,ny))
visual_u1.write_png("Image/P%g_u1_%gx%g"%(num,nx,ny))
#visual_u1.set_contour("Data/u1_%gx%g.vtk"%(nx,ny))
#visual_u1.write_vtk("Data/u1_%gx%g.vtk"%(nx,ny))
visual_u2.write_png("Image/P%g_u2_%gx%g"%(num,nx,ny))


# Plot exact solution
visual_ue1 = plot(u_ex[0],
		  wireframe = False,
		  title="the exact solution of u1",
		  rescale = True,
		  axes = False,
		  basename = "deflection",
		  legend ="u1")
visual_ue2 = plot(u_ex[1],
		  wireframe = False,
		  title="the exact solution of u2",
		  rescale = True,
		  axes = False,
		  basename = "deflection",
		  legend ="u2")
visual_ue1.elevate(-20)
visual_ue1.write_png("Image/P%gu1_exact_%gx%g.png"%(num,nx,ny))
visual_ue2.write_png("Image/P%gu2_exact_%gx%g.png"%(num,nx,ny))
interactive()

#Define L2 norm and H1 norm relative errors 
u1_error = (u0[0]-u_ex[0])**2*dx
u2_error = (u0[1]-u_ex[1])**2*dx
u1_ex_L2 = u_ex[0]**2*dx
u2_ex_L2 = u_ex[1]**2*dx
L2_error_u1 = sqrt(assemble(u1_error)/assemble(u1_ex_L2))
L2_error_u2 = sqrt(assemble(u2_error)/assemble(u2_ex_L2))

a_u1_0=inner(curl(u0[0]-u_ex[0]),curl(u0[0]-u_ex[0]))*dx + inner(div(u0[0]-u_ex[0]),div(u0[0]-u_ex[0]))*dx
a_u2_0=inner(curl(u0[1]-u_ex[1]),curl(u0[1]-u_ex[1]))*dx + inner(div(u0[1]-u_ex[1]),div(u0[1]-u_ex[1]))*dx
a_u1_1=inner(curl(u_ex[0]),curl(u_ex[0]))*dx + inner(div(u_ex[0]),div(u_ex[0]))*dx
a_u2_1=inner(curl(u_ex[1]),curl(u_ex[1]))*dx + inner(div(u_ex[1]),div(u_ex[1]))*dx
Curl_Div_error_u1 = sqrt(assemble(a_u1_0)/assemble(a_u1_1))
Curl_Div_error_u2 = sqrt(assemble(a_u2_0)/assemble(a_u2_1))


print("h=" ,CellSize(mesh))
print("The number of cells(triangle) in the mesh:" ,mesh.num_cells())
print("The number of vertices in the mesh:" ,mesh.num_vertices())
print("L2_error_u1=" ,L2_error_u1)
print("L2_error_u2=" ,L2_error_u2)
print("Curl_error_u1=" ,Curl_Div_error_u1)
print("Curl_error_u2=" ,Curl_Div_error_u2)
print("Total time is=",time.time() - t0)
print("Assemble time is=",t2-t1)
print("Interpolate exact u time is=",t4 - t3)

file_object = open('Error_P%g.txt'%(num),'a')
file_object.writelines("The number of cells(triangle) in the mesh:%g" %(mesh.num_cells()))
file_object.writelines("The number of vertices in the mesh:%g"%(mesh.num_vertices()))
file_object.writelines("Mesh:%gx%gP%g\n"%(nx,ny,num))
file_object.writelines("L2_error_u1= %g\n"%(L2_error_u1))
file_object.writelines("L2_error_u2= %g\n"%(L2_error_u2))
file_object.writelines("Curl_error_u1= %g\n" %(Curl_Div_error_u1))
file_object.writelines("Curl_error_u2= %g\n\n"%(Curl_Div_error_u2))
file_object.close()

file_output = open('Ratio_P%g.txt'%(num),'a')
E = np.array([nx,L2_error_u1,L2_error_u2,Curl_Div_error_u1,Curl_Div_error_u2])
print(E)
file_output.writelines("%g %g %g %g %g\n"%(E[0],E[1],E[2],E[3],E[4]))


