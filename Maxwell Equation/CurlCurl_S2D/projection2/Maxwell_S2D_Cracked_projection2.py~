# _*_ coding:utf-8 _*_
"""
We solved use mixed finite element method.
This demo program solves Maxwell equation in 2D
    curl curl u - omega^2 u = f    in  \Omega
    div u = g  in  \Omega
    u x n = 0   on \Omega

on the unit square with  f given by
    
    f = (f1,f2)'
    f1 = -exp(x[0] + x[1])*sin(pi*x[1])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
    f2 = -exp(x[0] + x[1])*sin(pi*x[0])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
and exact solution given by
    u1 = exp(x[0] + x[1])*sin(pi*x[1])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
    u2 = exp(x[0] + x[1])*sin(pi*x[0])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
Domain:
    (-1,1) ______________________(1,1)
	   |                    |
	   |                    |
	   |                    |
	   |                    |
	   |          __________|(1.0)  
	   |           (0,0)    |
	   |                    |
	   |                    |
	   |                    |
   (-1,-1) |____________________|(0,-1)   
  
"""
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
num = int(sys.argv[3])
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 2, 2)
editor.init_vertices(10)
editor.init_cells(8)
editor.add_vertex(0, 0.0, 0.0)
editor.add_vertex(1, 1.0, 0.0)
editor.add_vertex(2, 1.0, 1.0)
editor.add_vertex(3, 0.0, 1.0)
editor.add_vertex(4, -1.0, 1.0)
editor.add_vertex(5, -1.0, 0.0)
editor.add_vertex(6, -1.0, -1.0)
editor.add_vertex(7, 0.0, -1.0)
editor.add_vertex(8, 1.0, -1.0)
editor.add_vertex(9, 1.0, 0.0)
editor.add_cell(0, 0, 1, 3)
editor.add_cell(1, 1, 2, 3)
editor.add_cell(2, 0, 3, 5)
editor.add_cell(3, 3, 4, 5)
editor.add_cell(4, 0, 5, 7)
editor.add_cell(5, 5, 6, 7)
editor.add_cell(6, 0, 7, 9)
editor.add_cell(7, 7, 8, 9)
editor.close()
num_refine = int(math.log(int(sys.argv[1]),2))
h=0.5**num_refine
for i in range(num_refine):
	mesh=refine(mesh)


# Define function space
U_h = VectorFunctionSpace(mesh, "Lagrange", num) #num次元
B = VectorFunctionSpace(mesh, "Bubble", 3)
Q_h = FunctionSpace(mesh, "CG", 1)  #projection space of R(div)
W_h = FunctionSpace(mesh, "CG", 1)  #projection space of R(curl)
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
        return on_boundary and ( abs(x[0]-1)<tol or abs(x[0]+1)<tol)

w0 = Constant("0.0")
u_init = Constant(('0.0','0.0'))
bc0 = DirichletBC(Mix_h.sub(0).sub(0), w0, u0_boundary)
bc1 = DirichletBC(Mix_h.sub(0).sub(1), w0, u1_boundary)
bc2 = DirichletBC(Mix_h.sub(1), u_init, project_boundary)
bc_0 = DirichletBC(Mix_h.sub(2), w0, project_boundary)
bc = [bc0,bc1,bc2,bc_0] 
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
        value[0] = -exp(x[0] + x[1])*sin(pi*x[1])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
        value[1] = -exp(x[0] + x[1])*sin(pi*x[0])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
    def value_shape(self):
        return (2,)
f = f_Expression()
Ue=VectorFunctionSpace(mesh,"Lagrange",degree=3)
f = interpolate(f,Ue)


class g_Expression(Expression):
    def eval(self, value, x):
        value[0] = 2.0*exp(x[0] + x[1])*((1.0 - pi**2)*sin(pi*x[0])*sin(pi*x[1]) + pi*sin(pi*x[0] + pi*x[1]))

g = g_Expression()
g = interpolate(g,FunctionSpace(mesh,"Lagrange",degree= 3))
plot(g)


class u_Expression(Expression):
    def eval(self, value, x):
        value[0] = exp(x[0] + x[1])*sin(pi*x[1])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
        value[1] = exp(x[0] + x[1])*sin(pi*x[0])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
    def value_shape(self):
        return (2,)
u_exact = u_Expression()

# Define variational problem
U = TrialFunction(Mix_h)
V = TestFunction(Mix_h)
(u,u_b,DivRu,CurlRu) = split(U)
(v,v_b,q,w) = split(V)

# Projection varitional problem
s_g = TrialFunction(Q_h)
v_q = TestFunction(Q_h)
a_s = inner(s_g,v_q)*dx
L_s = inner(g,v_q)*dx
Sg = Function(Q_h)
bc_sg = DirichletBC(Q_h, w0, project_boundary)
(A_s, b_s) = assemble_system(a_s, L_s, bc_sg)
solve(A_s, Sg.vector(), b_s)
plot(Sg)

#S_g = project(g,Q_h)
a = inner(curl(CurlRu),v + v_b)*dx - inner(grad(DivRu),v + v_b)*dx - omega**2*inner(u + u_b,v + v_b)*dx + inner(DivRu,q)*dx + inner(u + u_b,grad(q))*dx+ inner(CurlRu,w)*dx - inner(u + u_b,curl(w))*dx
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

u0 = u0 + ub
# Save solution in VTK format
file = File("Data/Maxwell S2D_%gx%gP%g.pvd"%(nx,ny,num))
#file << u0


# Plot solution(mesh,surf,contour)

visual_mesh = plot(mesh,title = "Mesh")
visual_u = plot(u0,wireframe = True,title = "the approximation of u",rescale = True , axes = True, basename = "deflection" ,legend ="u0")
visual_u1 = plot(u0[0],wireframe = False,title="the approximation of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_u2 = plot(u0[1],wireframe = False,title = "the approximation of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_mesh.write_png("Image/P%g_mesh_%gx%g.png"%(num,nx,ny))
visual_u.write_png("Image/P%g_u_%gx%g"%(num,nx,ny))
visual_u1.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_u2.elevate(-65)
visual_u1.write_png("Image/P%g_u1_%gx%g.png"%(num,nx,ny))
visual_u2.write_png("Image/P%g_u2_%gx%g.png"%(num,nx,ny))
"""
X = 0; Y = 1;
u1,u2 = u0.split(deepcopy = True)

us = u0[0] if u1.ufl_element().degree() == 1 else \
     interpolate(u1, FunctionSpace(mesh, 'Lagrange', 1))
u_box = scitools.BoxField.dolfin_function2BoxField(us, mesh, (nx,ny), uniform_mesh=True)


ev.contour(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values, 20, 
           savefig='Image/Contour of u1_P%g(mesh:%g-%g).png'% (num,nx,ny), title='Contour plot of u1', colorbar='on')
ev.figure()
ev.surf(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values, shading='interp', colorbar='on', 
        title='surf plot of u1', savefig='Image/Surf of u1_P%g(mesh:%g-%g).png'% (num,nx,ny))
ev.figure()
ev.mesh(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values,colorbar='on',shading='interp',
        title='mesh plot of u1', savefig='Image/Mesh of u1_P%g(mesh:%g-%g).png'% (num,nx,ny))
"""
# Define a higher-order approximation to the exact solution
t3 = time.time()
Ue=VectorFunctionSpace(mesh,"Lagrange",degree= 3)
u_ex = interpolate(u_exact,Ue)
t4 = time.time()
# Plot exact solution

visual_ue1 = plot(u_ex[0],wireframe = False,title="the exact solution of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_ue2 = plot(u_ex[1],wireframe = False,title="the exact solution of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_ue1.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_ue2.elevate(-65) #tilt camera -65 degree(latitude dir)
visual_ue1.write_png("Image/P%gu1_exact_%gx%g.png"%(num,nx,ny))
visual_ue2.write_png("Image/P%gu2_exact_%gx%g.png"%(num,nx,ny))
interactive()
"""
u1_ex,u2_ex = u_ex.split(deepcopy = True)
u1_ex = u1_ex if u1_ex.ufl_element().degree() == 1 else \
     interpolate(u1_ex, FunctionSpace(mesh, 'Lagrange', 1))
u_box = scitools.BoxField.dolfin_function2BoxField(u1_ex, mesh, (nx,ny), uniform_mesh=True)

ev.figure()
ev.contour(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values, 20, 
           savefig='Image/Contour of exact solution u1_P%g(mesh:%g-%g).png'% (num,nx,ny), title='Contour plot of exact u1',colorbar='on')
ev.figure()
ev.surf(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values, shading='interp', colorbar='on', 
        title='surf plot of exact u1', savefig='Image/Surf of exact solution u1_P%g(mesh:%g-%g).png'% (num,nx,ny))
ev.figure()
ev.mesh(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values, colorbar="on",
        title='mesh plot of exact u1', savefig='Image/Mesh of exact solution u1_P%g(mesh:%g-%g).png'% (num,nx,ny))
"""

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


