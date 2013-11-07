# _*_ coding:utf-8 _*_
"""
This demo program solves Maxwell equations in 3D
    curl curl u - omega^2 u = f    in  \Omega
    u x n = 0   on \Omega

on the unit cubi with f given by
    
    f = (f1,f2,f3)'
    f1 = -exp(x[0] + x[1] + x[2])*sin(pi*x[1])*sin(pi*x[2])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
    f2 = -exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[2])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
    f3 = -exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[1])*(sin(pi*x[2]) + pi*cos(pi*x[2]))
and exact solution given by
    u1 = -exp(x[0] + x[1] + x[2])*sin(pi*x[1])*sin(pi*x[2])*(sin(pi*x[0]) + pi*cos(pi*x[0]))
    u2 = -exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[2])*(sin(pi*x[1]) + pi*cos(pi*x[1]))
    u3 = -exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[1])*(sin(pi*x[2]) + pi*cos(pi*x[2]))
"""

from dolfin import *
import numpy, sys
from numpy import pi
import scitools.BoxField
import scitools.easyviz as ev

# Create mesh and define function space
# For example: python Maxwell_S3D.py  8 8 8
nx = int(sys.argv[1])
ny = int(sys.argv[2])
nz = int(sys.argv[3])
mesh=UnitCube( nx, ny, nz)
U_h = VectorFunctionSpace(mesh, "Lagrange", 2)

# Define trial and test function
u = TrialFunction(U_h)
v = TestFunction(U_h)
 

# Define boundary condition(x = 0 or x = 1 or y=0 or y=1) u x n = u_1*n_2 - u_2*n1 = 0 
def boundary1(x):
    return x[0] > 1.0 - DOLFIN_EPS or x[0] < DOLFIN_EPS
def boundary2(x):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS  
def boundary3(x):
    return x[2] > 1.0 - DOLFIN_EPS or x[2] < DOLFIN_EPS

u0 = Expression("0.0")
bc1 = DirichletBC(U_h.sub(1), u0, boundary1)
bc2 = DirichletBC(U_h.sub(0), u0, boundary2)
bc3 = DirichletBC(U_h.sub(0), u0, boundary3)
bc4 = DirichletBC(U_h.sub(2), u0, boundary1)
bc5 = DirichletBC(U_h.sub(2), u0, boundary2)
bc6 = DirichletBC(U_h.sub(1), u0, boundary3)
bc=[bc1,bc2,bc3,bc4,bc5,bc6]
print "bc is created!"
	
# Define parameter and f
omega = Constant("1.0")
f = Expression(("-exp(x[0] + x[1] + x[2])*sin(pi*x[1])*sin(pi*x[2])*(sin(pi*x[0]) + pi*cos(pi*x[0]))",\
                "-exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[2])*(sin(pi*x[1]) + pi*cos(pi*x[1]))",\
                "-exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[1])*(sin(pi*x[2]) + pi*cos(pi*x[2]))"),pi = pi)
   
# Define variational form
F = inner(curl(u),curl(v))*dx - omega**2*inner(u,v)*dx - inner(f,v)*dx
a = lhs(F)
L = rhs(F)
# Solve
(A,b) = assemble_system(a,L,bc)
print "A and b is assembled!"
u0 = Function(U_h)
solver = KrylovSolver("cg", "ilu")
solver.parameters["absolute_tolerance"] = 1E-15
solver.parameters["relative_tolerance"] = 1E-15
solver.parameters["maximum_iterations"] = 1000
set_log_level(DEBUG)
solve(A, u0.vector(), b)
	
# Save solution in VTK format

file = File("Data/Maxwell S3D_%gx%g.pvd"%(nx,ny))
file << u0

print "u0 be solved!"
# Plot solution 
"""
visual_mesh = plot(mesh,title = "Mesh")
visual_u0 = plot(u0,wireframe =  True,title = "the approximation of u",rescale = True , axes = True, basename = "deflection" ,legend ="u")
visual_u1 = plot(u0[0],wireframe = True,title="the approximation of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_u2 = plot(u0[1],wireframe = True,title="the approximation of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_u3 = plot(u0[2],title = "the approximation of u3",rescale = True , axes = True, basename = "deflection" ,legend ="u3")
visual_mesh.write_png("Image/mesh_%gx%gx%g.png"%(nx,ny,nz))
visual_u1.elevate(-65) #tilt camera -65 degree(latitude dir)
#visual_u1.set_min_max(0,0.5*max_u0[0]) #color scale
#visual_u1.update(u0[0])  #bring setting above into action
visual_u1.write_png("Image/u1_%gx%gx%g.png"%(nx,ny,nz))
visual_u2.write_png("Image/u2_%gx%gx%g.png"%(nx,ny,nz))
visual_u3.write_png("Image/u3_%gx%gx%g.png"%(nx,ny,nz))
"""
# Define a higher-order approximation to the exact solution
Ue=VectorFunctionSpace(mesh,"Lagrange",degree=5)
u_exact = Expression(("exp(x[0] + x[1] + x[2])*sin(pi*x[1])*sin(pi*x[2])*(sin(pi*x[0]) + pi*cos(pi*x[0]))",\
                      "exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[2])*(sin(pi*x[1]) + pi*cos(pi*x[1]))",\
                      "exp(x[0] + x[1] + x[2])*sin(pi*x[0])*sin(pi*x[1])*(sin(pi*x[2]) + pi*cos(pi*x[2]))"),pi = pi)
   
u_ex = interpolate(u_exact,Ue)

"""
visual_ue1 = plot(u_ex[0],wireframe = True,title="the exact solution of u1",rescale = True , axes = True, basename = "deflection" ,legend ="u1")
visual_ue2 = plot(u_ex[1],wireframe = True,title="the exact solution of u2",rescale = True , axes = True, basename = "deflection" ,legend ="u2")
visual_ue3 = plot(u_ex[2],wireframe = True,title="the exact solution of u3",rescale = True , axes = True, basename = "deflection" ,legend ="u3")
visual_ue1.write_png("Image/u1_exact_%gx%gx%g.png"%(nx,ny,nz))
visual_ue2.write_png("Image/u2_exact_%gx%gx%g.png"%(nx,ny,nz))
visual_ue3.write_png("Image/u3_exact_%gx%gx%g.png"%(nx,ny,nz))
interactive()
"""

#Define L2 norm and H1 norm relative errors 
u1_error = (u0[0]-u_ex[0])**2*dx
u2_error = (u0[1]-u_ex[1])**2*dx
u3_error = (u0[2]-u_ex[2])**2*dx
u1_ex_L2 = u_ex[0]**2*dx
u2_ex_L2 = u_ex[1]**2*dx
u3_ex_L2 = u_ex[2]**2*dx

L2_error_u1 = sqrt(assemble(u1_error)/assemble(u1_ex_L2))
L2_error_u2 = sqrt(assemble(u2_error)/assemble(u2_ex_L2))
L2_error_u3 = sqrt(assemble(u3_error)/assemble(u3_ex_L2))
Curl_error_u1 = sqrt(assemble(inner(curl(u0[0]-u_ex[0]),curl(u0[0]-u_ex[0]))*dx)/assemble(inner(curl(u_ex[0]),curl(u_ex[0]))*dx))
Curl_error_u2 = sqrt(assemble(inner(curl(u0[1]-u_ex[1]),curl(u0[1]-u_ex[1]))*dx)/assemble(inner(curl(u_ex[1]),curl(u_ex[1]))*dx))
Curl_error_u3 = sqrt(assemble(inner(curl(u0[2]-u_ex[2]),curl(u0[2]-u_ex[2]))*dx)/assemble(inner(curl(u_ex[2]),curl(u_ex[2]))*dx))


print "h=" ,CellSize(mesh)
print "The number of cells(triangle) in the mesh:" ,mesh.num_cells()
print "The number of vertices in the mesh:" ,mesh.num_vertices()
print "L2_error_u1=" ,L2_error_u1
print "L2_error_u2=" ,L2_error_u2
print "L2_error_u3=" ,L2_error_u3
print "L2_error_u1=" ,Curl_error_u1
print "L2_error_u2=" ,Curl_error_u2
print "L2_error_u3=" ,Curl_error_u3

file_object = open('Error.txt','a')
file_object.writelines("Mesh:%gx%g\n"%(nx,ny))
file_object.writelines("L2_error_u1= %g\n"%(L2_error_u1))
file_object.writelines("L2_error_u2= %g\n"%(L2_error_u2))
file_object.writelines("L2_error_u3= %g\n"%(L2_error_u3))
file_object.writelines("Curl_error_u1= %g\n" %(Curl_error_u1))
file_object.writelines("Curl_error_u2= %g\n\n"%(Curl_error_u2))
file_object.writelines("Curl_error_u3= %g\n\n"%(Curl_error_u3))
file_object.close()

file_output = open('Ratio.txt','a')
E = np.array([nx,L2_error_u1,L2_error_u2,L2_error_u3,Curl_error_u1,Curl_error_u2,Curl_error_u3])
print E
file_output.writelines("%g %g %g %g\n"%(E[0],E[1],E[2],E[3],E[4],E[5],E[6]))
