"""

This is the solver code for the fully-coupled poroelasticity formulation
and reads in the required information from the driver script before solving 
the three PDEs (conservation of mass, darcy's law, conservation of momentum)   
using a Mixed Finite Element approach (i.e. a monolithic solve).

The transfer of functions from the driver script to solver script is handled
by reading and writing the files in each iteration. These files are saved in
the directory called Exchange_Files, which is deleted at the end of the 
simulation.

Primary unkowns are considered to be perturbations from the litho- and hydro-
static conditions, as is common in poroelasticity. Pressure or stress/strain 
dependent variables are related to either the perturbation or the total value
(i.e. perturbation + static condition).

Refer to the User Guide for more information on each section of the code below.

"""

######### INITIALIZE CODE #################################################

from fenics import *
import numpy as np
import ufl

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

###########################################################################

######### DEFINE MODEL TYPE ###############################################

model_type = np.loadtxt('Exchange_Files/model_type.txt')

Linear = model_type[0]
Nonlinear = model_type[1]

if Linear == 1:
    weight = Constant(0.0)
    linear_flag = 1

if Nonlinear == 1:
    weight = Constant(1.0)
    linear_flag = 0

###########################################################################

######### MESHING #########################################################

Domain_params = np.loadtxt('Exchange_Files/Domain_params.txt')

x0 = Domain_params[0]
x1 = Domain_params[1]
y0 = Domain_params[2]
y1 = Domain_params[3]
z0 = Domain_params[4]
z1 = Domain_params[5]
nx = Domain_params[6]
ny = Domain_params[7]
nz = Domain_params[8]

mesh = Mesh('Exchange_Files/Problem_mesh.xml')

###########################################################################

######### FUNCTION SPACES #################################################

### Field variable function spaces ###
ele_p  = FiniteElement("DG",  mesh.ufl_cell(), 0) # pressure
ele_q = FiniteElement("BDM", mesh.ufl_cell(), 1) # fluid flux
ele_us  = VectorElement("CG",  mesh.ufl_cell(), 1) # solid displacement
W = MixedElement([ele_p, ele_q, ele_us])
W = FunctionSpace(mesh, W)

### Function spaces for other variables ###
Z = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "DG", 0)
S = TensorFunctionSpace(mesh, "DG", 0)
P = FunctionSpace(mesh, "DG", 0)

###########################################################################

######### PARAMETERS ######################################################

Params = np.loadtxt('Exchange_Files/Params.txt')

g = Params[0]
k = Params[1]
mu = Params[2]
phi_0 = Params[3]
phi_min = Params[4]
p_0 = Params[5]
rho_0 = Params[6]
beta_m = Params[7]
beta_f = Params[8]
beta_s = Params[9]
nu = Params[10]
K = Params[11]
G = Params[12]
alpha = Params[13]

### Hydrostatic Condition ###
p_h = project(Expression('rho_0*g*(z1-x[2])',degree=1,rho_0=rho_0,g=g,z1=z1),Z)

### Solution dependent properties ###
# Total Stress
def sigma(us,p,alpha,K,G):
    sigma_val = sigma_e(us,K,G) + alpha*p*Identity(len(us))
    return sigma_val

# Effective Stress
def sigma_e(us,K,G):
    sigma_e_val = -2.0*G*epsilon(us) - (K-2.0/3.0*G)*epsilon_v(us)*Identity(len(us))
    return sigma_e_val

# Strain
def epsilon(us):
    epsilon_val = 1.0/2.0*(grad(us)+grad(us).T)
    return epsilon_val

# Volumetric Strain
def epsilon_v(us):
    epsilon_v_val = div(us)
    return epsilon_v_val

if Linear == 1:
    # Fluid Density
    def rho_f(p,p_0,p_h,rho_0,beta_f):
        rho_val = Constant(rho_0)
        return rho_val

    # Porosity
    def phi(alpha,us,p,beta_s,phi_0,phi_min,t):
        phi_val = Constant(phi_0)
        return phi_val

if Nonlinear == 1:
    # Fluid Density
    def rho_f(p,p_0,p_h,rho_0,beta_f):
        rho_val = Constant(rho_0)*exp(Constant(beta_f)*(p+p_h-Constant(p_0)))
        return rho_val

    # Porosity
    def phi(alpha,us,p,beta_s,phi_0,phi_min,t):
        phi_val = alpha - (alpha-phi_0)*exp(-(epsilon_v(us)+beta_s*p))
        phi_val = conditional(ge(phi_val,phi_min),phi_val,Constant(phi_min))
        return phi_val

###########################################################################

######### TIME PARAMETERS #################################################

Time_params = np.loadtxt('Exchange_Files/Time_params.txt')

tend = Time_params[0]
nsteps = Time_params[1]
dt = Time_params[2]

# Read in the time
t = np.loadtxt('Exchange_Files/t.txt')
t = np.array(t).tolist()

###########################################################################

######### BOUNDARY CONDITIONS  ############################################

### Boundary Conditions ###
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],x0)
left_boundary = LeftBoundary()

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],x1)
right_boundary = RightBoundary()

class BackBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],y0)
back_boundary = BackBoundary()

class FrontBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],y1)
front_boundary = FrontBoundary()

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],z0)
bottom_boundary = BottomBoundary()

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],z1)
top_boundary = TopBoundary()

boundary_facet_function = MeshFunction('size_t', mesh, 2)
boundary_facet_function.set_all(0)
left_boundary.mark(boundary_facet_function,1)
right_boundary.mark(boundary_facet_function,2)
back_boundary.mark(boundary_facet_function,3)
front_boundary.mark(boundary_facet_function,4)
bottom_boundary.mark(boundary_facet_function,5)
top_boundary.mark(boundary_facet_function,6)

def GetBoundaryConditions(t):
    
    bcs = []

    # Left Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #     boundary_facet_function, 1)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #     boundary_facet_function, 1)) 

    # Right Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 2)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 2)) 

    # Back Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 3)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 3)) 

    # Front Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 4)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 4)) 

    # Bottom Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 5)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 5)) 

    # Top Boundary
    # # Flux Boundary (normal)
    # bcs.append(DirichletBC(W.sub(1), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 6)) 
    # # Displacement Boundary
    # bcs.append(DirichletBC(W.sub(2), Constant((0.0,0.0,0.0)), \
    #    boundary_facet_function, 6)) 

    return bcs

###########################################################################

######### PREVIOUS SOLUTIONS ##############################################

X_n = Function(W)
X_m = Function(W)
phi_n = Function(Z)

file_X_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_n.h5','r')
file_X_m = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_m.h5','r')
file_phi_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/phi_n.h5','r')
file_X_n.read(X_n,'X_n')
file_X_m.read(X_m,'X_m')
file_phi_n.read(phi_n,'phi_n')
del file_X_n
del file_X_m
del file_phi_n

###########################################################################

######### SOLVER SET-UP  ##################################################

U = TrialFunction(W)
V = TestFunction(W)

n = FacetNormal(mesh)
norm = as_vector([n[0], n[1], n[2]])

ff = Constant(0.0)	# fluid source

X = Function(W)

# dx = Measure("dx")(subdomain_data=***subdomain-face-function-name***)
ds = Measure("ds")(subdomain_data=boundary_facet_function)

# Function of ones to evaluate grad(rho_f) in weak form
ones_func = project(Constant(1.0),Z,solver_type='gmres')

def WeakForm(U,V,X_n,t):

    p, q, us = split(U)
    Pt, Qt, Ust = split(V)
    p_n, q_n, us_n = split(X_n)
    p_m, q_m, us_m = split(X_m)

    ### Weak Forms ###
    # Conservation of Mass
    CoMass_l_1 = rho_f(p_m,p_0,p_h,rho_0,beta_f)*Constant(alpha)\
        *epsilon_v(us)*Pt*dx
    CoMass_l_2 = rho_f(p_m,p_0,p_h,rho_0,beta_f)*((Constant(alpha)\
        -phi(alpha,us_m,p_m,beta_s,phi_0,phi_min,t))*Constant(beta_s)\
        + phi(alpha,us_m,p_m,beta_s,phi_0,phi_min,t)*Constant(beta_f))*p\
        *Pt*dx
    CoMass_l_3 = dt*rho_f(p_m,p_0,p_h,rho_0,beta_f)*div(q)*Pt*dx
    CoMass_l_4 = dt*Constant(weight)*inner(q,\
        grad(rho_f(p_m,p_0,p_h,rho_0,beta_f)*ones_func))*Pt*dx
    CoMass_l = CoMass_l_1 + CoMass_l_2 + CoMass_l_3 + CoMass_l_4
    CoMass_r_1 = dt*ff*Pt*dx
    CoMass_r_2 = rho_f(p_m,p_0,p_h,rho_0,beta_f)*Constant(alpha)\
        *epsilon_v(us_n)*Pt*dx
    CoMass_r_3 = rho_f(p_m,p_0,p_h,rho_0,beta_f)*((Constant(alpha)\
        -phi(alpha,us_m,p_m,beta_s,phi_0,phi_min,t))*Constant(beta_s) \
        + phi(alpha,us_m,p_m,beta_s,phi_0,phi_min,t)*Constant(beta_f))*p_n\
        *Pt*dx
    CoMass_r = CoMass_r_1 + CoMass_r_2 + CoMass_r_3

    # Darcy's Law
    DL_l = mu/k*inner(q,Qt)*dx - p*div(Qt)*dx
    # DL_r = -Constant(***pressure***)*inner(Qt,norm)*ds(-)

    # Conservation of Momentum
    CoMom_l = inner(sigma(us,p,alpha,K,G),grad(Ust))*dx
    # CoMom_r = inner(Constant((0.0,***loading***)),Ust)*ds(-)

    A = CoMass_l + DL_l + CoMom_l
    B = CoMass_r #+ DL_r + CoMom_r

    return A,B

def LinearSolver(U,V,X_n,t,bcs):
    a,L = WeakForm(U,V,X_n,t)
    A, b = assemble_system(a,L,bcs)
    solve(A, X.vector(), b, 'mumps')
    return X

###########################################################################

######### ASSEMBLE, SOLVE AND SAVE ########################################

bcs = GetBoundaryConditions(t)

### Solve ###
X = LinearSolver(U,V,X_n,t,bcs)

file_X = HDF5File(mesh.mpi_comm(),'Exchange_Files/X.h5','w')
file_X.write(X,'X')
del file_X

solver_finish = 1
np.savetxt('Exchange_Files/solver_finish.txt',[solver_finish])

###########################################################################
