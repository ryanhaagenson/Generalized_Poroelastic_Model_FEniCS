"""

GPMF

A Generalized Poroelastic Model using FEniCS

Code for a fully-coupled poroelastic formulation: This solves the three
PDEs (conservation of mass, darcy's law, conservation of momentum) using 
a Mixed Finite Element approach (i.e. a monolithic solve). The linear 
model uses a constant fluid density and porosity. The nonlinear model uses 
a d(phi)/dt porosity model to update porosity in each time step, which is 
derived from solid continuity assuming small strains. Fluid density is also
allowed to vary.

Primary unkowns are considered to be perturbations from the litho- and 
hydro-static conditions, as is common in poroelasticity. Pressure or 
stress/strain dependent variables are related to either the perturbation 
or the total value (i.e. perturbation + static condition).

Refer to the User Guide for more information on each section of the code 
below.

"""

######### INITIALIZE CODE #################################################

from fenics import *
import numpy as np
import ufl
from time import gmtime, strftime

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

if MPI.rank(mpi_comm_world()) == 0:
    print "Start date and time:"
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())

###########################################################################

######### DEFINE MODEL TYPE ###############################################

Linear = 0
Nonlinear = 1

if Linear + Nonlinear != 1:
    if MPI.rank(mpi_comm_world()) == 0:
        print "You must choose a model type."
    quit()

if Linear == 1:
    if MPI.rank(mpi_comm_world()) == 0:
        print "Running the Linear Poroelastic Model."
    weight = Constant(0.0)
    linear_flag = 1

if Nonlinear == 1:
    if MPI.rank(mpi_comm_world()) == 0:
        print "Running the Nonlinear Poroelastic Model."
    weight = Constant(1.0)
    linear_flag = 0

###########################################################################

######### DOMAIN AND SUBDOMAINS ###########################################

### Domain Constants ###
x0 = 0.0        # Minimum x
x1 = 1.0        # Maximum x
y0 = 0.0        # Minimum y
y1 = 1.0        # Maximum y
z0 = 0.0        # Minimum z
z1 = 1.0        # Maximum z
nx = 1          # Number of cells in x-direction
ny = 1          # Number of cells in y-direction
nz = 1          # Number of cells in z-direction

###########################################################################

######### MESHING #########################################################

if MPI.rank(mpi_comm_world()) == 0:
    print ('Building mesh...')
mesh = BoxMesh(Point(x0,y0,z0),Point(x1,y1,z1),nx,ny,nz)

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

######### DEFINE PARAMETERS AND RELATIONSHIPS #############################

### Physical Constants ###
g = 9.81                                # Gravity                (m/sec^2)

### Hydraulic Parameters ###
k = 1.0                                 # Permeability           (m^2)
mu = 1.0                                # Viscosity              (Pa*s)
phi_0 = 0.5                             # Initial Porosity       (-)
phi_min = 0.01                          # Minimum allowed Porosity
p_0 = 0.0                               # Reference Pressure     (Pa)
rho_0 = 1.0                             # Reference Density      (kg/m^3)

### Mechanical Parameters ###
beta_m = 1.0                            # Matrix Compressibility (Pa^-1)
beta_f = 1.0                            # Fluid Compressibility  (Pa^-1)
beta_s = 1.0                            # Solid Compressibility  (Pa^-1)
nu = 0.25                               # Poisson's Ratio        (-)
K = beta_m**(-1)                        # Bulk Modulus           (Pa)
G = 3.0/2.0*K*(1.0-2.0*nu)/(1.0+nu)     # Shear Modulus          (Pa)
alpha = 1.0 - beta_s/beta_m             # Biot Coefficient       (-)

### Hydrostatic Condition ###
p_h = project(Expression('rho_0*g*(z1-x[2])',degree=1,rho_0=rho_0,g=g,\
    z1=z1),Z)

### Solution dependent properties ###
# Total Stress
def sigma(us,p,alpha,K,G):
    sigma_val = sigma_e(us,K,G) + alpha*p*Identity(len(us))
    return sigma_val

# Effective Stress
def sigma_e(us,K,G):
    sigma_e_val = -2.0*G*epsilon(us) - (K-2.0/3.0*G)*epsilon_v(us)*\
    Identity(len(us))
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
        rho_val = Constant(rho_0)*exp(Constant(beta_f)*(p+p_h-\
            Constant(p_0)))
        return rho_val

    # Porosity
    def phi(alpha,us,p,beta_s,phi_0,phi_min,t):
        phi_val = alpha - (alpha-phi_0)*exp(-(epsilon_v(us)+beta_s*p))
        phi_val = conditional(ge(phi_val,phi_min),phi_val,\
            Constant(phi_min))
        return phi_val

###########################################################################

######### TIME PARAMETERS  ################################################

### Time parameters ###
tend = 1.0
nsteps = 10
dt = tend/nsteps

###########################################################################

######### INITIAL CONDITIONS  #############################################

### Initial Condition ###
X_i = Expression(
        (
            "0.0",       		# p    
            "0.0","0.0","0.0", 	# (q1, q2, q3)
            "0.0","0.0","0.0"  	# (us1, us2, us3)
        ),degree = 2)
X_n = interpolate(X_i,W)

# Initial porosity
phi_n = interpolate(Constant(phi_0),Z)

# Initial Picard solution estimate
X_m = interpolate(X_i,W)

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

######### SOLVER SET-UP  ##################################################

U = TrialFunction(W)
V = TestFunction(W)

n = FacetNormal(mesh)
norm = as_vector([n[0], n[1], n[2]])

ff = Constant(0.0)  # fluid source

X = Function(W)

density_save = Function(Z)
porosity_save = Function(Z)

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

######### OUTPUT  #########################################################

pressure_file = XDMFFile('General/pressure.xdmf')
flux_file = XDMFFile('General/flux.xdmf')
disp_file = XDMFFile('General/disp.xdmf')
density_file = XDMFFile('General/density.xdmf')
porosity_file = XDMFFile('General/porosity.xdmf')

###########################################################################

######### TIME LOOP #######################################################

t = 0.0

if MPI.rank(mpi_comm_world()) == 0:
    print ('Starting Time Loop...')

### Time Loop ###
for n in range(nsteps):

    MPI.barrier(mpi_comm_world())

    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print "###############"
        print ""
        print "NEW TIME STEP"
        print "Time =", t
        print ""
        print np.round(t/tend*100.0,6),"% Complete"

    ### Convergence criteria ###
    reltol = 1E-4			    # Relative error tolerance for Picard
    rel_error_max_global = 9999	# Initialize relative error 
    max_iter = 100			    # Maximum Picard iterations
    omega = 1.0				    # Relaxation coefficient

    iter = 0

    if linear_flag == 0:
        if MPI.rank(mpi_comm_world()) == 0:
            print "Entering Picard Iteration:"
            print ""

    # Get boundary conditions for this time
    bcs = GetBoundaryConditions(t)

    ### Picard Iteration Loop ###
    while (rel_error_max_global > reltol):

        if linear_flag == 0:
            if MPI.rank(mpi_comm_world()) == 0:
                print "ITERATE"

        iter += 1

        if linear_flag == 0:
            if MPI.rank(mpi_comm_world()) == 0:
                print "iteration = ", iter
                print ""

        ### Solve ###
        X = LinearSolver(U,V,X_n,t,bcs)

        if MPI.rank(mpi_comm_world()) == 0:
            print ""

        if linear_flag == 0:
            if MPI.rank(mpi_comm_world()) == 0:
                print "Solved for a new solution estimate."
                print ""

        p, q, us = X.split(True)
        p_m, q_m, us_m = X_m.split(True)
        p_n, q_n, us_n = X_n.split(True)

        # Evaluate for convergence of pressure solution on each processor
        if linear_flag == 1:
            rel_error_max_local = 0.0
        if linear_flag == 0:
            if MPI.rank(mpi_comm_world()) == 0:
                print "Evaluate for Convergence"
                print "-------------"
            cell_values_p = p.vector().get_local()
            cell_values_p_m = p_m.vector().get_local()
            rel_error_max_local = np.nanmax(np.divide(np.abs(cell_values_p \
                - cell_values_p_m),np.abs(cell_values_p_m)))

        # Find the global maximum value of the relative error
        rel_error_max_global = MPI.max(mpi_comm_world(),rel_error_max_local)

        if MPI.rank(mpi_comm_world()) == 0:
            print "Relative Error = ",rel_error_max_global
            print "-------------"
            print ""

        # Update estimate
        X_new = X_m + omega*(X - X_m)
        X_m.assign(X_new)

        if iter == max_iter:
            if MPI.rank(mpi_comm_world()) == 0:
                print "Maximum iterations met"
                print "Solution doesn't converge"
            quit()

    if linear_flag == 0:
        if MPI.rank(mpi_comm_world()) == 0:
            print "The solution has converged."
            print "Total iterations = ", iter
            print ""
    
    if MPI.rank(mpi_comm_world()) == 0:
        print "Saving solutions."
        print ""
    pressure_file.write(p,t)
    flux_file.write(q,t)
    disp_file.write(us,t)
    density_save.assign(project(rho_f(p,p_0,p_h,rho_0,beta_f),P))
    porosity_save.assign(project(phi(alpha,us,p,beta_s,phi_0,phi_min,t),P))
    density_file.write(density_save,t)
    porosity_file.write(porosity_save,t)

    # Update solution at last time step
    X_n.assign(X)
    phi_n.assign(project(phi(alpha,us,p,beta_s,phi_0,phi_min,t),P))
    if MPI.rank(mpi_comm_world()) == 0:
        print "Just updated last solution."
        print ""
        print "###############"
        print ""

###########################################################################

if MPI.rank(mpi_comm_world()) == 0:
    print "This code finished at"
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
