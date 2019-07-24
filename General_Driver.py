"""

This is the driver code for the time and Picard loop of the fully-coupled
poroelastic formulation. The driver calls the solver in parallel, which
reads in all necessary information from the driver and solves the three
PDEs (conservation of mass, darcy's law, conservation of momentum) using 
a Mixed Finite Element approach (i.e. a monolithic solve). The linear model 
uses a constant fluid density and porosity. The nonlinear model uses a 
d(phi)/dt porosity model to update porosity in each time step, which is 
derived from solid continuity assuming small strains. Fluid density is also
allowed to vary.

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
import subprocess
import os
import shutil
import ufl

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

if not os.path.isdir("./Exchange_Files"):
	os.mkdir('Exchange_Files')

from time import gmtime, strftime
print "Start date and time:"
print strftime("%Y-%m-%d %H:%M:%S", gmtime())

###########################################################################

######### DEFINE MODEL TYPE ###############################################

Linear = 1
Nonlinear = 0

if Linear + Nonlinear != 1:
    print "You must choose a model type."
    quit()

if Linear == 1:
    print "Running the Linear Poroelastic Model."
    weight = Constant(0.0)
    linear_flag = 1

if Nonlinear == 1:
    print "Running the Nonlinear Poroelastic Model."
    weight = Constant(1.0)
    linear_flag = 0

model_type = [Linear,Nonlinear]
np.savetxt('Exchange_Files/model_type.txt',model_type)

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

Domain_params = [x0,x1,y0,y1,z0,z1,nx,ny,nz]
np.savetxt('Exchange_Files/Domain_params.txt',Domain_params)

###########################################################################

######### MESHING #########################################################

print ('Building mesh...')
mesh = BoxMesh(Point(x0,y0,z0),Point(x1,y1,z1),nx,ny,nz)
File('Exchange_Files/Problem_mesh.xml') << mesh

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

### Print the number of degrees of freedom ###
print "Number of DOFS = ", W.dofmap().global_dimension()

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

Params = [g,k,mu,phi_0,phi_min,p_0,rho_0,beta_m,beta_f,beta_s,nu,K,G,alpha]
np.savetxt('Exchange_Files/Params.txt',Params)

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
nsteps = 1
dt = tend/nsteps

Time_params = [tend,nsteps,dt]
np.savetxt('Exchange_Files/Time_params.txt',Time_params)

###########################################################################

######### INITIAL CONDITIONS  #############################################

### Initial Condition ###
X_i = Expression(
        (
            "0.0",       		# p    
            "0.0","0.0","0.0", 	# (q1, q2)
            "0.0","0.0","0.0"  	# (us1, us2)
        ),degree = 2)
X_n = interpolate(X_i,W)

# Initial porosity
phi_n = interpolate(Constant(phi_0),Z)

# Initial Picard solution estimate
X_m = interpolate(X_i,W)

file_X_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_n.h5','w')
file_X_m = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_m.h5','w')
file_phi_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/phi_n.h5','w')
file_X_n.write(X_n,'X_n')
file_X_m.write(X_m,'X_m')
file_phi_n.write(phi_n,'phi_n')
del file_X_n
del file_X_m
del file_phi_n

###########################################################################

######### SOLVER SET-UP  ##################################################

X = Function(W)

density_save = Function(Z)
porosity_save = Function(Z)

num_proc = 1 	# Number of processers

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

print ('Starting Time Loop...')

### Time Loop ###
for n in range(nsteps):

    t += dt

    print "###############"
    print ""
    print "NEW TIME STEP"
    print "Time =", t
    print ""
    print np.round(t/tend*100.0,6),"% Complete"

    np.savetxt('Exchange_Files/t.txt',[t])

    ### Convergence criteria ###
    reltol = 1E-4			# Relative error tolerance for Picard
    rel_error_max = 9999	# Initialize relative error 
    max_iter = 100			# Maximum Picard iterations
    omega = 1.0				# Relaxation coefficient

    iter = 0

    if linear_flag == 0:
        print "Entering Picard Iteration:"
        print ""

    ### Picard Iteration Loop ###
    while (rel_error_max > reltol):

        if linear_flag == 0:
            print "ITERATE"

        iter += 1

        if linear_flag == 0:
            print "iteration = ", iter
            print ""

        # Call solver script
        solver_finish = 0
        np.savetxt('Exchange_Files/solver_finish.txt',[solver_finish])
        subprocess.call(['mpirun','-n',str(num_proc),'python',\
            'General_Solver.py'])
        solver_finish = np.loadtxt('Exchange_Files/solver_finish.txt')
        if solver_finish == 0:
            print 'Solver script could not finish.'
            quit()

        print ""
        if linear_flag == 0:
            print "Solved for a new solution estimate."
            print ""

        file_X = HDF5File(mesh.mpi_comm(),'Exchange_Files/X.h5','r')
        file_X.read(X,'X')
        del file_X

        p, q, us = X.split(True)
        p_m, q_m, us_m = X_m.split(True)
        p_n, q_n, us_n = X_n.split(True)

        # Evaluate for convergence of pressure solution
        if linear_flag == 1:
            rel_error_max = 0.0
        if linear_flag == 0:
            print "Evaluate for Convergence"
            print "-------------"
            vertex_values_p = p.compute_vertex_values(mesh)
            vertex_values_p_m = p_m.compute_vertex_values(mesh)
            rel_error_max = np.nanmax(np.divide(np.abs(vertex_values_p \
                - vertex_values_p_m),np.abs(vertex_values_p_m)))
            print "Relative Error = ",rel_error_max
            print "-------------"
            print ""

        # Update estimate
        X_new = X_m + omega*(X - X_m)
        X_m.assign(X_new)
        file_X_m = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_m.h5','w')
        file_X_m.write(X_m,'X_m')
        del file_X_m

        if iter == max_iter:
            print "Maximum iterations met"
            print "Solution doesn't converge"
            quit()

    if linear_flag == 0:
        print "The solution has converged."
        print "Total iterations = ", iter
        print ""
    
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
    print "Just updated last solution."
    print ""
    print "###############"
    print ""

    file_X_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/X_n.h5','w')
    file_phi_n = HDF5File(mesh.mpi_comm(),'Exchange_Files/phi_n.h5','w')
    file_X_n.write(X_n,'X_n')
    file_phi_n.write(phi_n,'phi_n')
    del file_X_n
    del file_phi_n

###########################################################################

shutil.rmtree('Exchange_Files')

print "This code finished at"
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
