import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import pathlib
import json
import glob
import sys
import pickle
from dedalus.tools.general import natural_sort
import logging
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

######### PARAMETERS ###############################################################

# File parameters
evp_name = 'sim9_evp2' # name of axismmetric IVP simulation from which to take base flow
bivp_basename = 'bivp3' # Change this if you change any of the parameters below
ivp_write_num = 645#525

# Physical parameters
Ek_override = None # If not None, will override the value of Ek from the IVP
perturbation_amp = 1e-4
tracer = False
Sc = 1
noise = True
fastest_growing_mode = False

# Numerical parameters
nphi = 512 # Max azimuthal order
ns = 256 # Number of radial grid points
dealias = 3/2
dtype = np.float64
timestepper = d3.RK443

# Cadences and stop time
max_timestep = 2.5e-6
output_cadence = 10
stop_sim_time = 1.2 #"end" # If "end", uses the last timestamp in the EVP as the stopping time
snapshot_dt_over_stop_sim_time = 1/3 * 1/1000

######### SIMULATION CODE ####################################################

# Set up paths and make directories
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
evp_subfiles_path = data_path.joinpath(evp_name)
evp_params_path = params_path.joinpath(evp_name+".json")
ivp_params_path = params_path.joinpath(evp_name.split("_")[0]+".json")

# Load parameters
with open(evp_params_path) as f: 
    params_evp = json.load(f)
with open(ivp_params_path) as f: 
    params_ivp = json.load(f)

# Parameters
dtype = np.float64
R = params_evp['R']
if Ek_override == None:
    Ek = params_evp['Ek']
else:
    Ek = Ek_override

# Get data from EVP subfile to use as initial condition
evp_subfiles = natural_sort(glob.glob(str(evp_subfiles_path.joinpath("*"))))
evp_subfile_idx = [idx for idx,path in enumerate(evp_subfiles) if int(pathlib.Path(path).stem) == ivp_write_num][0]
evp_subfile = evp_subfiles[evp_subfile_idx]
with open(evp_subfile, 'rb') as handle:
    evp_dict = pickle.load(handle)
m_list = evp_dict['m']
sigma_list = evp_dict['sigma']
u_list = evp_dict['u']
p_list = evp_dict['p']

# Get slowest decaying mode
m_slowest_decaying = m_list[m_list>0][np.argmax(sigma_list[:,0].real[m_list>0])]
sigma_slowest_decaying = sigma_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0]
u_pert = u_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0].real
p_pert = p_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0].real

# Get rotation rate
sys.path.append(str(data_path.joinpath(evp_name.split("_")[0])))
import DelOmega
PeakOmega = params_ivp['PeakOmega']
Omega_func = lambda t: 1 + DelOmega.full_DelOmega_func(t,PeakOmega)

# Get times from EVP
t_arr = np.zeros(len(evp_subfiles[evp_subfile_idx:]))
for i,path in enumerate(evp_subfiles[evp_subfile_idx:]):
    with open(path, 'rb') as handle:
        each_evp_dict = pickle.load(handle)
        t_arr[i] = each_evp_dict['t']

# Set stop time and snapshot dt
if stop_sim_time == "end":
    stop_sim_time = t_arr[-1]
snapshot_dt = snapshot_dt_over_stop_sim_time * stop_sim_time

# Make problem
## Bases
coords = d3.PolarCoordinates('phi', 's')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(nphi, ns), radius=R, dtype=dtype, dealias=3/2)
phi_grid, s_grid = dist.local_grids(disk)

## Fields
u = dist.VectorField(coords, name='u', bases=disk)
p = dist.Field(name='p', bases=disk)
s = dist.Field(name='s', bases=disk.radial_basis)
s['g'] = s_grid
ses = dist.VectorField(coords, name='ses', bases=disk.radial_basis)
ses['g'][1] = s_grid
tau_u = dist.VectorField(coords, name='tau_u', bases=disk.edge)
tau_p = dist.Field(name='tau_p')
t = dist.Field()
t['g'] = evp_dict['t']

if tracer:
    c = dist.Field(name='c', bases=disk)
    tau_c = dist.Field(name='tau_c', bases=disk.edge)
    ds = lambda A: 1/s*(d3.grad(A))@ses

    ## Set tracer field initial condition
    c.change_scales((u_pert.shape[1]/nphi,params_evp['ns']/ns))
    c['g'] += evp_dict['uphi0'][rank*params_evp['ns']//size:(rank+1)*params_evp['ns']//size]
    c.change_scales(1)
    c['g'] += -s_grid
    c['g'] = (c['g'] - np.min(c['g']))/(np.max(c['g']) - np.min(c['g']))

## Substitutions
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
avg = lambda A: 1/(np.pi*R**2)*d3.Integrate(A)

## Initial condition
### Add random noise
if noise:
    u.fill_random('g', seed=42, distribution='standard_normal') # Random noise
    u.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    u['g'] *= perturbation_amp
### Add background azimuthal flow
u.change_scales((u_pert.shape[1]/nphi,params_evp['ns']/ns))
p.change_scales((u_pert.shape[1]/nphi,params_evp['ns']/ns))
u['g'][0] += evp_dict['uphi0'][rank*params_evp['ns']//size:(rank+1)*params_evp['ns']//size]
### Add first fastest growing mode from EVP
if fastest_growing_mode:
    u['g'][0] = evp_dict['uphi0'][rank*params_evp['ns']//size:(rank+1)*params_evp['ns']//size]
    u['g'] += perturbation_amp * u_pert[:,:,rank*params_evp['ns']//size:(rank+1)*params_evp['ns']//size]
    p['g'] = perturbation_amp * p_pert[:,rank*params_evp['ns']//size:(rank+1)*params_evp['ns']//size]

u.change_scales(1)
p.change_scales(1)

# Boundary forcing
Omega = Omega_func(t)

## Problem
if tracer:
    problem = d3.IVP([u, p, c, tau_u, tau_p, tau_c], time=t, namespace=locals())
    problem.add_equation("np.sqrt(Ek)*dt(c) - Ek/Sc*lap(c) + lift(tau_c) = - u@grad(c)")
    problem.add_equation("ds(c)(s=R) = 0")
else:
    problem = d3.IVP([u, p, tau_u, tau_p], time=t, namespace=locals())

problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("np.sqrt(Ek)*dt(u) + grad(p) - Ek*lap(u) + lift(tau_u) = - u@grad(u)")
problem.add_equation("integ(p) = 0")

# non-axisymmetric modes
problem.add_equation("radial( u(s=R) ) = 0", condition='nphi!=0')
problem.add_equation("azimuthal( u(s=R) ) = 0", condition='nphi!=0')

# axisymmetric modes
problem.add_equation("radial( u(s=R) ) = 0", condition='nphi==0')
problem.equations[-1]['valid_modes'][1] = True
problem.add_equation("azimuthal( u(s=R) ) = Omega*R", condition='nphi==0')
problem.equations[-1]['valid_modes'][1] = True

# Build solver
solver = problem.build_solver(timestepper)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Create data and params directories if needed
sim_name = evp_name + "_" + bivp_basename
save_data_path = data_path.joinpath(sim_name)
save_params_path = params_path.joinpath(sim_name+".json")

if rank == 0:
    # Save parameters
    params_dict = {'ns':ns,'nphi':nphi,'max_timestep':max_timestep,
                   'Ek':Ek,'PeakOmega':PeakOmega,'Ls':R, 'R':R,
                   'perturbation_amp':perturbation_amp, 'tracer':tracer, 'Sc':Sc,
                   'noise':noise, 'fastest_growing_mode':fastest_growing_mode,
                   'ivp_write_num':ivp_write_num,'stop_sim_time':stop_sim_time}
    params_json = json.dumps(params_dict, indent = 4)
    with open(save_params_path, "w") as outfile: 
        outfile.write(params_json)

# Analysis
snapshots = solver.evaluator.add_file_handler(str(save_data_path), sim_dt=snapshot_dt, max_writes=1000)
snapshots.add_task(u)
snapshots.add_task(p)
if tracer:
    snapshots.add_task(c)
snapshots.add_task(Omega,name='Omega')
snapshots.add_task(-d3.div(d3.skew(u)),name='vort')
snapshots.add_task(0.5*avg(u@u),name='KE')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(u@u, name='u2')
flow.add_property(np.abs(u@ses), name='abs(us*s)')
flow.add_property(np.abs(p), name='abs_p')
if tracer:
    flow.add_property(np.abs(c), name='abs_c')
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % output_cadence == 0:
            max_u = np.sqrt(flow.max('u2'))
            max_p = flow.max('abs_p')
            max_us_s = flow.max('abs(us*s)')
            if tracer:
                max_c = flow.max('abs_c')
                logger.info("Iteration=%i, Time=%e, dt=%e, max|u^2|=%e, max|u_s*s|=%e, max|p|=%e, max|c|=%e" %(solver.iteration, solver.sim_time, timestep, max_u, max_us_s, max_p, max_c))
            else:
                logger.info("Iteration=%i, Time=%e, dt=%e, max|u^2|=%e, max|u_s*s|=%e, max|p|=%e" %(solver.iteration, solver.sim_time, timestep, max_u, max_us_s, max_p))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()