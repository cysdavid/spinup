import numpy as np
from mpi4py import MPI
import time
import json

from dedalus import public as d3

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.parallel import Sync
import pathlib

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Simulation name
sim_name = 'sim1'

# Numerical Parameters
ns, nz = (256,512) #(128,256)
dealias = 3/2
dtype = np.float64
timestepper = d3.RK443 #d3.RK222

# Physical parameters
Ek = 0.1/4 #0.1
PeakOmega = 0.1
Lz = 1
Ls = 0.5
w = 0.1 # tank wall thickness
eta = 3e-4/4  # Volume penalty damping timescale

# Cadences and stop time
timestep = 1e-5#1e-4
output_cadence = 10
stop_sim_time = 0.1 #0.1
snapshot_dt = stop_sim_time/1000

# Create bases and domain
coords = d3.CartesianCoordinates('s', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
sbasis = d3.Chebyshev(coords['s'], ns, bounds=(0, Ls), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], nz, bounds=(-Lz/2-w, Lz/2+w), dealias=3/2)
sgrid, zgrid = dist.local_grids(sbasis, zbasis)
# ss = np.outer(sgrid,np.ones(zgrid.shape[-1]))
# zz = np.outer(np.ones(sgrid.shape[0]),zgrid)

# Fields
s = dist.Field(name='s', bases=sbasis)
s['g'] = sgrid
z = dist.Field(name='z', bases=zbasis)
z['g'] = zgrid
p = dist.Field(name='p', bases=(sbasis,zbasis))
us = dist.Field(name='us', bases=(sbasis,zbasis))
uphi = dist.Field(name='uphi', bases=(sbasis,zbasis))
uz = dist.Field(name='uz', bases=(sbasis,zbasis))
t = dist.Field()

# Boundary forcing
# Examples

## Spin-down/up:
# delay = 1e-2
# DelOmega_func = lambda tt : PeakOmega * (0.5*(1 + np.tanh((2*(-2*delay + tt))/delay)))

## Spin-down then spin-up:
DelOmega_func = lambda tt : PeakOmega * -1/np.cosh((tt - 0.05)/0.005)

DelOmega = DelOmega_func(t)

# Substitutions
ds = lambda A: d3.Differentiate(A, coords['s'])
dz = lambda A: d3.Differentiate(A, coords['z'])
tank_integ = lambda A: 2*np.pi*d3.Integrate(s*A, ('s','z'))
Vol = np.pi*Ls**2*(Lz+2*w)
tank_avg = lambda A: 1/Vol * tank_integ(A)
integ_z = lambda A: d3.Integrate(A, ('z'))

# Tau terms p, us2, uphi2, uz2
tau_p = dist.Field(name='tau_p', bases=zbasis)
tau_us2 = dist.Field(name='tau_us2', bases=zbasis)
tau_uphi2 = dist.Field(name='tau_uphi2', bases=zbasis)
tau_uz2 = dist.Field(name='tau_uz2', bases=zbasis)
lift_basis = sbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
ds_us = ds(us)
ds_uphi = ds(uphi)
ds_uz = ds(uz)

# Volume penalty terms
mask = lambda x: 0.5*(1+np.tanh(-2*x))
eps = (eta*Ek)**(1/2)
delta = 2.64822828*eps # sharpness of mask for obstacle
Kceil = dist.Field(name='Kceil', bases=(zbasis))
Kfloor = dist.Field(name='Kfloor', bases=(zbasis))
Kceil['g'] = mask(-(zgrid - (Lz/2))/delta)
Kfloor['g'] = mask((zgrid - (-Lz/2))/delta)


# Problem
problem = d3.IVP([p, us, uphi, uz, tau_p, tau_us2, tau_uphi2, tau_uz2], time=t, namespace=locals())
## Equations
problem.add_equation("-2*s**2*uphi + Ek*us + s**2*ds(p) - Ek*s*ds_us - Ek*s**2*ds(ds_us) + np.sqrt(Ek)*s**2*dt(us) - Ek*s**2*dz(dz(us)) + s**2*lift(tau_us2) = s*uphi**2 - s**2*us*ds_us - s**2*uz*dz(us) - s**2*((Kceil+Kfloor)*us)/eta")
problem.add_equation("Ek*uphi + 2*s**2*us - Ek*s*ds_uphi - Ek*s**2*ds(ds_uphi) + np.sqrt(Ek)*s**2*dt(uphi) - Ek*s**2*dz(dz(uphi)) + s**2*lift(tau_uphi2) = -(s*uphi*us) - s**2*us*ds_uphi - s**2*uz*dz(uphi) - s**2*((Kceil+Kfloor)*(uphi-DelOmega*s))/eta")
problem.add_equation("-(Ek*ds_uz) - Ek*s*ds(ds_uz) + np.sqrt(Ek)*s*dt(uz) + s*dz(p) - Ek*s*dz(dz(uz)) + s*lift(tau_uz2) = -(s*us*ds_uz) - s*uz*dz(uz) - s*((Kceil+Kfloor)*uz)/eta")
problem.add_equation("us + s*ds_us + s*dz(uz) + tau_p = 0")
## BCs
problem.add_equation("us(s=Ls) = 0")
problem.add_equation("uphi(s=Ls) = DelOmega*Ls")
problem.add_equation("uz(s=Ls) = 0")
problem.add_equation("p(s=0) = 0", condition="nz==0") # Sets pressure gauge
problem.add_equation("ds(us)(s=Ls) = 0", condition="nz!=0") # Consequence of continuity eqn at the wall

# Build solver
solver = problem.build_solver(timestepper)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Create data and params directories if needed
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
with Sync() as sync:
    if sync.comm.rank == 0:
        if not data_path.exists():
            data_path.mkdir()
        if not params_path.exists():
            params_path.mkdir()
save_data_path = data_path.joinpath(sim_name)
static_fields_path = save_data_path.joinpath('static_fields')
save_params_path = params_path.joinpath(sim_name+".json")

# Save parameters
# Ek, D, Lz, Ls, w, eta, switch_s, switch_phi, switch_z
if rank == 0:
    params_dict = {'ns':ns,'nz':nz,'timestep':timestep,
                   'Ek':Ek,'PeakOmega':PeakOmega,'Lz':Lz,'Ls':Ls,
                   'w':w,'eta':eta}
    params_json = json.dumps(params_dict, indent = 4)
    with open(save_params_path, "w") as outfile: 
        outfile.write(params_json)

# Analysis
snapshots = solver.evaluator.add_file_handler(str(save_data_path), sim_dt=snapshot_dt, max_writes=1000)
# snapshots.add_tasks(solver.state)
snapshots.add_task(us)
snapshots.add_task(uphi)
snapshots.add_task(uz)
snapshots.add_task(p)
snapshots.add_task(DelOmega,name='DelOmega')
snapshots.add_task(dz(us)-ds(uz),name='vort_phi')
snapshots.add_task(tank_avg(uphi),name='avg_uphi')
snapshots.add_task(0.5*tank_avg(uphi**2),name='E_phi')
snapshots.add_task(0.5*tank_avg(us**2 + uz**2),name='E_sz')

static_fields = solver.evaluator.add_file_handler(str(static_fields_path), sim_dt = stop_sim_time)
static_fields.add_task(Kceil)
static_fields.add_task(Kfloor)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property(np.abs(uphi), name='abs_uphi')
flow.add_property(np.abs(uz), name='abs_uz')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % output_cadence == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, timestep))
            logger.info('Max |uphi| = {}, Max |uz| = {}'.format(flow.max('abs_uphi'), flow.max('abs_uz')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
