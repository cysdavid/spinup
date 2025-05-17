import numpy as np
from mpi4py import MPI
import time
import json

from dedalus import public as d3

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.parallel import Sync
import pathlib
import inspect

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

######### PARAMETERS ###############################################################

# Simulation name
sim_name = 'sim25'

# Numerical Parameters
ns, nz = (1024,1024)
dealias = 3/2
dtype = np.float64
timestepper = d3.RK222

# Physical parameters
Ek = 1.36e-4 # Ekman number, Ek = nu/(Omega*H**2)
PeakOmega = -0.75 # Maximum (absolute) change in rotation rate
Lz = 1 # height of cylinder
Ls = 9/4 # radius of cylinder
w = 0.05  # thickness of top and bottom "lids"
eta = 1e-2 # Volume penalty damping timescale (enforces no-slip at top and bottom), 
            # eta * sqrt(Ek) = (dimensional damping timescale)/(H/sqrt(nu Omega_0))
            # eta * sqrt(Ek) should be the fastest timescale in the system (e.g., faster than timescale over which Omega changes)
perturbation_amp = 1e-4 # RMSE amplitude of us, uz noise
free_surface = True # whether to impose a stress-free, no-penetration condition at z=0

# Boundary forcing function, i.e., how the tank rotation rate should vary with time, t
# Examples:

## Spin-down:
# full_DelOmega_func = lambda t,PeakOmega : PeakOmega * (0.5*(1 + np.tanh((2*(-1e-2 + t))/5e-3)))

## Spin-down then spin-up:
# full_DelOmega_func = lambda t,PeakOmega : PeakOmega * (0.5*(np.tanh((2*(-1e-2 + t))/5e-3) + np.tanh(-((2*(t - 0.13))/(5e-3)))))

## Spin-up then spin-down:
full_DelOmega_func = lambda t,PeakOmega : PeakOmega * (1 - 0.5*(np.tanh((2*(-1e-2 + t))/5e-3) + np.tanh(-((2*(t - 0.13))/(5e-3)))))

## Write your own function:
# full_DelOmega_func = lambda t,PeakOmega : <your function of t>

# Cadences and stop time
timestep = 5e-6 #3.2e-6 #5e-6#5e-6 #2.5e-6#5e-6 #1e-5
output_cadence = 10
stop_sim_time = 0.3
snapshot_dt = stop_sim_time/1000

######### SIMULATION CODE ##########################################################

# Create bases and domain
coords = d3.CartesianCoordinates('z', 's') # Fourier direction comes first, for parallelization
dist = d3.Distributor(coords, dtype=np.float64)
sbasis = d3.Chebyshev(coords['s'], ns, bounds=(0, Ls), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], nz, bounds=(-Lz/2-w, Lz/2+w), dealias=3/2)
zgrid, sgrid = dist.local_grids(zbasis, sbasis)

# Fields
s = dist.Field(name='s', bases=sbasis)
s['g'] = sgrid
z = dist.Field(name='z', bases=zbasis)
z['g'] = zgrid
p = dist.Field(name='p', bases=(zbasis,sbasis))
us = dist.Field(name='us', bases=(zbasis,sbasis))
uphi = dist.Field(name='uphi', bases=(zbasis,sbasis))
uz = dist.Field(name='uz', bases=(zbasis,sbasis))
t = dist.Field()

# Boundary forcing
DelOmega_func = lambda t: full_DelOmega_func(t,PeakOmega)
DelOmega = DelOmega_func(t)

# Volume penalty terms
mask = lambda x: 0.5*(1+np.tanh(-2*x))
eps = (eta*Ek)**(1/2)
delta = 2.64822828*eps # sharpness of mask for obstacle
Kwall = dist.Field(name='Kwall', bases=(zbasis))
Kwall['g'] = mask(-(zgrid - (Lz/2))/delta) + mask((zgrid - (-Lz/2))/delta)

# Substitutions
ds = lambda A: d3.Differentiate(A, coords['s'])
dz = lambda A: d3.Differentiate(A, coords['z'])
if free_surface:
    Klowerhalf = dist.Field(name='Klowerhalf', bases=(zbasis))
    Klowerhalf['g'] = mask((zgrid)/delta)
    tank_integ = lambda A: 2*np.pi*d3.Integrate(Klowerhalf*s*A, ('z','s'))
    Vol = np.pi*Ls**2*(Lz/2+w)
    integ_z = lambda A: d3.Integrate(Klowerhalf*A, ('z'))
else:
    tank_integ = lambda A: 2*np.pi*d3.Integrate(s*A, ('z','s'))
    Vol = np.pi*Ls**2*(Lz+2*w)
    integ_z = lambda A: d3.Integrate(A, ('z'))
tank_avg = lambda A: 1/Vol * tank_integ(A)

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

# Initial condition: incompressible noise in us, uz
uphi['g'] = np.outer(np.ones(nz),DelOmega_func(0)*sgrid)
psi = dist.Field(name='psi', bases=(zbasis,sbasis))
psi.fill_random('g', seed=42, distribution='normal')
psi.low_pass_filter(scales=0.25)
psi['g'] = (psi * s/Ls * np.tanh(-(s - Ls)/0.1) * (1-(Kwall)))['g']
us.change_scales(dealias)
uz.change_scales(dealias)
us['g'] = (dz(psi))['g']
uz['g'] = (-psi/s- ds(psi))['g']
ubot_rmse = np.sqrt(tank_avg(us**2 + uz**2))
us['g'] = (us * 1/ubot_rmse * perturbation_amp)['g']
uz['g'] = (uz * 1/ubot_rmse * perturbation_amp)['g']

# Enforce stress-free, no-penetration condition spectrally
if free_surface:
    us['c'][1::2] = 0 # Set sines to 0 --> cosines only
    uphi['c'][1::2] = 0 # Set sines to 0 --> cosines only
    p['c'][1::2] = 0 # Set sines to 0 --> cosines only
    tau_us2['c'][1::2] = 0 # Set sines to 0 --> cosines only
    tau_uphi2['c'][1::2] = 0 # Set sines to 0 --> cosines only
    tau_p['c'][1::2] = 0 # Set sines to 0 --> cosines only
    Kwall['c'][1::2] = 0 # Set sines to 0 --> cosines only

    uz['c'][::2] = 0 # Set cosines to 0 --> sines only
    tau_uz2['c'][::2] = 0 # Set cosines to 0 --> sines only

# Problem
problem = d3.IVP([p, us, uphi, uz, tau_p, tau_us2, tau_uphi2, tau_uz2], time=t, namespace=locals())
## Equations
problem.add_equation("-2*s**2*uphi + Ek*us + s**2*ds(p) - Ek*s*ds_us - Ek*s**2*ds(ds_us) + np.sqrt(Ek)*s**2*dt(us) - Ek*s**2*dz(dz(us)) + s**2*lift(tau_us2) = s*uphi**2 - s**2*us*ds_us - s**2*uz*dz(us) - s**2*((Kwall)*us)/eta")
problem.add_equation("Ek*uphi + 2*s**2*us - Ek*s*ds_uphi - Ek*s**2*ds(ds_uphi) + np.sqrt(Ek)*s**2*dt(uphi) - Ek*s**2*dz(dz(uphi)) + s**2*lift(tau_uphi2) = -(s*uphi*us) - s**2*us*ds_uphi - s**2*uz*dz(uphi) - s**2*((Kwall)*(uphi-DelOmega*s))/eta")
problem.add_equation("-(Ek*ds_uz) - Ek*s*ds(ds_uz) + np.sqrt(Ek)*s*dt(uz) + s*dz(p) - Ek*s*dz(dz(uz)) + s*lift(tau_uz2) = -(s*us*ds_uz) - s*uz*dz(uz) - s*((Kwall)*uz)/eta")
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
save_data_path = data_path.joinpath(sim_name)
static_fields_path = save_data_path.joinpath('static_fields')
save_params_path = params_path.joinpath(sim_name+".json")
DelOmega_path = save_data_path.joinpath('DelOmega.py')
if rank == 0:
    if not data_path.exists():
        data_path.mkdir()
    if not params_path.exists():
        params_path.mkdir()
    if not save_data_path.exists():
        save_data_path.mkdir()

    # Export function for change in rotation rate
    DelOmega_path.touch(exist_ok=True)
    with open(DelOmega_path, "w") as file:
        file.write("import numpy as np\n")
        file.write(inspect.getsource(full_DelOmega_func))

    # Save parameters
    params_dict = {'ns':ns,'nz':nz,'timestep':timestep,
                   'Ek':Ek,'PeakOmega':PeakOmega,'Lz':Lz,'Ls':Ls,
                   'w':w,'eta':eta,'perturbation_amp':perturbation_amp,'free_surface':free_surface}
    params_json = json.dumps(params_dict, indent = 4)
    with open(save_params_path, "w") as outfile: 
        outfile.write(params_json)

# Analysis
snapshots = solver.evaluator.add_file_handler(str(save_data_path), sim_dt=snapshot_dt, max_writes=1000)
snapshots.add_task(us)
snapshots.add_task(uphi)
snapshots.add_task(uz)
snapshots.add_task(p)
snapshots.add_task(DelOmega,name='DelOmega')
snapshots.add_task(dz(us)-ds(uz),name='vort_phi')
snapshots.add_task(tank_avg(uphi),name='avg_uphi')
snapshots.add_task(0.5*tank_avg(uphi**2),name='E_phi')
snapshots.add_task(0.5*tank_avg(us**2 + uz**2),name='E_sz')
snapshots.add_task(ds_us,name='ds_us')
snapshots.add_task(dz(uz),name='dz_uz')

static_fields = solver.evaluator.add_file_handler(str(static_fields_path), sim_dt = stop_sim_time)
static_fields.add_task(Kwall)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property(np.abs(uphi), name='abs_uphi')
flow.add_property(np.abs(uz), name='abs_uz')
flow.add_property(np.abs(p), name='abs_p')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        solver.step(timestep)
        if free_surface:
            us['c'][1::2] = 0 # Set sines to 0 --> cosines only
            uphi['c'][1::2] = 0 # Set sines to 0 --> cosines only
            p['c'][1::2] = 0 # Set sines to 0 --> cosines only
            tau_us2['c'][1::2] = 0 # Set sines to 0 --> cosines only
            tau_uphi2['c'][1::2] = 0 # Set sines to 0 --> cosines only
            tau_p['c'][1::2] = 0 # Set sines to 0 --> cosines only

            uz['c'][::2] = 0 # Set cosines to 0 --> sines only
            tau_uz2['c'][::2] = 0 # Set cosines to 0 --> sines only

        if (solver.iteration-1) % output_cadence == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, timestep))
            logger.info('Max |uphi| = {}, Max |uz| = {}, Max |p| = {}'.format(flow.max('abs_uphi'), flow.max('abs_uz'), flow.max('abs_p')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
