import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import pathlib
import json
import glob
import h5py
import pickle
from dedalus.tools.general import natural_sort
import logging
logger = logging.getLogger(__name__)

from scipy import interpolate

######### PARAMETERS ###############################################################

sim_name = 'sim9' # name of axismmetric IVP simulation from which to take base flow
evp_basename = 'evp3' # Change this if you change any of the parameters below
strd = 4 # How often to import snapshots of uphi0 from the axisymmetric IVP (e.g. strd=2 means every other snapshots)
Ek_override = 5e-5 # If not None, will override the value of Ek from the IVP
M = 8 # Max azimuthal order
ns = 128 # Number of radial grid points

######### EIGENVALUE PROBLEM CODE #####################################################

# Set up paths and make directories
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
ivp_data_path = data_path.joinpath(sim_name)
static_fields_path = ivp_data_path.joinpath('static_fields')
ivp_params_path = params_path.joinpath(sim_name+".json")
data_files = natural_sort(glob.glob(str(ivp_data_path.joinpath('*.h5'))))
static_fields_files = natural_sort(glob.glob(str(static_fields_path.joinpath('*.h5'))))
save_params_path = params_path.joinpath(sim_name+"_"+evp_basename+".json")
subfolders_path = data_path.joinpath(sim_name+"_"+evp_basename)

if not subfolders_path.exists():
    subfolders_path.mkdir()

# Load parameters
with open(ivp_params_path) as f: 
    params_ivp = json.load(f)

# Computed parameters
if Ek_override == None:
    Ek = params_ivp['Ek']
else:
    Ek = Ek_override
R = params_ivp['Ls']
nphi = 2 * M + 2
dtype = np.complex128
rank = 0
size = 1

# Save parameters
params_evp = {'Ek':Ek, 'R':R, 'M':M, 'nphi':nphi, 'ns':ns}
params_json = json.dumps(params_evp, indent = 4)
with open(save_params_path, "w") as outfile: 
    outfile.write(params_json)

# Main loop
for filename in data_files:
    with h5py.File(filename, mode='r') as f:
        num_snapshots = f['tasks']['uphi'].shape[0]
    for it in range(strd*rank,num_snapshots,strd*size):
        # Load base flow
        print(f"Loading snapshot {it+1}/{num_snapshots}...")
        with h5py.File(filename, mode='r') as f:
            s_data = f['tasks']['uphi'].dims[2][0][:]
            z_data = f['tasks']['uphi'].dims[1][0][:]
            t_data = f['tasks']['E_phi'].dims[0]['sim_time'][it]
            write_num = f['tasks']['uphi'].dims[0]['write_number'][it]
            DelOmega_data = f['tasks']['DelOmega'][it][0][0]
            uphi_data = f['tasks']['uphi'][it]

        uphi0_data = uphi_data[np.argmin(np.abs(z_data)),:] + (1)*s_data # Shift from Omega_0 frame to inertial frame
        uphi0_func = interpolate.interp1d(s_data,uphi0_data,kind='cubic')

        # Make problem
        ## Bases
        coords = d3.PolarCoordinates('phi', 's')
        dist = d3.Distributor(coords, dtype=dtype)
        disk = d3.DiskBasis(coords, shape=(nphi, ns), radius=R, dtype=dtype)
        phi_grid, s_grid = dist.local_grids(disk)

        ## Fields
        sigma = dist.Field(name='sigma')
        u = dist.VectorField(coords, name='u', bases=disk)
        p = dist.Field(name='p', bases=disk)
        s = dist.Field(name='s', bases=disk.radial_basis)
        s['g'] = s_grid
        tau_u = dist.VectorField(coords, name='tau_u', bases=disk.edge)
        tau_p = dist.Field(name='tau_p')

        ## Substitutions
        dt = lambda A: sigma*A
        lift_basis = disk.derivative_basis(2)
        lift = lambda A: d3.Lift(A, lift_basis, -1)

        ## Background
        Uphi0 = dist.VectorField(coords,name='Uphi0', bases=disk.radial_basis)
        Uphi0['g'][0] = np.outer(np.ones(1),uphi0_func(s_grid))

        ## Problem
        problem = d3.EVP([u, p, tau_u, tau_p], eigenvalue=sigma, namespace=locals())
        problem.add_equation("div(u) + tau_p = 0")
        problem.add_equation("np.sqrt(Ek)*dt(u) + Uphi0@grad(u) + u@grad(Uphi0) + grad(p) - Ek*lap(u) + lift(tau_u) = 0")
        problem.add_equation("u(s=R) = 0")
        problem.add_equation("integ(p) = 0")

        # Solver
        solver = problem.build_solver()

        # Set up arrays for saving
        Ns_save = min((2*M+1),ns)
        m_list = np.arange(-M,M+1)
        sigma_list = np.zeros((len(m_list),Ns_save),dtype=dtype)
        u_list = np.zeros((len(m_list),Ns_save,2,nphi,ns),dtype=dtype)
        p_list = np.zeros((len(m_list),Ns_save,nphi,ns),dtype=dtype)
        vort_list = np.zeros((len(m_list),Ns_save,nphi,ns),dtype=dtype)

        for i,m in enumerate(m_list):
            sp = solver.subproblems_by_group[(m, None)]
            solver.solve_dense(sp, rebuild_matrices=True)
            evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
            evals = evals[np.argsort(-evals.real)][:Ns_save]
            sigma_list[i] = evals

            for j,each_eval in enumerate(evals):
                solver.set_state(np.argmin(np.abs(solver.eigenvalues - each_eval)), sp.subsystems[0])

                vort = -d3.div(d3.skew(u)).evaluate() # Compute vorticity

                u_list[i,j] = np.copy(u['g'])
                p_list[i,j] = np.copy(p['g'])
                vort_list[i,j] = np.copy(vort['g'])

        # Print slowest decaying mode
        m_slowest_decaying = m_list[m_list>0][np.argmax(sigma_list[:,0].real[m_list>0])]
        sigma_slowest_decaying = sigma_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0]
        print(f"Slowest decaying mode: m = {m_slowest_decaying}, sigma = {sigma_slowest_decaying:.2f}")

        # Save at each point in time
        each_save_dict = {'write_num': write_num, 't': t_data, 'Omega': 1+DelOmega_data, 's': s_grid[0,:], 'phi': phi_grid[:,0], 'uphi0': Uphi0['g'][0][0].real, 'm': m_list, 'sigma': sigma_list, 'p': p_list, 'u': u_list, 'vort': vort_list, 'params': params_evp}
        each_save_dict_path = subfolders_path.joinpath(f"{write_num:06}.pickle")
        with open(each_save_dict_path, 'wb') as handle:
            pickle.dump(each_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)