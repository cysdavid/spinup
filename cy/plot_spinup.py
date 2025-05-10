"""
Plot planes from joint analysis files.

Usage:
    plot_spinup.py <sim_name> [--stride=<strd>]

Options:
    --stride=<strd>  Interval of snapshots to plot [default: 1]
"""

import h5py
import json
from docopt import docopt
import glob
import pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from dedalus.tools.general import natural_sort
from dedalus.tools.parallel import Sync

from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

args = docopt(__doc__)
sim_name = args['<sim_name>']
strd = int(args['--stride'])
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
save_data_path = data_path.joinpath(sim_name)
static_fields_path = save_data_path.joinpath('static_fields')
save_params_path = params_path.joinpath(sim_name+".json")
data_files = natural_sort(glob.glob(str(save_data_path.joinpath('*.h5'))))
static_fields_files = natural_sort(glob.glob(str(static_fields_path.joinpath('*.h5'))))

# Load parameters
with open(save_params_path) as f: 
    params= json.load(f)

# Strides for quiver plot
quivspacing = 0.02
s_quivstrd = int(quivspacing/(params['Ls']/params['ns']))
z_quivstrd = int(quivspacing/((params['Lz']+2*params['w'])/params['nz']))

with Sync() as sync:
    if sync.comm.rank == 0:
        print("Getting time series...")

# Get time series and maxima

vort_quant = 0.99

DelOmega_arr = []
t_arr = []
rel_uphi_maxes = []
uphi_maxes = []
vort_maxes = []
us_maxes = []
uz_maxes = []

for filename in data_files:
    with h5py.File(filename, mode='r') as f:
        if filename == data_files[0]:
            s_data = f['tasks']['uphi'].dims[2][0][:]
            z_data = f['tasks']['uphi'].dims[1][0][:]

            ss_data = np.outer(np.ones(len(z_data)),s_data)
            zz_data = np.outer(z_data,np.ones(len(s_data)))

            ss_reg = np.outer(np.ones(params['nz']),np.linspace(s_data[0],s_data[-1],params['ns']))
            zz_reg = np.outer(np.linspace(z_data[0],z_data[-1],params['nz']),np.ones(params['ns']))

        uphi_rel_maxes = []
        file_vort_maxes = []
        for it in range(0,f['tasks']['DelOmega'].shape[0],strd):
            DelOmega_field = f['tasks']['DelOmega']
            DelOmega_i = DelOmega_field[it][0]
            t_i = DelOmega_field.dims[0]['sim_time'][it]
            DelOmega_arr.append(DelOmega_i)
            t_arr.append(t_i)
            
            uphi_rel_maxes.append(np.nanmax(np.abs(f['tasks']['uphi'][it]-DelOmega_i*ss_data)))
            file_vort_maxes.append(np.nanquantile(np.abs(f['tasks']['vort_phi'][it]),q=vort_quant))

        file_uphi_rel_max = np.nanmax(uphi_rel_maxes)
        file_uphi_max = np.nanmax(np.abs(f['tasks']['uphi'][::strd]))
        file_us_max = np.nanmax(np.abs(f['tasks']['us'][::strd]))
        file_uz_max= np.nanmax(np.abs(f['tasks']['uz'][::strd]))
        file_vort_max = np.nanquantile(file_vort_maxes,q=vort_quant)
        rel_uphi_maxes.append(file_uphi_rel_max)
        uphi_maxes.append(file_uphi_max)
        us_maxes.append(file_us_max)
        uz_maxes.append(file_uz_max)
        vort_maxes.append(file_vort_max)

DelOmega_arr = np.array(DelOmega_arr)
t_arr = np.array(t_arr)
rel_uphi_max = np.max(rel_uphi_maxes)
uphi_max = np.max(uphi_maxes)
us_max = np.max(us_maxes)
uz_max = np.max(uz_maxes)
vort_max = np.quantile(vort_maxes,q=vort_quant)

# Create output directory if needed
frame_path = pathlib.Path('frames').absolute()
output_path = frame_path.joinpath(sim_name)

with Sync() as sync:
    if sync.comm.rank == 0:
        print("Done.\n")
        if not frame_path.exists():
            frame_path.mkdir()
        if not output_path.exists():
            output_path.mkdir()

        print("Plotting...")

# Import data and plot
for filename in data_files:
    with h5py.File(filename, mode='r') as f:
        for it in range(strd*rank,f['tasks']['uphi'].shape[0],strd*size):
            
            write_num = f['tasks']['uphi'].dims[0]['write_number'][it]
            DelOmega_data = f['tasks']['DelOmega'][it][0][0]
            uphi_data = f['tasks']['uphi'][it]
            vort_data = f['tasks']['vort_phi'][it]
            us_data = f['tasks']['us'][it]
            uz_data = f['tasks']['uz'][it]
        
            t_data = f['tasks']['E_phi'].dims[0]['sim_time'][it]

            us_interp_fxn = RegularGridInterpolator((z_data,s_data),us_data, bounds_error=False)
            us_reg = us_interp_fxn((zz_reg,ss_reg))

            uz_interp_fxn = RegularGridInterpolator((z_data,s_data),uz_data, bounds_error=False)
            uz_reg = uz_interp_fxn((zz_reg,ss_reg))

            fig, axs = plt.subplot_mosaic([['time','field','vort'],['prof','field','vort']],figsize=(0.5*7.5 + 15*params['Ls']/params['Lz'],5))
            
            axs['time'].plot(t_arr,1+DelOmega_arr)
            axs['time'].scatter(t_data,1+DelOmega_data)
            axs['time'].set_ylabel(r"$\Omega(t)/\Omega_0$")
            axs['time'].set_xlabel(r"$t \sqrt{\nu \Omega_0}/H$")
            
            pm = axs['field'].pcolormesh(ss_data,zz_data,(uphi_data - DelOmega_data*ss_data),cmap='RdBu_r',vmin=(-rel_uphi_max,rel_uphi_max),shading='gouraud')
            plt.colorbar(pm,ax=axs['field'],extend='both',label=r"$u_\phi/(\Omega H)$")
            axs['field'].quiver(ss_reg[::z_quivstrd,::s_quivstrd],zz_reg[::z_quivstrd,::s_quivstrd],us_reg[::z_quivstrd,::s_quivstrd],uz_reg[::z_quivstrd,::s_quivstrd],scale=np.sqrt(us_max**2+uz_max**2)*5)
            axs['field'].axhline(params['Lz']/2,color='k')
            axs['field'].axhline(-params['Lz']/2,color='k')
            axs['field'].set_aspect(1)
            axs['field'].set_xlabel(r'$s/H$')
            axs['field'].set_ylabel(r'$z/H$')

            pm = axs['vort'].pcolormesh(ss_data,zz_data,vort_data,cmap='PuOr_r',vmin=(-vort_max,vort_max))
            plt.colorbar(pm,ax=axs['vort'],extend='both',label=r"$\Omega^{-1} (\mathbf{\nabla} \times \mathbf{u}) \cdot \mathbf{e}_\phi$")
            axs['vort'].axhline(params['Lz']/2,color='k')
            axs['vort'].axhline(-params['Lz']/2,color='k')
            axs['vort'].set_aspect(1)
            axs['vort'].set_xlabel(r'$s/H$')

            axs['prof'].plot(s_data,(uphi_data - DelOmega_data*ss_data)[np.argmin(np.abs(z_data)),:])
            axs['prof'].set_ylim(-rel_uphi_max,rel_uphi_max)
            axs['prof'].grid()
            axs['prof'].set_xlabel(r'$s/H$')
            axs['prof'].set_ylabel(r"$u_\phi/(\Omega H)$")

            plt.suptitle(f"$t \\sqrt{{\\nu \\Omega_0}}/H = {t_data:.2f}$")

            plt.subplots_adjust(hspace=0.4)

            savepath = output_path.joinpath('write_%06i.jpg' %(write_num))

            if rank==0:
                print(f"Saving image {int(write_num/strd)}/{len(t_arr)}\r", end="")

            plt.savefig(str(savepath), dpi=300)
            plt.close()
        