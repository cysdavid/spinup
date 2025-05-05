"""
Plot barotropic IVP data in inertial frame

Usage:
    plot_barotropic_ivp.py <sim_name> [--stride=<strd>]

Options:
    --stride=<strd>  Interval of snapshots to plot [default: 1]
"""
import dedalus.public as d3
import h5py
import json
from docopt import docopt
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from dedalus.tools.general import natural_sort

from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

args = docopt(__doc__)
sim_name = args['<sim_name>']
strd = int(args['--stride'])
data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()
save_data_path = data_path.joinpath(sim_name)
save_params_path = params_path.joinpath(sim_name+".json")
data_files = natural_sort(glob.glob(str(save_data_path.joinpath('*.h5'))))

# Load parameters
with open(save_params_path) as f: 
    params = json.load(f)

# Get time series and maxima
vort_quant = 0.9

Omega_arr = []
t_arr = []
rel_uphi_maxes = []
uphi_maxes = []
vort_maxes = []
us_maxes = []

for filename in data_files:
    with h5py.File(filename, mode='r') as f:
        if filename == data_files[0]:
            phi_data = f['tasks']['u'].dims[2][0][:]
            s_data = f['tasks']['u'].dims[3][0][:]
            ss_data,phiphi_data = np.meshgrid(s_data,phi_data)

        uphi_rel_maxes = []
        file_vort_maxes = []
        for it in range(0,f['tasks']['Omega'].shape[0],strd):
            Omega_field = f['tasks']['Omega']
            Omega_i = Omega_field[it][0]
            t_i = Omega_field.dims[0]['sim_time'][it]
            Omega_arr.append(Omega_i)
            t_arr.append(t_i)
            
            uphi_rel_maxes.append(np.nanmax(np.abs(f['tasks']['u'][it][0] - (Omega_i*ss_data))))
            file_vort_maxes.append(np.nanquantile(np.abs(f['tasks']['vort'][it]),q=vort_quant))

        file_uphi_rel_max = np.nanmax(uphi_rel_maxes)
        file_uphi_max = np.nanmax(np.abs(f['tasks']['u'][::strd,0]))
        file_us_max = np.nanmax(np.abs(f['tasks']['u'][::strd,1]))
        file_vort_max = np.nanquantile(file_vort_maxes,q=vort_quant)
        rel_uphi_maxes.append(file_uphi_rel_max)
        uphi_maxes.append(file_uphi_max)
        us_maxes.append(file_us_max)
        vort_maxes.append(file_vort_max)

Omega_arr = np.array(Omega_arr)
t_arr = np.array(t_arr)
rel_uphi_max = np.max(rel_uphi_maxes)
uphi_max = np.max(uphi_maxes)
us_max = np.max(us_maxes)
vort_max = np.quantile(vort_maxes,q=vort_quant)

# Make Cartesian grid
x = ss_data * np.cos(phiphi_data)
y = ss_data * np.sin(phiphi_data)

# Create output directory if needed
frame_path = pathlib.Path('frames').absolute()
output_path = frame_path.joinpath(sim_name)

if rank == 0:
    print("Done.\n")
    if not frame_path.exists():
        frame_path.mkdir()
    if not output_path.exists():
        output_path.mkdir()

    print("Plotting...")

# Import data and plot
for filename in data_files:
    with h5py.File(filename, mode='r') as f:
        for it in range(strd*rank,f['tasks']['u'].shape[0],strd*size):
            
            write_num = f['tasks']['u'].dims[0]['write_number'][it]
            Omega_data = f['tasks']['Omega'][it][0][0]
            u_data = f['tasks']['u'][it]
            vort_data = f['tasks']['vort'][it]
            p_data = f['tasks']['p'][it]
            t_data = f['tasks']['u'].dims[0]['sim_time'][it]

            fig, axs = plt.subplot_mosaic([['time','field','vort']],figsize=(1*7.5,5/2))
            axs['time'].plot(t_arr,Omega_arr)
            axs['time'].scatter(t_data,Omega_data)
            axs['time'].set_ylabel(r"$\Omega(t)/\Omega_0$")
            axs['time'].set_xlabel(r"$t \sqrt{\nu \Omega_0}/H$")

            axs['field'].pcolormesh(x, y, u_data[1], cmap='RdBu_r', vmin=(-us_max,us_max))
            axs['field'].set_aspect('equal')
            axs['field'].set_axis_off()
            axs['field'].set_title(r"$u_s$")

            axs['vort'].pcolormesh(x, y, vort_data, cmap='RdBu_r', vmin=(-vort_max,vort_max))
            axs['vort'].set_aspect('equal')
            axs['vort'].set_axis_off()
            axs['vort'].set_title(r"$\omega$")

            axs['time'].set_title(f"$t \\sqrt{{\\nu \\Omega_0}}/H = {t_data:.2f}$")

            savepath = output_path.joinpath('write_%06i.jpg' %(write_num))

            if rank==0:
                print(f"Saving image {int(write_num/strd)}/{len(t_arr)}\r", end="")

            plt.savefig(str(savepath), dpi=300)
            plt.close()