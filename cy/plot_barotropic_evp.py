"""
Plot eigenvalues and slowest-decaying eigenmode from EVP.

Usage:
    plot_barotropic_evp.py <evp_name>
"""

import pickle
import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import pathlib
import glob
from docopt import docopt
from dedalus.tools.general import natural_sort

args = docopt(__doc__)
evpname = args['<evp_name>']

data_path = pathlib.Path('data').absolute()
params_path = pathlib.Path('params').absolute()

save_params_path = params_path.joinpath(evpname+".json")
subfolders_path = data_path.joinpath(evpname)

subfiles = natural_sort(glob.glob(str(subfolders_path.joinpath("*"))))

# Create output directory if needed
frame_path = pathlib.Path('frames').absolute()
output_path = frame_path.joinpath(evpname)

if not frame_path.exists():
    frame_path.mkdir()
if not output_path.exists():
    output_path.mkdir()

print("Getting time series...")

# Get time series

Omega_arr = []
t_arr = []
rel_uphi_maxes = []
all_sigmas = []
for subfile in subfiles:
    with open(subfile, 'rb') as handle:
        evp_dict = pickle.load(handle)
    Omega_arr.append(evp_dict['Omega'])
    t_arr.append(evp_dict['t'])
    rel_uphi_maxes.append(np.max(np.abs(evp_dict['uphi0'] - evp_dict['Omega']*evp_dict['s'])))
    all_sigmas.append(evp_dict['sigma'])

Omega_arr = np.array(Omega_arr)
t_arr = np.array(t_arr)
rel_uphi_max = np.max(rel_uphi_maxes)
all_sigmas = np.array(all_sigmas)

# Main loop
for k in range(len(subfiles)):
    subfile = subfiles[k]
    with open(subfile, 'rb') as handle:
        evp_dict = pickle.load(handle)
    
    # Extract data
    m_list = evp_dict['m']
    sigma_list = evp_dict['sigma']
    u_list = evp_dict['u']
    vort_list = evp_dict['vort']
    params = evp_dict['params']
    R = params['R']
    nphi = params['nphi']
    ns = params['ns']
    t = evp_dict['t']
    s = evp_dict['s']
    Omega = evp_dict['Omega']
    uphi0 = evp_dict['uphi0']

    # Get slowest decaying mode
    m_slowest_decaying = m_list[m_list>0][np.argmax(sigma_list[:,0].real[m_list>0])]
    sigma_slowest_decaying = sigma_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0]

    # Make dedalus bases for upscaling fields associated with slowest decaying mode
    ## Bases
    coords_plot = d3.PolarCoordinates('phi', 's')
    dist_plot = d3.Distributor(coords_plot, dtype=np.complex128)
    disk_plot = d3.DiskBasis(coords_plot, shape=(nphi, ns), radius=R, dtype=np.complex128)
    phi_grid, s_grid = dist_plot.local_grids(disk_plot)

    ## Fields
    u_plot = dist_plot.VectorField(coords_plot, name='u_plot', bases=disk_plot)
    vort_plot = dist_plot.Field(name='vort_plot', bases=disk_plot)
    u_plot['g'] = u_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0]
    vort_plot['g'] = vort_list[np.argmin(np.abs(m_list - m_slowest_decaying)),0]

    ## Change scales
    scales = (max(1,256//nphi), max(1,256//ns))
    u_plot.change_scales(scales)
    vort_plot.change_scales(scales)
    phi_plot, s_plot = dist_plot.local_grids(disk_plot, scales=scales)
    x, y = coords_plot.cartesian(phi_plot, s_plot)

    ## Compute streamfunction: \boldsymbol{u} = \boldsymbol{\nabla} \times (-\psi \boldsymbol{e}_z)
    psi_plot = 1j*s_plot*u_plot['g'][1]/m_slowest_decaying

    ## Phase shift
    max_phi_idx = np.unravel_index(np.argmax(psi_plot.real), psi_plot.shape)[0]
    phase_shift = (phi_plot[max_phi_idx,0]%(2*np.pi/m_slowest_decaying))/(2*np.pi/m_slowest_decaying) * 2*np.pi 

    # Plot
    shading = 'gouraud'
    cmap = 'RdBu_r'

    fig, axs = plt.subplot_mosaic([['time','.','evals','.','t','t'],['time','b1','evals','b2','uphi','us'],['b3','b1','evals','b2','vort','psi'],['prof','b1','evals','b2','vort','psi']],width_ratios=[1,0.3,2,0.5,1.2,1.2],height_ratios=[0.1,1,0.3,1],figsize=(10,5))
                
    axs['time'].plot(t_arr,Omega_arr)
    axs['time'].scatter(t,Omega)
    axs['time'].set_ylabel(r"$\Omega(t)/\Omega_0$")
    axs['time'].set_xlabel(r"$t \sqrt{\nu \Omega_0}/H$")
    axs['time'].set_title(f"$t \\sqrt{{\\nu \\Omega_0}}/H = {t:.2f}$")

    axs['prof'].plot(s,(uphi0 - (Omega)*s))
    axs['prof'].set_ylim(-rel_uphi_max,rel_uphi_max)
    axs['prof'].grid()
    axs['prof'].set_xlabel(r'$s/H$')
    axs['prof'].set_ylabel(r"$u_\phi/(\Omega H)$")

    marker_list = ['o','>','d','<','*','^','P','v','X']

    for i in range(len(m_list)):
        m = m_list[-1-i]
        color = np.array(plt.cm.RdBu(i/len(m_list)))
        color[-1] = 0.3
        axs['evals'].scatter(sigma_list[-1-i].real,sigma_list[-1-i].imag,facecolor=color,edgecolors=plt.cm.RdBu(i/len(m_list)),linewidths=1,label=f"{m}",marker=marker_list[np.abs(m)%len(marker_list)])
    axs['evals'].axvline(color='k')
    axs['evals'].axhline(color='k')
    axs['evals'].grid()
    axs['evals'].legend(ncols=1,bbox_to_anchor=(1.2,0.5),loc='center',title="$m$")
    axs['evals'].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs['evals'].set_title(r"Growth rate $\tilde{\sigma} = \sigma H/\sqrt{\nu \Omega_0}$")
    axs['evals'].set_ylabel(r"$\Im\{\tilde{\sigma}\}$")
    axs['evals'].set_xlabel(r"$\Re\{\tilde{\sigma}\}$")
    xlim_r = max(np.min(all_sigmas.real),max(0.1*np.abs(np.min(all_sigmas.real)),np.max(all_sigmas.real)))
    axs['evals'].set_xlim(np.min(all_sigmas.real),xlim_r)
    axs['evals'].set_ylim(np.min(all_sigmas.imag),np.max(all_sigmas.imag))

    axs['uphi'].pcolormesh(x, y, (np.exp(1j*phase_shift)*u_plot['g'][0]).real, cmap=cmap, shading=shading)
    axs['uphi'].set_title(r"$u_\phi$")

    axs['us'].pcolormesh(x, y, (np.exp(1j*phase_shift)*u_plot['g'][1]).real, cmap=cmap, shading=shading)
    axs['us'].set_title(r"$u_s$")

    axs['vort'].pcolormesh(x, y, (np.exp(1j*phase_shift)*vort_plot['g']).real, cmap=cmap, shading=shading)
    axs['vort'].set_title(r"$\omega$")

    axs['psi'].pcolormesh(x, y, (np.exp(1j*phase_shift)*psi_plot).real, cmap=cmap, shading=shading)
    axs['psi'].set_title(r"$\psi$")

    for key in ['uphi','us','vort','psi']:
        axs[key].set_aspect('equal')
        axs[key].set_axis_off()
        axs[key].set_xlim(-R,R)
        axs[key].set_ylim(-R,R)

    axs['b1'].set_axis_off()
    axs['b2'].set_axis_off()
    axs['b3'].set_axis_off()
    axs['t'].set_axis_off()

    title = "Slowest decaying mode:\n"
    title = title + f"$m = {m_slowest_decaying},\\; \\tilde{{\\sigma}} = {sigma_slowest_decaying:.2f}$".replace("j","i")
    if sigma_slowest_decaying.real > 0:
        color = 'r'
    else:
        color = 'k'
    axs['t'].set_title(title,color=color)

    write_num = evp_dict['write_num']
    savepath = output_path.joinpath('write_%06i.jpg' %(write_num))

    print(f"Saving image {k+1}/{len(t_arr)}\r", end="")

    plt.savefig(str(savepath), dpi=300)
    plt.close()