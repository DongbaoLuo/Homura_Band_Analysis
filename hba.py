# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 04:02:45 2019

@author: Homura
"""

import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cut_gaussian(x, x0, sigma=0.000005, cut=8):
    prefix = 1 / sigma / np.sqrt(2*np.pi)
    gcut = prefix * np.exp(-cut**2/2)

    return prefix * np.exp(-(x-x0)**2/(2*sigma**2))
def get_outcar():
    inFile = 'OUTCAR'
    outcar = [line for line in open(inFile) if line.strip()]
    
    for ii, line in enumerate(outcar):
    #    print(ii,line)
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])
    
        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])
    
        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1
    
        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1
    
        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1
            break
    
    # basis vector of reciprocal lattice
    B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]], 
                 dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in outcar[Lvkpts:Lvkpts+nkpts]],
                   dtype=float)
    vkpts = tmp[:,:3]
    wkpts = tmp[:,-1]
    
    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        bands.append(float(line.split()[1]))
    
    # vkpts = np.array(vkpts, dtype=float)[:nkpts]
    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))
    
    if os.path.isfile('KPOINTS'):
        kpoints = [line for line in open('KPOINTS') if line.strip()]
    
    if os.path.isfile('KPOINTS') and kpoints[2][0].upper() == 'L':
            Nk_in_seg = int(kpoints[1].strip())
            kpts_namelist=[]
            kpts_namelist.append(kpoints[4].split()[3][1:])
            for ii in range(int((len(kpoints)-4)/2)):
                kpts_namelist.append(kpoints[4+(2*ii+1)].split()[3][1:])
            Nseg = int(nkpts / Nk_in_seg)
            vkpt_diff = np.zeros_like(vkpts, dtype=float)
            
            for ii in range(Nseg):
                start = ii * Nk_in_seg
                end = (ii + 1) * Nk_in_seg
                vkpt_diff[start:end, :] = vkpts[start:end,:] - vkpts[start,:]
        
            kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
            # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
            for ii in range(1, Nseg):
                start = ii * Nk_in_seg
                end = (ii + 1) * Nk_in_seg
                kpt_path[start:end] += kpt_path[start-1]
        
            kpt_path /= kpt_path[-1]
            kpt_bounds =  np.concatenate((kpt_path[0::Nk_in_seg], [1.0,]))
    elif os.path.isfile('syml'):
        syml = [line for line in open('syml') if line.strip()]
        Nseg = int(syml[0].split()[0])-1
        Nk_in_seg = int(syml[0].split()[0])
        vkpts_use = []
        nkpts = 0
        for ii in range(len(wkpts)):
            if wkpts[ii] == 0:
                vkpts_use.append(vkpts[ii])
                nkpts += 1
        # get band path
        vkpt_diff = np.diff(vkpts_use, axis=0)
        kpt_path = np.zeros(nkpts, dtype=float)
        kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
        kpt_path /= kpt_path[-1]
    
        # get boundaries of band path
        xx = np.diff(kpt_path)
        kpt_bounds = np.concatenate(([0.0,], kpt_path[np.isclose(xx, 0.0)], [1.0,]))
    return Efermi,wkpts,kpt_bounds,kpt_path,bands,kpts_namelist

################################################################################

Efermi, wkpts, kpt_bounds, kpt_path, bands, kpts_namelist = get_outcar()
ispin, nkpts, nband = bands.shape

# set energy zeros to Fermi energy
Ezero = Efermi
 
bands -= Ezero
        
################################################################################
# The Plotting part
################################################################################
def bandplot():
    import matplotlib as mpl
    import matplotlib.pyplot as plt    
    from scipy.interpolate import interp1d
    
    mpl.rcParams['axes.unicode_minus'] = False

    plotemin = -2
    plotemax = 2
    fig = plt.figure()
    fig.set_size_inches((8.0, 4.0))
    
    ax  = plt.subplot(111)
    plt.subplots_adjust(left=0.12, right=0.95,
                        bottom=0.08, top=0.95,
                        wspace=0.10, hspace=0.10)
    
    divider = make_axes_locatable(ax)
    axDos = divider.append_axes('right', size='45%', pad=0.10)
    
    clrs = ['r', 'b']
    
    for ii in np.arange(ispin):
        for jj in np.arange(nband):
    
#            ax.plot(kpt_path, bands[ii,:,jj], '-',
#                    ms=3, mfc=clrs[ii],mew=0.0,
#                    color='k', lw=1.0,
#                    alpha=0.6)
            ax.plot(kpt_path, bands[ii,:,jj],
                    color='k',
                    alpha=0.6,lw=2)
    
    for bd in kpt_bounds:
    #    print(bd)
        ax.axvline(x=bd, ls=':', color='k', lw=0.5, alpha=0.6)
    
    ax.axhline(y=0.0, ls=':', color='k', lw=0.5, alpha=0.6)
    ax.set_ylim(plotemin, plotemax)
    ax.minorticks_on()
    ax.tick_params(which='both', labelsize='large')
    
    ax.set_ylabel('Energy [eV]', fontsize='large')
    
    # pos = [0,] + list(kpt_bounds) + [1,]
    # ax.set_xticks(pos)
    ax.set_xticks(kpt_bounds)
    kpts_name =[xx for xx in kpts_namelist][:kpt_bounds.size]
    # kpts_name =['M', r'$\Gamma$', 'K', 'M']
    ax.set_xticklabels(kpts_name, fontsize='large')                   
    #
    ################################################################################
    EXTRA = 0.10
    NEDOS = 1000
    SIGMA = 0.05
    DOS = np.zeros((NEDOS, ispin), dtype=float)
    
    emin = bands.min()
    emax = bands.max()
    eran = emax - emin
    
    emin = emin - EXTRA * eran
    emax = emax + EXTRA * eran
    
    x = np.linspace(emin, emax, NEDOS)
    ################################################################################
    # dos generation
    
    for sp in range(ispin):
        for kp in range(nkpts):
            for nb in range(nband):
                en = bands[sp,kp,nb]
                DOS[:,sp] += cut_gaussian(x, en, SIGMA, cut=6.0)
    
        axDos.plot(DOS[:,sp], x, ls='-', color=clrs[sp], lw=1.5, alpha=0.6)
    
    axDos.set_xlabel('DOS [a.u.]', fontsize='large')
    
    axDos.set_ylim(plotemin, plotemax)
    axDos.set_xticks([])
    axDos.set_yticklabels([])
    axDos.set_xticklabels([])
    
    axDos.minorticks_on()
    axDos.tick_params(which='both', labelsize='large')
    
    plt.savefig('band.png', dpi=720)
    plt.show()

def get_poscar():
    poscar = [line for line in open('POSCAR') if line.strip()]
    atom_name = poscar[5].split()
    atom_num_list = poscar[6].split()
    atom_num = []
    atom_total = 0
    for terms in atom_num_list:
        atom_num.append(int(terms))
        atom_total += int(terms)
    return atom_name,atom_num,atom_total

def get_procar():
    procar = [line for line in open('PROCAR') if line.strip()]
#    poscar = [line for line in open('POSCAR') if line.strip()]
#    atom_name = poscar[5].split()
#    atom_num_list = poscar[6].split()
    orbit_list = procar[4].split()
    orbit_list.remove('ion')
    orbit_num = len(orbit_list)
#    atom_num = []
#    atom_total = 0
#    for terms in atom_num_list:
#        atom_num.append(int(terms))
#        atom_total += int(terms)
    atom_name,atom_num,atom_total = get_poscar()
    Efermi, wkpts, kpt_bounds, kpt_path, bands, kpts_namelist = get_outcar()
    ispin, nkpts, nband = bands.shape
    atom_total += 1
    kpoint_line = []
    if ispin == 1:
        spin_list = ['s_tot','s_x','s_y','s_z']
        spin_num = 4
        band_project = np.zeros([nkpts,nband,spin_num,atom_total,orbit_num])
        for ii,line in enumerate(procar):
            if 'k-point' in line and 'k-points' not in line:
                kpoint_line.append(ii)
        for ii in range(len(kpoint_line)-1):
            k_index = ii
            band_perk = procar[kpoint_line[k_index]:kpoint_line[k_index+1]]
            for jj,band_line in enumerate(band_perk):
                if 'band' in band_line:
                    band_index = int(band_line.split()[1])-1
                    for spin_index in range(spin_num):
                        for atom_index in range(atom_total):
                            for orbit_index in range(orbit_num):
                                to_add = band_perk[jj+2+(atom_total)*spin_index+atom_index].split()
                                to_add = to_add[1:]
                                to_add1 = []
                                for terms in to_add:
                                    to_add1.append(float(terms))
                                band_project[k_index,band_index,spin_index,atom_index,orbit_index]\
                                =to_add1[orbit_index]
        k_index = len(kpoint_line)-1
        band_perk = procar[kpoint_line[k_index]:]
        for jj,band_line in enumerate(band_perk):
            if 'band' in band_line:
                band_index = int(band_line.split()[1])-1
                for spin_index in range(spin_num):
                    for atom_index in range(atom_total):
                        for orbit_index in range(orbit_num):
                            to_add = band_perk[jj+2+(atom_total)*spin_index+atom_index].split()
                            to_add = to_add[1:]
                            to_add1 = []
                            for terms in to_add:
                                to_add1.append(float(terms))
                            band_project[k_index,band_index,spin_index,atom_index,orbit_index]\
                            =to_add1[orbit_index]
    return band_project,atom_name,atom_num,atom_total,orbit_list,orbit_num,spin_list,spin_num

def cl_blend(color_array):
    import numpy as np
    #color_array = np.array([0.2,0.2,0.2,0.2,0.2])
    color_list = ['#5edc1f','#0bf9ea','#0d75f8','#ff08e8','#fd3c06','#fffd01']
    dimension = len(color_array)
    r = []
    g = []
    b = []
    for terms in color_list[:dimension]:
        r.append(int('0x'+terms[1:3],16))
        g.append(int('0x'+terms[3:5],16))
        b.append(int('0x'+terms[5:7],16))
    r_blend = hex(int(np.dot(color_array,np.array(r))))
    g_blend = hex(int(np.dot(color_array,np.array(g))))
    b_blend = hex(int(np.dot(color_array,np.array(b))))
    color_blend = '#'+r_blend[2:].zfill(2)+g_blend[2:].zfill(2)\
    +b_blend[2:].zfill(2)
    return color_blend,color_list

##################################################

def fatband():
    import matplotlib as mpl
    import matplotlib.pyplot as plt    
    from scipy.interpolate import interp1d
    
    eps = 200
    plotemin = -2
    plotemax = 2
    
    band_project,atom_name,atom_num,atom_total,orbit_list,orbit_num,spin_list,spin_num = get_procar()
    orbit_choose = ['s','p']
    orbit_position = []
    spin_choose = ['s_tot']
    spin_position = []
    band_sum = np.zeros([nkpts,nband,len(spin_choose),len(atom_num),len(orbit_choose)])
    atom_position = [0]+np.cumsum(atom_num).tolist()[:-1]
    orbit_len = 0
    if 's' in orbit_choose:
        orbit_position.append([0])
        orbit_len += 1
    if 'p' in orbit_choose:
        orbit_position.append([1,2,3])
        orbit_len += 3
    if 'd' in orbit_choose:
        orbit_position.append([4,5,6,7,8])
        orbit_len += 5
    for ii in spin_choose:
        for jj,terms in enumerate(spin_list):
            if ii == terms:
                spin_position.append(jj)
    for spin_index in range(len(spin_choose)):
        for orbit_index in range(len(orbit_position)):
            for orbit_add in range(len(orbit_position[orbit_index])):
                for ii in range(len(atom_num)):
                    for jj in range(atom_num[ii]):
            #            band_sum[:,:,:,ii,:] = band_project[:,:,:,jj+atom_position[ii],:]
                        band_sum[:,:,spin_index,ii,orbit_index] += band_project[\
                                :,:,spin_position[spin_index],jj+atom_position[ii],\
                                orbit_position[orbit_index][orbit_add]]
    for_flatten = band_sum.shape[2:]
    tot_d = np.cumprod(for_flatten)[-1]
    fat_weight = np.zeros([nkpts,nband,tot_d])
    #for k_index in range(nkpts):
    #    for b_index in range(nband):
    #        band_sum[k_index,b_index,spin_index,ii,orbit_index]
    count = 0
    legend = []
    for spin_index in range(for_flatten[0]):
        for atom_index in range(for_flatten[1]):
            for orbit_index in range(for_flatten[2]):
                fat_weight[:,:,count] = band_sum[:,:,spin_index,atom_index,orbit_index]
                count += 1
                legend.append(atom_name[atom_index]+' '+spin_choose[spin_index]+' '+orbit_choose[orbit_index])
                
    for k_index in range(nkpts):
        for b_index in range(nband):
            fat_weight[k_index,b_index] /= sum(fat_weight[k_index,b_index])
    
    mpl.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    fig.set_size_inches((8.0, 4.0))
    
    ax  = plt.subplot(111)
    plt.subplots_adjust(left=0.12, right=0.95,
                        bottom=0.08, top=0.90,
                        wspace=0.10, hspace=0.10)
    
    divider = make_axes_locatable(ax)
    axDos = divider.append_axes('right', size='35%', pad=0.10)
    
    clrs = ['r', 'b']
    fat_weight_interp = np.zeros([eps,nband,count])
    for ii in np.arange(ispin):
        for jj in np.arange(nband):
    #ii=0
    #jj=0
            f2=interp1d(kpt_path,bands[ii,:,jj],kind='linear')
            x_for_plot = np.linspace(0,1,eps)
            for kk in range(count):
                f1=interp1d(kpt_path,fat_weight[:,jj,kk],kind='linear')
                fat_weight_interp[:,jj,kk] = f1(x_for_plot)
                
            y_for_plot = f2(x_for_plot)
    
            x_line = []
            y_line = []
            for ll in range(len(x_for_plot)-1):
                x_line.append([x_for_plot[ll],x_for_plot[ll+1]])
                y_line.append([y_for_plot[ll],y_for_plot[ll+1]])
            for ll in range(len(x_line)):
                ax.plot(x_line[ll],y_line[ll],c=cl_blend(fat_weight_interp[ll,jj,:])[0],alpha=0.5,lw=1.5)
    
    for bd in kpt_bounds:
    #    print(bd)
        ax.axvline(x=bd, ls=':', color='k', lw=0.5, alpha=0.6)
    
    ax.axhline(y=0.0, ls=':', color='k', lw=0.5, alpha=0.6)
    ax.set_ylim(plotemin, plotemax)
    ax.minorticks_on()
    ax.tick_params(which='both', labelsize='large')
    for ii in range(tot_d):
        ax.plot([0.5,0.5],[0.5,0.5],lw=2,c=cl_blend(fat_weight_interp[ll,jj,:])[1][ii],label=legend[ii])
    ax.set_ylabel('Energy [eV]', fontsize='large')
    
    # pos = [0,] + list(kpt_bounds) + [1,]
    # ax.set_xticks(pos)
    ax.set_xticks(kpt_bounds)
    kpts_name =[xx for xx in kpts_namelist][:kpt_bounds.size]
    # kpts_name =['M', r'$\Gamma$', 'K', 'M']
    ax.set_xticklabels(kpts_name, fontsize='large')
    ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.05),ncol=tot_d)              
    #
    ################################################################################
    EXTRA = 0.10
    NEDOS = 1000
    SIGMA = 0.05
    DOS = np.zeros((NEDOS, ispin), dtype=float)
    
    emin = bands.min()
    emax = bands.max()
    eran = emax - emin
    
    emin = emin - EXTRA * eran
    emax = emax + EXTRA * eran
    
    x = np.linspace(emin, emax, NEDOS)
    ################################################################################
    # dos generation
    
    for sp in range(ispin):
        for kp in range(nkpts):
            for nb in range(nband):
                en = bands[sp,kp,nb]
                DOS[:,sp] += cut_gaussian(x, en, SIGMA, cut=6.0)
    
        axDos.plot(DOS[:,sp], x, ls='-', color=clrs[sp], lw=2, alpha=0.6)
    
    axDos.set_xlabel('DOS [a.u.]', fontsize='large')
    
    axDos.set_ylim(plotemin, plotemax)
    axDos.set_xticks([])
    axDos.set_yticklabels([])
    axDos.set_xticklabels([])
    
    axDos.minorticks_on()
    axDos.tick_params(which='both', labelsize='large')
    plt.legend()
    plt.savefig('fatband.png', dpi=720)
    plt.show()

#bandplot()
#fatband()