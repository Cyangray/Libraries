#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:15:13 2020

@author: francesco, updated December 28 2022

Library with useful functions to read data from TALYS and translating astrophysical
quantities
"""

import numpy as np
from dicts_and_consts import k_B, c_vacuum, N_A, ompdict
from utils import Z2Name, Name2Z, Name2ZandA, search_string_in_file

def findpath(nucleus, A, ldmodel, massmodel, strength, omp, ds_location):
    Xx = Z2Name(nucleus)
    Z = Name2Z(nucleus)
    zero = ''
    if Z < 100:
        zero = '0'
    filepath = ds_location + 'TALYS_dataset/' + zero + str(Z) + Xx + '/' + str(A) + Xx + '/' + str(ldmodel) + '-' + str(massmodel) + '-' + str(strength) + '-' + ompdict[omp] + '/'
    return filepath

def find_isotope_path(nucleus, A, ds_location):
    Xx = Z2Name(nucleus)
    Z = Name2Z(nucleus)
    zero = ''
    if Z < 100:
        zero = '0'
    filepath = ds_location + 'TALYS_dataset/' + zero + str(Z) + Xx + '/' + str(A) + Xx + '/'
    return filepath

def readstrength(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    #read the strength function from the gsf.txt mashup file
    filepath = find_isotope_path(nucleus, A-1, ds_location) + 'gsf.txt'
    rowsskip = 83* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    return np.loadtxt(filepath, skiprows = (rowsskip + 2), max_rows = 81)

def readstrength_path(path):
    #read the strength function from the output file
    #if path to output file is not given, it will be inferred from the simulation parameters
    rowsskip = search_string_in_file(path, 'f(E1)') + 2
    return np.loadtxt(path, skiprows = rowsskip, max_rows = 81)

def readstrength_legacy(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    filepath = findpath(nucleus, A-1, ldmodel, massmodel, strength, omp, ds_location) + 'output.txt'
    return readstrength_path(filepath)

def readldmodel_table(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    #read the strength function from the gsf.txt mashup file
    filepath = find_isotope_path(nucleus, A, ds_location) + 'nld_table.txt'
    rowsskip = 57* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    return np.loadtxt(filepath, skiprows = (rowsskip + 2), max_rows = 55)

def readldmodel(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    table = readldmodel_table(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/')
    return np.c_[table[:,0], table[:,3]]

def readldmodel_path(path):
    '''
    reads the nld model from the output file of TALYS, given its path
    '''
    
    max_rows = 55

    Parity = search_string_in_file(path, 'Positive parity')
    if Parity:
        posneg = True
        rowsskip = Parity + 2
        rowsskip2 = search_string_in_file(path, 'Negative parity') + 2
    else:
        posneg = False
        rowsskip = search_string_in_file(path, '    Ex     a    sigma') + 2
    
    if posneg:
        pos_par = np.loadtxt(path, skiprows = rowsskip, max_rows = max_rows)
        neg_par = np.loadtxt(path, skiprows = rowsskip2, max_rows = max_rows)
        out_matrix1 = np.zeros(pos_par.shape)
        out_matrix1[:,0] = pos_par[:,0]
        out_matrix1[:,1:] = np.add(pos_par[:,1:], neg_par[:,1:])
    else:
        out_matrix1 = np.loadtxt(path, skiprows = rowsskip, max_rows = max_rows)
    
    out_matrix2 = out_matrix1
    if out_matrix1.shape[1] == 11:
        out_matrix2 = np.zeros((out_matrix1.shape[0], out_matrix1.shape[1]+2))
        out_matrix2[:,0] = out_matrix1[:,0]
        out_matrix2[:,3:] = out_matrix1[:,1:]
        
    return out_matrix2

def readldmodel_legacy(nucleus, A, ldmodel, massmodel, strength, omp, ds_location):
    '''
    reads the nld model from the output file of TALYS.
    '''
    filepath = findpath(nucleus, A-1, ldmodel, massmodel, strength, omp, ds_location = ds_location) + 'output.txt'
    return readldmodel_path(filepath)

def readastro(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    '''
    reads the astrorate from the ncrates.txt mashup file
    '''
    filepath = find_isotope_path(nucleus, A, ds_location) + 'ncrates.txt' 
    rowsskip = 33* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    ncrates = np.loadtxt(filepath, skiprows = (rowsskip + 3), max_rows = 30)
    return np.delete(ncrates, [0,1,2,3], 0) #delete first four rows

def readastro_path(path):
    '''
    reads the astrorate from file
    '''
    filepath = path
    ncrates = np.loadtxt(filepath)
    return np.delete(ncrates, [0,1,2,3], 0) #delete first four rows

def readastro_legacy(nucleus, A, ldmodel, massmodel, strength, omp, ds_location):
    filepath = findpath(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = ds_location) + 'astrorate.g'
    return readastro_path(filepath)

def readreaclib(nucleus, A, reaclib_file = 'reaclib'):
    Xx = Z2Name(nucleus)
    nucleus1 = Xx.lower() + str(A)
    nucleus1 = (5 - len(nucleus1))*' ' + nucleus1
    nucleus2 = Xx.lower() + str(A + 1)
    nucleus2 = (5 - len(nucleus2))*' ' + nucleus2
    reaction = 'n' + nucleus1 + nucleus2
    try:
        rowsskip = search_string_in_file(reaclib_file, reaction) + 1
        row1 = np.genfromtxt(reaclib_file, skip_header = rowsskip, max_rows = 1, autostrip = True, delimiter = 13)
        row1 = row1[~np.isnan(row1)]
        row2 = np.genfromtxt(reaclib_file, skip_header = rowsskip + 1, max_rows = 1, autostrip = True, delimiter = 13)
        row2 = row2[~np.isnan(row2)]
        return np.concatenate((row1, row2))
    except:
        print("reaction \"" + reaction + "\" not in " + reaclib_file + "!")
        bad_a = np.zeros(7)
        bad_a[:] = np.NaN
        return bad_a
    
def nonsmoker(a, T9):
    return np.exp(a[0] + a[1]*T9**-1 + a[2]*T9**(-1/3) + a[3]*T9**(1/3) + a[4]*T9 + a[5]*T9**(5/3) + a[6]*np.log(T9))

def rate2MACS(rate,target_mass_in_au,T):
    '''
    Parameters
    ----------
    rate : ncrate, in cm^3 mol^-1 s^-1
    target_mass_in_au : self explanatory
    T : Temperature in GK

    Returns
    -------
    MACS: in mb

    '''
    
    m1 = target_mass_in_au
    mn = 1.008664915
    red_mass = (m1*mn)/(m1 + mn) * 931494.10242 #keV/c^2
    v_T = np.sqrt(2*k_B*T/red_mass)*c_vacuum
    return rate/v_T*1e25/(N_A) #mb

def MACS2rate(MACS,target_mass_in_au,T):
    '''
    Parameters
    ----------
    MACS : MACS, in mb
    target_mass_in_au : self explanatory
    T : Temperature in GK

    Returns
    -------
    rate: in cm^3 s^-1 mol^-1
    '''
    
    m1 = target_mass_in_au
    mn = 1.008664915
    red_mass = (m1*mn)/(m1 + mn) * 931494.10242 #keV/c^2
    v_T = np.sqrt(2*k_B*T/red_mass)*c_vacuum
    return MACS*v_T*1e-25*N_A #mb

def xs2rate(energies,xss,target_mass_in_au,Ts=None, lower_Elim = 0.1):
    '''
    Parameters
    ----------
    energies : array
        Energy array in keV
    xss : array
        cross section array in mb (must correspond to E)
    target_mass_in_au : float
        self explanatory

    Returns
    -------
    array matrix with temperature and n,gamma rate in cm^3/mole*s

    '''
    mb2cm2 = 1e-27 #mb to cm^2
    m1 = target_mass_in_au
    mn = 1.008664915
    red_mass = (m1*mn)/(m1 + mn) * 931494.10242/(c_vacuum*100)**2 #keV s^2/cm^2
    if Ts is None:
        T9extrange = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    else:
        T9extrange = Ts
    rate = []
    for T in T9extrange:
        integrand = [energies[i]*xss[i]*np.exp(-energies[i]/(k_B*T)) for i in range(len(energies[energies>lower_Elim]))]
        integral = np.trapz(integrand,x=energies[energies>lower_Elim])
        curr_rate = (8/(np.pi*red_mass))**(1/2)*(k_B*T)**(-3/2)*integral*N_A*mb2cm2
        rate.append(curr_rate)
    return rate


def xs2MACS(energies,xss,Ts=None,lower_Elim = 0.1):
    '''
    Parameters
    ----------
    energies : array
        Energy array in keV
    xss : array
        cross section array in mb (must correspond to E)
    target_mass_in_au : float
        self explanatory

    Returns
    -------
    array matrix with temperature and MACS in mb

    '''
    
    if Ts is None:
        T9extrange = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    else:
        T9extrange = Ts
    MACS = []
    for T in T9extrange:
        integrand = [energies[i]*xss[i]*np.exp(-energies[i]/(k_B*T)) for i in range(len(energies[energies>lower_Elim]))]
        integral = np.trapz(integrand,x=energies[energies>lower_Elim])
        curr_MACS = 2/(np.sqrt(np.pi)*(k_B*T)**2)*integral
        MACS.append(curr_MACS)
    return MACS 
    
def translate_nuclides(path):
    '''reads the "nuclides" file from SkyNet to a matrix (list of lists) in the
    form [[Z, A], [Z, A], ...], keeping the line order of "nuclides". This can
    then be associated with the isotope abundances output from SkyNet.'''
    
    text = np.genfromtxt(path, dtype = str)
    outmatrix = []
    for line in text:
        if line == 'n':
            Z = 0
            A = 1
        elif line == 'p':
            Z = 1
            A = 1
        elif line == 'd':
            Z = 1
            A = 2
        elif line == 't':
            Z = 1
            A = 3
        else:
            Z, A = Name2ZandA(line)
        outmatrix.append([Z,A])
    return outmatrix