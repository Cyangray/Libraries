#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:15:13 2020

@author: francesco, updated December 28 2022

Library with useful functions to read data from TALYS and translating astrophysical
quantities
"""

import numpy as np
from dicts_and_consts import k_B, c_vacuum, N_A, ompdict, import_T9
from utils import Z2Name, Name2Z, Name2ZandA, search_string_in_file
import sklearn.linear_model as lm

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

def readxs_path(path):
    #read the cross section from the output file
    return np.loadtxt(path, skiprows = 5)

def readldmodel_table(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    #read the level density from the nld_table.txt mashup file
    filepath = find_isotope_path(nucleus, A, ds_location) + 'nld_table.txt'
    rowsskip = 57* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    return np.loadtxt(filepath, skiprows = (rowsskip + 2), max_rows = 55)

def readldmodel(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    table = readldmodel_table(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = ds_location)
    return np.c_[table[:,0], table[:,3]]

def readldmodel_path(path):
    '''
    reads the nld model from the output file of TALYS, given its path.
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

def readldmodel_legacy(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    '''
    reads the nld model from the output file of TALYS.
    '''
    filepath = findpath(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = ds_location) + 'output.txt'
    table = readldmodel_path(filepath)
    return np.c_[table[:,0], table[:,3]]

def readastro(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    '''
    reads the astrorate from the ncrates.txt mashup file
    '''
    filepath = find_isotope_path(nucleus, A, ds_location) + 'ncrates.txt' 
    rowsskip = 34* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    ncrates = np.loadtxt(filepath, skiprows = (rowsskip + 3), max_rows = 30)
    return np.delete(ncrates, [0,1,2,3], 0) #delete first four rows

def read_Qvalue(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/', dtype = 'float'):
    '''
    read the Q value from the ncrates.txt mashup file
    '''
    filepath = find_isotope_path(nucleus, A, ds_location) + 'ncrates.txt' 
    rowsskip = 34* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    
    if dtype == 'str':
        line = np.genfromtxt(filepath, skip_header = (rowsskip + 33), max_rows = 1, dtype = 'float')
        #print(line)
        if line[0] >= 0.:
            return ' %11.5e'%line[0]
        else:
            return '%12.5e'%line[0]
    else:
        line = np.loadtxt(filepath, skiprows = (rowsskip + 33), max_rows = 1)
        Qvalue = line[0]
        return Qvalue

def read_DZ(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    '''
    read if the rate is calculated using the DZ mass model, from the ncrates.txt file
    returns 0 if not, 1 if yes
    '''
    filepath = find_isotope_path(nucleus, A, ds_location) + 'ncrates.txt' 
    rowsskip = 34* ( ((ldmodel-1)*3*8*2) + ((massmodel-1)*8*2) + ((strength-1)*2) + ((omp-1)) )
    line = np.loadtxt(filepath, skiprows = (rowsskip + 33), max_rows = 1)
    DZ = line[1]
    return int(DZ)

def readastro_path(path):
    '''
    reads the astrorate from file
    '''
    filepath = path
    ncrates = np.loadtxt(filepath)
    return np.delete(ncrates, [0,1,2,3], 0) #delete first four rows

def readastro_legacy(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
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
    
def lnnonsmoker(a, T9):
    return a[0] + a[1]*T9**-1 + a[2]*T9**(-1/3) + a[3]*T9**(1/3) + a[4]*T9 + a[5]*T9**(5/3) + a[6]*np.log(T9)

def DZcalc(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = 'data/'):
    #Find out if the astrorate uses the DZ massformula
    #NB: outdated! new file system doesn't have info on DZ. See if you can retrieve the information from the tar file, and maybe save it in a DZ file?
    filepath = findpath(nucleus, A, ldmodel, massmodel, strength, omp, ds_location) + 'output.txt'
    
    Aname = (3 - len(str(A)))*' ' + str(A)
    string = "TALYS-warning: Duflo-Zuker mass for " + Z2Name(nucleus) + Aname
    with open(filepath, 'r') as read_obj:
        for n, line in enumerate(read_obj):
            if string in line:
                return True
            if n > 25:
                return False

def find_Qvalue(nucleus, A, ldmodel, massmodel, strength, omp, ds_location):
    filepath = findpath(nucleus, A, ldmodel, massmodel, strength, omp, ds_location) + 'output.txt'
    rowsskip = search_string_in_file(filepath, 'Q(n,g):')
    line = np.genfromtxt(filepath, skip_header = rowsskip, max_rows = 1)
    if line[1]>=0:
        Qvalue = ' %11.5e'%line[1]
    else:
        Qvalue = '%12.5e'%line[1]
    return Qvalue
    
def astrofit(nucleus, A, ldmodel, massmodel, strength, omp, plot = False, ds_location = 'data/', reaclib_file = 'reaclib'):
    #fit the talys output to the NONSMOKER parametrization, outputs the new a_i parameters
    
    #data for neutron capture rates from TALYS
    talysrates = readastro(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = ds_location)
    
    #delete rows if rate is 0
    zerolist = []
    for n in range(len(talysrates[:,1])):
        if talysrates[n,1] == 0:
            zerolist.append(n)
    talysrates = np.delete(talysrates, zerolist, 0)
    
    
    
    #Linear fit of the talys rates to the NONSMOKER formula
    #Make the formula linear by taking the logarithm of the nonsmoker formula and the Talys target
    
    #Create design matrix
    X = np.zeros((talysrates.shape[0], 7))
    T9 = talysrates[:,0]
    X[:,0] = 1
    X[:,1] = T9**-1
    X[:,2] = T9**(-1/3)
    X[:,3] = T9**(1/3)
    X[:,4] = T9
    X[:,5] = T9**(5/3)
    X[:,6] = np.log(T9)
    
    #target
    target = np.log(talysrates[:,1])
    
    #Fetch the results for OLS, and ridge regressions (take exponentials to go back to the NONSMOKER original formula)
    clf = lm.LinearRegression(fit_intercept = False).fit(X, target)
    fity = np.exp(clf.predict(X))
    
    if plot:
        #plot (Logarithmic scale)
        if isinstance(nucleus, int):
            nucleusname = Z2Name(nucleus)# isodict[nucleus]
        else:
            nucleusname = nucleus
        #Data for neutron capture rates from REACLIB + NONSMOKER
        a = readreaclib(nucleus, A, reaclib_file = reaclib_file)
        T9range = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        nsrates = nonsmoker(a, T9range)
        '''
        nuclide = str(A) + nucleusname
        plt.figure()
        plt.title('Neutron capture rates for ' + nuclide + ' (logarithmic)')
        plt.loglog(talysrates[:,0], talysrates[:,1],'b-',label='TALYS rates')
        plt.loglog(T9range, nsrates, 'r-', label='NONSMOKER rates')
        plt.loglog(T9, fity, 'b--', label = 'fit')
        plt.xlabel('T9 (GK)')
        plt.ylabel('n capture rate')
        plt.grid()
        plt.legend()
        plt.show()
        '''
    
    return clf.coef_
    
def reaclib_replace(nucleus, A, ldmodel, massmodel, strength, omp, replace = False, input_file = 'reaclib', output_file = 'reaclib_n', ds_location = 'data/'):
    '''writes a new file called reaclib_n with the fitted parameters from TALYS
    with the respective omp and strength, replacing the old NONSMOKER ones.
    Normally it creates a new file with the changes, but if looping this function,
    only the last change will be visible. Thus, if looping through many nuclides, 
    you will have to write on the same file you read, thus: backup the original 
    file, and flag replace = True''' 
    
    #Fit TALYS output rates to NONSMOKER parameters
    a = astrofit(nucleus, A, ldmodel, massmodel, strength, omp, ds_location = ds_location)
    
    #Write the parameters into REACLIB format
    string1 = ''
    string2 = ''
    for i, coeff in enumerate(a):
        if i < 4:
            if coeff>=0:
                string1 += ' %12.6e'%coeff
            else:
                string1 += '%13.6e'%coeff
            if i == 3:
                string1 +='                      '
        else:
            if coeff>=0:
                string2 += ' %12.6e'%coeff
            else:
                string2 += '%13.6e'%coeff
            if i == 6:
                string2 += '                                   '
                
    #Find where the reaction is in the REACLIB, and generate new file
    if isinstance(nucleus, int):
        nucleusname = Z2Name(nucleus)#isodict[nucleus]
    else:
        nucleusname = nucleus
    
    nucleus1 = nucleusname.lower() + str(A)
    nucleus1 = (5 - len(nucleus1))*' ' + nucleus1
    nucleus2 = nucleusname.lower() + str(A + 1)
    nucleus2 = (5 - len(nucleus2))*' ' + nucleus2
    reaction = 'n' + nucleus1 + nucleus2
    
    reactionline = -3
    newfile_content = ''
    with open(input_file, 'r') as read_obj:
        for n, line in enumerate(read_obj):
            stripped_line = line
            new_line = stripped_line
            if reaction in line:
                reactionline = n
                Qvalue = read_Qvalue(nucleus, A, ldmodel, massmodel, strength, omp, ds_location, dtype = 'str')
                #find_Qvalue(nucleus, A, ldmodel, massmodel, strength, omp)
                new_line = '         ' + reaction + '                       ' + str(ldmodel) + str(massmodel) + str(strength) + str(omp) + '     ' + Qvalue + '          ' + '\n'
            elif n == reactionline + 1:
                new_line = string1 + '\n'
            elif n == reactionline + 2:
                new_line = string2 + '\n'
            newfile_content += new_line
        
    if reactionline == -3: #reaction not found in reaclib: add it at the beginning of chapter 4
        newfile_content = ''
        with open(input_file, 'r') as read_obj:
            for n, line in enumerate(read_obj):
                new_line = line
                if n==127580:
                    Qvalue = read_Qvalue(nucleus, A, ldmodel, massmodel, strength, omp, ds_location, dtype = 'str')
                    new_line = '4' + '\n' + '         ' + reaction + '                       ' + str(ldmodel) + str(massmodel) + str(strength) + str(omp) + '     ' + Qvalue + '          ' + '\n' + string1 + '\n' + string2 + '\n' + '4' + '\n'
                newfile_content += new_line
        if replace:
            print("reaction \"" + reaction + "\" added to " + output_file)
        else:
            print("reaction \"" + reaction + "\" added to " + input_file)
    #write the new file in another file (or the same file if replacing)
    if replace:
        writefile = input_file
    else:
        writefile = output_file
    with open(writefile, 'w') as write_obj:
        write_obj.write(newfile_content)

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
        T9extrange = import_T9(extended = True)
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
        T9extrange = import_T9(extended = True)
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
