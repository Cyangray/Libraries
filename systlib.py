"""
Created on Tue Oct  5 15:48:24 2021

@author: francesco

Library of useful functions and classes for nuclear data analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from readlib import readastro_path
from convert_talys import gen_nld_table, log_interp1d
from dicts_and_consts import const, sqrt2pi, k_B
from utils import Z2Name

def sigma2f(sigma, E_g):
    return const*sigma/E_g

def SLO_arglist(E, args):
    '''
    Standard Lorentzian, adapted from Kopecky & Uhl (1989) eq. (2.1)
    '''
    E0, Gamma0, sigma0 = args
    funct = const * sigma0 * E * Gamma0**2 / ( (E**2 - E0**2)**2 + E**2 * Gamma0**2 )
    return funct

def GLO_arglist(E, args):
    '''
    General Lorentzian, adapted from Kopecky & Uhl (1989) eq. (2.4)
    '''
    T, E0, Gamma0, sigma0 = args
    Gamma = Gamma0 * (E**2 + 4* np.pi**2 * T**2) / E0**2
    param1 = (E*Gamma)/( (E**2 - E0**2)**2 + E**2 * Gamma**2 )
    param2 = 0.7*Gamma0*4*np.pi**2 *T**2 /E0**5
    funct = const * (param1 + param2)*sigma0*Gamma0
    return funct

def gauss_arglist(E, args):
    '''
    Gaussian function
    '''
    E0, C, sigma = args
    return C*np.exp(-(E-E0)**2/(2*sigma**2))/(sqrt2pi*sigma)

def upbend_draw_arglist(E, args):
    '''
    exponential function, used to model the upbend
    '''
    a_up, C = args
    return C*np.exp(-a_up*E)

def GLO_hybrid_arglist(E, args):
    '''
    Goriely's Hybrid model
    Coded from the fstrength.f subroutine in TALYS
    '''
    T, E0, Gamma0, sigma0 = args
    ggredep = 0.7 * Gamma0 * ( E/E0 + (2*np.pi*T)**2/(E*E0))
    enumerator = ggredep*E
    denominator = (E**2 - E0**2)**2 + E**2 * ggredep * Gamma0
    factor1 = enumerator/denominator
    return const*sigma0*Gamma0*factor1

def PDR_to_MeVbarn(C,sigma,E0):
    '''
    assumes PDR described by Gaussian
    analytical result of 1/const * int_0^\inf dE E C 1/(sqrt(2*pi)) exp(-(E-E0)^2/(2*sigma^2))
    '''
    return (C*(sigma/np.sqrt(2) + E0))/const #mb*MeV

def chisquared(theo,exp,err,DoF=1,method = 'linear',reduced=True):
    if len(theo) == len(exp) and len(exp) == len(err):
        chi2=0
        if method == 'linear':
            for i in range(len(theo)):
                chi2+=((theo[i] - exp[i])/err[i])**2
            if reduced:
                return chi2/(len(theo)-DoF)
            else:
                return chi2
        elif method == 'log':
            for i in range(len(theo)):
                expmax = exp[i] + err[i]/2
                expmin = exp[i] - err[i]/2
                chi2+=(np.log(theo[i]/exp[i])/np.log(expmax/expmin))**2
            if reduced:
                return chi2/(len(theo)-DoF)
            else:
                return chi2
        else:
            print('Method not known')
    else:
        print('Lengths of arrays not matching')

def ToLatex(nucname):
    '''
    function translating a string of the form 'NNNXX' indicating the
    name of a nucleus, into something like '$^{NNN}$XX' for LaTeX rendering
    '''
    nums = ''
    letters = ''
    for char in nucname:
        if char.isnumeric():
            nums += char
        else:
            letters += char
    newstring = '$^{' + nums + '}$' + letters
    return newstring    

def import_ocl(path,a0,a1, fermi=False):
    '''
    import data generated with the oslo method software, and convert it to 
    omething readable
    '''
    raw_matrix = np.loadtxt(path)
    
    if fermi:
        channels = int(raw_matrix.shape[0])
        polished_matrix = np.zeros((channels,2))
    else:
        channels = int(raw_matrix.shape[0]/2)
        polished_matrix = np.zeros((channels,3))
        
    limit = channels    
    for i, el in enumerate(raw_matrix):
        if i<limit:
            polished_matrix[i,0] = a0 + a1*i
            if el == 0.0 or el == 'inf':
                polished_matrix[i,1] = np.nan
            else:
                polished_matrix[i,1] = el
        else:
            if el == 0.0 or el == 'inf':
                polished_matrix[i-channels,2] = np.nan
            else:
                polished_matrix[i-channels,2] = el
    return polished_matrix

def import_ocl_fermi(path,a0,a1):
    '''
    import data generated with the oslo method software, and convert it to 
    something readable
    '''
    return import_ocl(path,a0,a1, fermi=True)

def flat_distr_chi2_fade(upperlim, lowerlim, sigma, value):
    '''
    Calculates the chi2 score for a distribution that is max and flat between
    lowerlim and upperlim, and fading as a normal distribution with standard
    deviation like sigma, outside these values
    '''
    if isinstance(sigma,list) or isinstance(sigma,tuple):
        sigmadown = sigma[0]
        sigmaup = sigma[1]
    else:
        sigmaup = sigmadown = sigma
    
    if value < lowerlim:
        return ((lowerlim - value)/sigmadown)**2
    elif value <= upperlim:
        return 0
    elif value > upperlim:
        return ((upperlim - value)/sigmaup)**2

class gsf:
    '''
    Class for reading strength functions
    '''
    
    def __init__(self, path, label='', energycol = 0, xscol = 1, errcol = None, is_sigma = True, a0 = 0, a1 = 0, is_ocl = False, E_unit='MeV', xs_units='mb'):
        
        self.path = path
        self.label = label
        
        if is_ocl:
            self.rawmat = import_ocl(path,a0,a1)
            is_sigma = False
            errcol = 2
        else:
            self.rawmat = np.loadtxt(path)
        
        if E_unit=='eV':
            self.energies = self.x = self.rawmat[:,energycol]*1e-6
        else:
            self.energies = self.x = self.rawmat[:,energycol]
        
        if xs_units == 'b':
            b_factor = 1e3
        else:
            b_factor = 1
            
        if is_sigma:
            self.y = np.array([sigma2f(self.rawmat[i,xscol]*b_factor, self.energies[i]) for i in range(len(self.energies))])
        else:
            self.y = self.rawmat[:,xscol]
        
        if isinstance(errcol, int):
            self.error = True
            if is_sigma:
                self.yerr = np.array([sigma2f(self.rawmat[i,errcol]*b_factor, self.energies[i]) for i in range(len(self.energies))])
            else:
                self.yerr = self.rawmat[:,errcol]
            self.yerrplot = self.yerr
        elif isinstance(errcol, tuple) or isinstance(errcol, list):
            self.error = True
            if is_sigma:
                self.yerrdown = np.array([sigma2f(self.rawmat[i,errcol[0]]*b_factor, self.energies[i]) for i in range(len(self.energies))])
                self.yerrup = np.array([sigma2f(self.rawmat[i,errcol[1]]*b_factor, self.energies[i]) for i in range(len(self.energies))])
            else:
                self.yerrdown = self.rawmat[:,errcol[0]]
                self.yerrup = self.rawmat[:,errcol[1]]
            self.yerr = self.yerrup + self.yerrdown
            self.yerrplot = [self.yerrdown, self.yerrup]
        else:
            self.error = False
            
    def clean_nans(self):
        clean_E = []
        clean_y = []
        clean_yerr = []
        for E, y, yerr in zip(self.energies, self.y, self.yerr):
            if not math.isnan(y):
                clean_E.append(E)
                clean_y.append(y)
                clean_yerr.append(yerr)
        self.energies = self.x = np.array(clean_E)
        self.y = np.array(clean_y)
        self.yerr = np.array(clean_yerr)
        self.yerrplot = self.yerr
    
    def delete_point(self, position):
        self.y = np.delete(self.y, position)
        self.energies = self.x = np.delete(self.energies, position)
        self.yerr = np.delete(self.yerr, position)
        self.yerrplot = np.delete(self.yerrplot, position)
        
    def plot(self, ax = 0, alpha = 1, ploterrors = True, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.label
            
        if ax:
            if self.error and ploterrors:
                ax.errorbar(self.energies, self.y, yerr = self.yerrplot, **kwargs)
            else:
                ax.plot(self.energies, self.y, **kwargs)
        else:
            if self.error and ploterrors:
                plt.errorbar(self.energies, self.y, yerr = self.yerrplot, **kwargs)
            else:
                plt.plot(self.energies, self.y, **kwargs)

class xs(gsf):
    def __init__(self, path, A, Z, level = None):
        
        A_string = '0'*(3-len(str(A))) + str(A)
        Z_string = '0'*(3-len(str(Z))) + str(Z)
        if level == None:
            extension = 'tot'
        else:
            extension = 'L' + '0'*(2-len(str(level))) + str(level)
            
        name = 'rp' + Z_string + A_string + '.' + extension
        
        gsf.__init__(self,path = path + name, is_sigma = False)
        self.A = A
        self.Z = Z
        self.level = level

class nld:
    '''
    Class for reading level densities
    '''
    def __init__(self, path, label='', energycol = 0, ldcol = 1, errcol = None, a0 = 0, a1 = 0, is_ocl = False, E_unit = 'MeV'):
        
        self.path = path
        self.label = label
        
        if is_ocl:
            self.rawmat = import_ocl(path,a0,a1)
            errcol = 2
        else:
            self.rawmat = np.loadtxt(path)
        self.energies = self.x = self.rawmat[:,energycol]
        self.y = self.rawmat[:,ldcol]
        
        if E_unit=='eV':
            self.energies = self.x = self.rawmat[:,energycol]*1e-6
        else:
            self.energies = self.x = self.rawmat[:,energycol]
        
        if isinstance(errcol, int):
            self.yerr = self.rawmat[:,errcol]
            self.error = True
            self.yerrplot = self.yerr
        elif isinstance(errcol, tuple) or isinstance(errcol, list):
            self.error = True
            self.yerrdown = self.rawmat[:,errcol[0]]
            self.yerrup = self.rawmat[:,errcol[1]]
            self.yerr = self.yerrup + self.yerrdown
            self.yerrplot = [self.yerrdown, self.yerrup]
        else:
            self.error = False
    
    def clean_nans(self):
        clean_E = []
        clean_y = []
        clean_yerr = []
        for E, y, yerr in zip(self.energies, self.y, self.yerr):
            if not math.isnan(y):
                clean_E.append(E)
                clean_y.append(y)
                clean_yerr.append(yerr)
        self.energies = self.x = np.array(clean_E)
        self.y = np.array(clean_y)
        self.yerr = np.array(clean_yerr)
        
    def delete_point(self, position):
        self.y = np.delete(self.y, position)
        self.energies = self.x = np.delete(self.energies, position)
        self.yerr = np.delete(self.yerr, position)    
    
    def plot(self, ax = 0, alpha = 1, ploterrors = True, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.label
            
        if ax:
            if self.error and ploterrors:
                ax.errorbar(self.energies, self.y, yerr = self.yerrplot, **kwargs)
            else:
                ax.plot(self.energies, self.y, **kwargs)
        else:
            if self.error and ploterrors:
                plt.errorbar(self.energies, self.y, yerr = self.yerrplot, **kwargs)
            else:
                plt.plot(self.energies, self.y, **kwargs)

class astrorate:
    def __init__(self, path=None, label=''):
        if path:
            self.path = path
            self.ncrate_mat = readastro_path(path)
            self.T = self.x = self.ncrate_mat[:,0]
            self.ncrate = self.y = self.ncrate_mat[:,1]
            self.MACS = self.ncrate_mat[:,2]
        self.label = label
    
    def plot(self, ycol = 'ncrate', ax = 0, alpha = 1, color = '', style = 'o'):
        if ycol=='ncrate':
            y = self.ncrate
        elif ycol == 'MACS':
            y = self.MACS
        if ax:
            ax.plot(self.T, y, color+style, alpha = alpha, label = self.label)
        else:
            plt.plot(self.T, y, color+style, alpha = alpha, label = self.label)


def load_known_gsf(A,Z,lab='', author = '', nature = 'E1'):
    '''
    function to load the gsf of a nucleus
    TODO: get path as input and override the hard-coded one

    Parameters
    ----------
    A : int
        mass number
    Z : int,str
        element number, or element name
    lab : str, optional
        'oslo' if data from Oslo, 'darmstadt' if data from Darmstadt. 'o' and 'd' also work. The default is ''.
    nature : str, optional
        'E1' default, M1 can be chosen if lab = Darmstadt

    Returns
    -------
    gsf object
        gsf object containing the loaded data

    '''
    Oslo = False
    Darmstadt = False
    nucleus = str(int(A)) + Z2Name(Z)
    if (lab=='oslo') or (lab=='Oslo') or (lab=='o') or (lab=='ocl') or (lab=='OCL'):
        nucleus = nucleus + '_o'
        Oslo = True
    elif (lab=='darmstadt') or (lab=='Darmstadt') or (lab=='d'):
        nucleus = nucleus +'_d'
        Darmstadt = True
        if nature == 'M1':
            nucleus = nucleus + '_M1'
    
    if nucleus == '129I':
        return gsf('129I-gn.txt', label = 'I129', energycol = 1, xscol = 0)
    elif (Z2Name(Z) == 'Sn') and (A in (112,114,116,118,120,124)) and Darmstadt and nature=='E1':
        return gsf('data/nuclear/Tin/Darmstadt/' + str(int(A)) + 'Sn_Total_GSF_Darmstadt.dat', label = nucleus, energycol = 0, xscol = 1, errcol = 2, is_sigma = False)
    elif (Z2Name(Z) == 'Sn') and (A in (112,114,116,118,120,124)) and Darmstadt and nature=='M1':
        return gsf('data/nuclear/Tin/Darmstadt/' + str(int(A)) + 'Sn_dBM1dE_Darmstadt.dat', label = nucleus, energycol = 0, xscol = 1, errcol = 2, is_sigma = True)
    elif (Z2Name(Z) == 'Sn') and (A in (120,124)) and Oslo:
        return gsf('data/nuclear/Tin/Oslo/' + str(int(A)) + 'Sn_GSF.txt', label = 'Sn' + str(int(A)) + '_o', energycol = 0, xscol = 1, errcol = [3,2], is_sigma = False)
    elif (Z2Name(Z) == 'Ho') and (A == 165) and (author=='nguyen'):
        return gsf('data/nuclear/Ho/165Ho-n-nguyen.txt', label = 'Ho165_nguyen', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='eV', xs_units = 'b')
    elif (Z2Name(Z) == 'Ho') and (A == 165) and (author=='berman'):
        return gsf('data/nuclear/Ho/165Ho-np-berman.txt', label = 'Ho165_berman', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='eV', xs_units = 'b')
    elif (Z2Name(Z) == 'Ho') and (A == 165) and (author=='bergere'):
        return gsf('data/nuclear/Ho/165Ho-np-bergere.txt', label = 'Ho165_bergere', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='eV', xs_units = 'b')
    elif (Z2Name(Z) == 'Ho') and (A == 165) and (author=='varlamov'):
        return gsf('data/nuclear/Ho/165Ho-n-varlamov.txt', label = 'Ho165_varlamov', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='MeV', xs_units = 'mb')
    elif (Z2Name(Z) == 'Ho') and (A == 165) and (author=='thiep'):
        return gsf('data/nuclear/Ho/165Ho-n-thiep.txt', label = 'Ho165_thiep', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='eV', xs_units = 'b')
    elif (Z2Name(Z) == 'Ho') and (A == 166) and (nature=='M1'):
        return gsf('data/nuclear/Ho/fM1_exp_067_166_arc.dat', label = 'Ho166_M1', energycol = 0, xscol = 2, errcol = 3, is_sigma = False)
    elif (Z2Name(Z) == 'Ho') and (A == 166) and (nature=='E1'):
        return gsf('data/nuclear/Ho/fE1_exp_067_166_arc.dat', label = 'Ho166_E1', energycol = 0, xscol = 2, errcol = 3, is_sigma = False)
    elif (Z2Name(Z) == 'Dy') and (A == 163) and (author=='renstroem'):
        return gsf('data/nuclear/Dy/gsf_163dy_big.txt', label = 'Dy163_renstroem', energycol = 2, xscol = 0, errcol = 1, is_sigma = True, E_unit='eV', xs_units = 'b')
    elif (Z2Name(Z) == 'Dy') and (A == 164) and (author=='renstroem'):
        return gsf('data/nuclear/Dy/gsf_164dy_3he_3he_164dy.txt', label = 'Dy164_renstroem', energycol = 0, xscol = 1, errcol = 2, is_sigma = False,)
    
    elif (Z2Name(Z) == 'Te') and (A == 128):
        Te128_n = gsf('data/nuclear/128Te-gn3.txt', label = 'Te128_n', energycol = 3, xscol = 0, errcol = 1)
        Te128_2n = gsf('data/nuclear/128Te-gn3-2n.txt', label = 'Te128_2n', energycol = 3, xscol = 0, errcol = 1)
        Te128 = gsf('data/nuclear/128Te-gn3-2n.txt', label='Te128', energycol = 3, xscol = 0, errcol = 1)
        #unite Te128_n and Te128_2n into Te128
        for i, energy in enumerate(Te128_2n.energies):
            index = np.where(Te128_n.energies == energy)[0]
            if len(index) == 0:
                Te128.energies[i] = np.nan
                Te128.y[i] = np.nan
                Te128.yerr[i] = np.nan
            else:
                Te128.energies[i] = energy
                Te128.y[i] += Te128_n.y[index[0]]
                Te128.yerr[i] += Te128_n.yerr[index[0]]
        Te128.energies = Te128.energies[~np.isnan(Te128.energies)]
        Te128.y = Te128.y[~np.isnan(Te128.y)]
        Te128.yerr = Te128.yerr[~np.isnan(Te128.yerr)]
        
        #for E<Emin for the new set, put Te128_n
        Te128_Emin = min(Te128.energies)
        lowenergy_E = []
        lowenergy_err = []
        lowenergy_gsf = []
        for i, energy in enumerate(Te128_n.energies):
            if energy < Te128_Emin:
                lowenergy_E.append(energy)
                lowenergy_err.append(Te128_n.yerr[i])
                lowenergy_gsf.append(Te128_n.y[i])
            else:
                break
        
        Te128.energies = np.delete(np.concatenate((np.array(lowenergy_E),Te128.energies)), [0,1,2], 0) #delete first three rows
        Te128.y = np.delete(np.concatenate((np.array(lowenergy_gsf), Te128.y)), [0,1,2], 0) #delete first three rows
        Te128.yerr = np.delete(np.concatenate((np.array(lowenergy_err),Te128.yerr)), [0,1,2], 0) #delete first three rows
        Te128.yerrplot = Te128.yerr
        return Te128
        
def import_Anorm_alpha(path):
    string = np.genfromtxt(path)
    Anorm, alpha = [x for x in string if np.isnan(x) == False]
    return Anorm, alpha
    
def import_Bnorm(path):
    arr = np.loadtxt(path, skiprows = 3)
    return arr.item()
    
def import_T(path):
    #string = np.genfromtxt(path, skip_header=4,)
    string = np.loadtxt(path,skiprows = 4,max_rows=1)
    T = string[1]
    return T

def rho2D(rho, target_spin, spin_cutoff):
    '''
    calculate D from rho at Bn (Larsen et al. 2011, eq. (20))
    Takes as input rho as 1/MeV, and gives output D as eV
    target_spin is the spin of the target nucleus
    spin_cutoff is self-explanatory - calculate with robin
    '''
    factor = 2*spin_cutoff**2/((target_spin + 1)*np.exp(-(target_spin + 1)**2/(2*spin_cutoff**2)) + target_spin*np.exp(-target_spin**2/(2*spin_cutoff**2)))
    D0 = factor/rho
    return D0*1e6

def D2rho(D0, target_spin, spin_cutoff):
    '''
    calculate D from rho at Bn (Larsen et al. 2011, eq. (20))
    Takes as input rho as 1/MeV, and gives output D as eV
    target_spin is the spin of the target nucleus
    spin_cutoff is self-explanatory - calculate with robin
    '''
    factor = 2*spin_cutoff**2/((target_spin + 1)*np.exp(-(target_spin + 1)**2/(2*spin_cutoff**2)) + target_spin*np.exp(-target_spin**2/(2*spin_cutoff**2)))
    rho = factor/(D0*1e-6)
    return rho

def drho(target_spin, sig, dsig, D0, dD0, rho = None):
    '''
    Calculate the uncertainty in rho, given the input parameters. sig and dsig
    are the spin cutoff parameter and its uncertainty, respectively. Code taken
    from D2rho in the oslo_method_software package.
    '''
    
    alpha = 2*sig**2
    dalpha = 4*sig*dsig
    if target_spin == 0:
        y1a = (target_spin+1.0)*np.exp(-(target_spin+1.0)**2/alpha)
        y1b = (target_spin+1.)**2*y1a
        z1  = y1b
        z2  = y1a
    else:
        y1a = target_spin*np.exp(-target_spin**2/alpha)
        y2a = (target_spin+1.)*np.exp(-(target_spin+1.)**2/alpha)
        y1b = target_spin**2*y1a
        y2b = (target_spin+1.)**2*y2a
        z1  = y1b+y2b
        z2  = y1a+y2a
    u1 = dD0/D0
    u2 = dalpha/alpha
    u3 = 1-z1/(alpha*z2)
    if rho == None:
        rho = D2rho(D0, target_spin, sig)
    return rho*np.sqrt(u1**2 + u2**2*u3**2)

def make_E1_M1_files_core(gsf, A, Z, M1, target_folder, high_energy_interp):
    '''
    Function that takes the energies and the values of gsf and writes two tables 
    for both E1 and M1 ready to be taken as input by TALYS.
    '''
    
    if high_energy_interp is not None:
        gsf = np.vstack((gsf,high_energy_interp))
    
    gsf_folder_path = ''
    if target_folder is not None:
        if target_folder != '':
            if target_folder[-1] != '/':
                target_folder = target_folder + '/'
        gsf_folder_path = target_folder
    
    fn_gsf_outE1 = gsf_folder_path + "gsfE1.dat"
    fn_gsf_outM1 = gsf_folder_path + "gsfM1.dat"
    
    # The file is/should be writen in [MeV] [MeV^-3] [MeV^-3]
    if gsf[0, 0] == 0:
        gsf = gsf[1:, :]
    Egsf = gsf[:, 0]
    
    if isinstance(M1,float):
        method = 'frac'
    elif isinstance(M1, list):
        if len(M1) == 3:
            method = 'SLO'
        elif len(M1) == 6:
            method = 'SLO2'
    
    if method == 'frac':
        gsfE1 = gsf[:, 1]*(1-M1)
        gsfM1 = gsf[:, 1]*M1
    elif method =='SLO':
        M1_vals = SLO_arglist(gsf[:,0], M1[:3])
        gsfE1 = gsf[:,1] - M1_vals
        gsfM1 = M1_vals
    elif method =='SLO2':
        M1_vals1 = SLO_arglist(gsf[:,0], M1[:3])
        M1_vals2 = SLO_arglist(gsf[:,0], M1[3:])
        M1_vals = M1_vals1 + M1_vals2
        gsfE1 = gsf[:,1] - M1_vals
        gsfM1 = M1_vals

    # REMEMBER that the TALYS functions are given in mb/MeV (Goriely's tables)
    # so we must convert it (simple factor)
    factor_from_mb = 8.6737E-08   # const. factor in mb^(-1) MeV^(-2)
    
    fE1 = log_interp1d(Egsf, gsfE1, fill_value="extrapolate")
    fM1 = log_interp1d(Egsf, gsfM1, fill_value="extrapolate")
    
    Egsf_out = np.arange(0.1, 30.1, 0.1)
    
    header = f" Z=  {Z} A=  {A}\n" + "  U[MeV]  fE1[mb/MeV]"
    # gsfE1 /= factor_from_mb
    np.savetxt(fn_gsf_outE1, np.c_[Egsf_out, fE1(Egsf_out)/factor_from_mb],
               fmt="%9.3f%12.3E", header=header)
    # gsfM1 /= factor_from_mb
    np.savetxt(fn_gsf_outM1, np.c_[Egsf_out, fM1(Egsf_out)/factor_from_mb],
               fmt="%9.3f%12.3E", header=header)
    return Egsf_out, fE1(Egsf_out)/factor_from_mb, fM1(Egsf_out)/factor_from_mb



def make_E1_M1_files_simple(energies, values, A, Z, M1 = 0.1, target_folder = None, high_energy_interp=None, delete_points = None):
    '''
    Function that takes the energies and the values of gsf and writes two tables 
    for both E1 and M1 ready to be taken as input by TALYS.
    '''
    gsf = np.c_[energies,values]
    if delete_points is not None:
        gsf = np.delete(gsf, delete_points, 0)
    return make_E1_M1_files_core(gsf, A, Z, M1, target_folder, high_energy_interp)

def make_E1_M1_files(gsf_folder_path, A, Z, a0, a1, M1 = 0.1, filename = 'strength.nrm', target_folder = None, high_energy_interp=None, delete_points = None):
    '''
    Function that takes the path of a Oslo Method generated gsf with energy in the first column and 
    the gsf in the second, and writes two tables for both E1 and M1 ready to be taken
    as input by TALYS.
    '''
    
    # read/write gsf files
    if gsf_folder_path != '':
        if gsf_folder_path[-1] != '/':
            gsf_folder_path = gsf_folder_path + '/'
        gsf_path = gsf_folder_path + filename
    else:
        gsf_path = filename
        
    gsf = import_ocl(gsf_path, a0, a1)
    gsf = gsf[~np.isnan(gsf).any(axis=1)]
    if delete_points is not None:
        gsf = np.delete(gsf, delete_points, 0)
    gsf = gsf[:,:-1]
    return make_E1_M1_files_core(gsf, A, Z, M1, target_folder, high_energy_interp)
    
def make_TALYS_tab_file(talys_nld_path, ocl_nld_path, A, Z):
    '''
    Function that incorporates the talys_nld_cnt.txt produced by counting, into
    the big Zz.tab file from TALYS. The code overwrites the Zz.tab file, so be careful
    '''
    newfile_content = ''
    if Z < 10:
        Zstring = '  ' + str(Z)
    elif Z < 100:
        Zstring = ' ' + str(Z)
    else:
        Zstring = str(Z)
        
    if A < 10:
        Astring = '  ' + str(A)
    elif A < 100:
        Astring = ' ' + str(A)
    else:
        Astring = str(A)
        
    isotope_strip = 'Z='+ Zstring +' A=' + Astring
    isotopeline = 100000
    with open(talys_nld_path, 'r') as read_obj:
        with open(ocl_nld_path, 'r') as ocl_nld_f:
            ocl_nld = ocl_nld_f.readlines()
            for n, line in enumerate(read_obj):
                stripped_line = line
                new_line = stripped_line
                if isotope_strip in line:
                    isotopeline = n
                if n >= isotopeline + 3:
                    if (n - (isotopeline + 3)) < len(ocl_nld):
                        new_line = ocl_nld[n - (isotopeline + 3)]
                newfile_content += new_line
    
    with open(talys_nld_path, 'w') as write_obj:
        write_obj.write(newfile_content)

def make_TALYS_tab_files_linear_calc(nld_obj, Sn, A, Z, target_up = None, target_down = None):
    '''
    Function that creates two new .tab files by scaling up and down the experimental
    values of the nld according to the statistical errors.
    '''
    
    ocl_nld_path = nld_obj.path[:-10] + 'talys_nld_cnt.txt'
    talys_nld_txt = np.loadtxt(ocl_nld_path)
    nld_obj.clean_nans()
    rel_errs = nld_obj.yerr/nld_obj.y
    max_Ex = nld_obj.energies[-1]
    Sn_rel_err = nld_obj.drho/nld_obj.rho
    tab_energies = talys_nld_txt[:,0]
    rel_errs_tab = np.zeros_like(tab_energies)
    talys_nld_txt_up = talys_nld_txt.copy()
    talys_nld_txt_down = talys_nld_txt.copy()
    for i, Ex in enumerate(tab_energies):
        if Ex <= max_Ex:
            #for energies less than Ex_max: interpolate, find relative error
            rel_errs_tab[i] = np.interp(Ex, nld_obj.energies, rel_errs)
        elif max_Ex < Ex <= Sn:
            #for energies between Ex_max and Sn, relative error interpolate between Ex_max and Sn
            rel_errs_tab[i] = np.interp(Ex, [nld_obj.energies[-1], Sn], [rel_errs[-1], Sn_rel_err])
        elif Ex > Sn:
            #for energies above Sn: relative error as by Sn
            rel_errs_tab[i] = Sn_rel_err
        talys_nld_txt_up[i, 2:] = talys_nld_txt[i, 2:]*(1.+rel_errs_tab[i])
        talys_nld_txt_down[i, 2:] = talys_nld_txt[i, 2:]*(1.-rel_errs_tab[i])
        
    fmt = "%7.2f %6.3f %9.2E %8.2E %8.2E" + 30*" %8.2E"
    np.savetxt('talys_nld_txt_up.txt', talys_nld_txt_up, fmt = fmt)
    np.savetxt('talys_nld_txt_down.txt', talys_nld_txt_down, fmt = fmt)
    
    if target_up is None:
        target_up = Z2Name(Z) + '_up.tab'
    if target_down is None:
        target_down = Z2Name(Z) + '_down.tab'
    
    make_TALYS_tab_file(target_up, 'talys_nld_txt_up.txt', A, Z)
    make_TALYS_tab_file(target_down, 'talys_nld_txt_down.txt', A, Z)

def make_TALYS_tab_file_from_vals(A, Z, Estop, nld_energies, nld_vals, nld_table_target_path, talys_nld_path, spinpars, spinmodel = 'EB05'):#(talys_nld_path, ocl_nld_path, A, Z):
    
    '''
    Function that incorporates the talys_nld_cnt.txt produced by counting, into
    the big Zz.tab file from TALYS. The code overwrites the Zz.tab file, so be careful
    '''

    # load/write nld
    fn_nld_out = nld_table_target_path
    
    # If you comment out extrapolation below, it will do a log-linear
    # extrapolation of the last two points. This is probably not what you want.
    # fnld = log_interp1d(nld[:, 0], nld[:, 1], fill_value="extrapolate")
    fnld = log_interp1d(nld_energies, nld_vals, fill_value = 'extrapolate')

    # print(f"Below {nld[0, 0]} the nld is just an extrapolation
    #       "Best will be to use discrete levels in talys below that")
    table = gen_nld_table(fnld=fnld, Estop=Estop, model=spinmodel, spinpars=spinpars, A=A)
    fmt = "%7.2f %6.3f %9.2E %8.2E %8.2E" + 30*" %8.2E"
    if A % 2 == 1:
        header = "U[MeV]  T[MeV]  NCUMUL   RHOOBS   RHOTOT     J=1/2    J=3/2    J=5/2    J=7/2    J=9/2    J=11/2   J=13/2   J=15/2   J=17/2   J=19/2   J=21/2   J=23/2   J=25/2   J=27/2   J=29/2   J=31/2   J=33/2   J=35/2   J=37/2   J=39/2   J=41/2   J=43/2   J=45/2   J=47/2   J=49/2   J=51/2   J=53/2   J=55/2   J=57/2   J=59/2"
    else:
        header = "U[MeV]  T[MeV]  NCUMUL   RHOOBS   RHOTOT     J=0      J=1      J=2      J=3      J=4      J=5      J=6      J=7      J=8      J=9     J=10     J=11     J=12     J=13     J=14     J=15     J=16     J=17     J=18     J=19     J=20     J=21     J=22     J=23     J=24     J=25     J=26     J=27     J=28     J=29"
    np.savetxt(fn_nld_out, table, fmt=fmt)#, header = header)
    
    newfile_content = ''
    if Z < 10:
        Zstring = '  ' + str(Z)
    elif Z < 100:
        Zstring = ' ' + str(Z)
    else:
        Zstring = str(Z)
        
    if A < 10:
        Astring = '  ' + str(A)
    elif A < 100:
        Astring = ' ' + str(A)
    else:
        Astring = str(A)
        
    isotope_strip = 'Z='+ Zstring +' A=' + Astring
    isotopeline = 100000
    with open(talys_nld_path, 'r') as read_obj:
        with open(nld_table_target_path, 'r') as nld_f:
            new_nld = nld_f.readlines()
            for n, line in enumerate(read_obj):
                stripped_line = line
                new_line = stripped_line
                if isotope_strip in line:
                    isotopeline = n
                    
                if n >= isotopeline + 3:
                    if (n - (isotopeline + 3)) < len(new_nld):
                        new_line = new_nld[n - (isotopeline + 3)]
            
                newfile_content += new_line
    
    #print(newfile_content)
    
    with open(talys_nld_path, 'w') as write_obj:
        write_obj.write(newfile_content)


def gen_nld_table_simple(A, NLDa, Eshift, Estop, nld_energy, nld_vals, nld_table_target_path):
    spinpars = {"mass": A, "NLDa": NLDa, "Eshift": Eshift}

    # load/write nld
    fn_nld_out = nld_table_target_path
    
    # If you comment out extrapolation below, it will do a log-linear
    # extrapolation of the last two points. This is probably not what you want.
    # fnld = log_interp1d(nld[:, 0], nld[:, 1], fill_value="extrapolate")
    fnld = log_interp1d(nld_energy, nld_vals)

    # print(f"Below {nld[0, 0]} the nld is just an extrapolation
    #       "Best will be to use discrete levels in talys below that")
    table = gen_nld_table(fnld=fnld, Estop=Estop, model="EB05", spinpars=spinpars, A=A)
    fmt = "%7.2f %6.3f %9.2E %8.2E %8.2E" + 30*" %8.2E"
    np.savetxt(fn_nld_out, table, fmt=fmt)
    
    return table

def find_chis(vals,chis):
    #function taking as input all chi2-scores associated to the values for a single energy or temperature
    #it finds where the function crosses the chi2+1 line
    whole_mat = np.c_[vals,chis]#np.vstack((vals,chis)).T
    chimin = np.min(chis)
    lower_mat = whole_mat[chis<=(chimin+1)]
    upper_mat = whole_mat[(chis>(chimin+1)) & (chis<(chimin+3))]
    
    min1 = upper_mat[upper_mat[:,0]==np.min(upper_mat[:,0])][0]
    min2 = lower_mat[lower_mat[:,0]==np.min(lower_mat[:,0])][0]
    max1 = lower_mat[lower_mat[:,0]==np.max(lower_mat[:,0])][0]
    max2 = upper_mat[upper_mat[:,0]==np.max(upper_mat[:,0])][0]
    
    # y(x) = A + Bx
    Bmin = (min2[1]-min1[1])/(min2[0]-min1[0])
    Amin = min2[1]-Bmin*min2[0]
    Bmax = (max2[1]-max1[1])/(max2[0]-max1[0])
    Amax = max2[1]-Bmax*max2[0]
    
    # evaluate at Y = chimin + 1
    Y = chimin + 1
    Xmin = (Y-Amin)/Bmin
    Xmax = (Y-Amax)/Bmax
    return [Xmin,Xmax]
    
def find_chis_interp(vals, chis, iterations = 2):
    '''
    New, more precise algorithm than find_chis when this is not good enough. Potentially slower.
    concept: first, make an array of datapoints (e.g. "points") with vals as x and chis as y.
    Then sort these for increasing chi.
    '''
    points = np.c_[vals,chis]
    points = points[np.lexsort((points[:,1],points[:,0]))]
    chimin_index = np.argmin(points[:,1])
    
    vertices_less = points[0]
    vertices_less = np.expand_dims(vertices_less, axis=0)
    
    for i, point in enumerate(points):
        if point[1] < vertices_less[-1,1]:
            vertices_less = np.vstack((vertices_less, point))
        if i==chimin_index: #(just consider the points with vals less than chimin)
            break
    
    vertices_more = points[-1]
    vertices_more = np.expand_dims(vertices_more, axis=0)
    for i, point in enumerate(points[::-1]):
        if i==chimin_index: #(just consider the points with vals more than chimin)
            break
        if point[1] < vertices_more[-1,1]:
            vertices_more = np.vstack((vertices_more, point))

    def delete_points(vertices_input, invert):
        if invert:
            vertices = np.c_[vertices_input[:,0]*-1, vertices_input[:,1]]
        else:
            vertices = vertices_input
        delete_indexes = []
        x0 = vertices[0,0]
        y0 = vertices[0,1]
        for i, point in enumerate(vertices):
            if i == len(vertices)-2:
                break
            elif i == 0:
                pass
            else:
                x1 = vertices[i,0]
                y1 = vertices[i,1]
                x2 = vertices[i+1,0]
                y2 = vertices[i+1,1]
                
                if x2 == x1:
                    delete_indexes.append(i)
                elif x1 == x0:
                    pass
                else:
                    prev_steepness = (y1-y0)/(x1-x0)
                    next_steepness = (y2-y1)/(x2-x1)
                    if next_steepness < prev_steepness:
                        delete_indexes.append(i)
                    else:
                        x0 = vertices[i,0]
                        y0 = vertices[i,1]
        
        return np.delete(vertices_input, delete_indexes, 0)
        
        
    for i in range(iterations):
        vertices_less = delete_points(vertices_less, invert = False)  
        vertices_more = delete_points(vertices_more, invert = True)
    vertices = np.vstack((vertices_less, vertices_more[::-1]))
    chimin = np.min(vertices[:,1])
    for i, vertex in enumerate(vertices):
        if vertex[1] < (chimin+1):
            min1 = vertices[i-1,:]
            min2 = vertices[i,:]
            break
    for i, vertex in reversed(list(enumerate(vertices))):
        if vertex[1] < (chimin+1):
            max1 = vertices[i+1,:]
            max2 = vertices[i,:]
            break
    # y(x) = A + Bx
    Bmin = (min2[1]-min1[1])/(min2[0]-min1[0])
    Amin = min2[1]-Bmin*min2[0]
    Bmax = (max2[1]-max1[1])/(max2[0]-max1[0])
    Amax = max2[1]-Bmax*max2[0]
    # evaluate at Y = chimin + 1
    Y = chimin + 1
    Xmin = (Y-Amin)/Bmin
    Xmax = (Y-Amax)/Bmax
    return [Xmin,Xmax]

def calc_errors_chis_2sigma(lst):
    
    xx = lst[4].x #all energy or temperature vectors are alike. Pick the 5th, but any would do
    val_matrix = np.zeros((xx.size,6))
    for i, x in enumerate(xx):
        chis = []
        vals = []
        row = np.zeros(6)
        counterflag = 0
        for graph in lst:
            if not np.isnan(graph.y[i]):
                chis.append(graph.chi2)
                vals.append(graph.y[i])
                #it may be that all y[i] are nans. Do a check so that the code only saves data if there actually are values to analyze
                counterflag += 1
        if counterflag > 10:
            index_of_best_fit = np.argwhere(chis==np.min(chis))
            if len(index_of_best_fit)>1:
                index_of_best_fit = index_of_best_fit[0,0]
            else:
                index_of_best_fit = index_of_best_fit.item()
            best_fit = vals[index_of_best_fit]
            errmin, errmax = find_chis_interp(vals,chis)
            row[:] = [x, best_fit, -best_fit+2*errmin, errmin, errmax, 2*errmax-best_fit]
        else:
            row[:] = [x, np.nan, np.nan, np.nan, np.nan, np.nan]
        val_matrix[i,:] = row[:]
    return val_matrix

def calc_errors_chis(lst, graphic_function = find_chis):
    xx = lst[4].x #all energy or temperature vectors are alike. Pick the 5th, but any would do
    val_matrix = np.zeros((xx.size,4))
    for i, x in enumerate(xx):
        chis = []
        vals = []
        row = np.zeros(4)
        counterflag = 0
        for graph in lst:
            if not np.isnan(graph.y[i]):
                chis.append(graph.chi2)
                vals.append(graph.y[i])
                #it may be that all y[i] are nans. Do a check so that the code only saves data if there actually are values to analyze
                counterflag += 1
        if counterflag > 10:
            index_of_best_fit = np.argwhere(chis==np.min(chis))
            if len(index_of_best_fit)>1:
                index_of_best_fit = index_of_best_fit[0,0]
            else:
                index_of_best_fit = index_of_best_fit.item()
            best_fit = vals[index_of_best_fit]
            errmin, errmax = graphic_function(vals,chis)
            row[:] = [x, best_fit, errmin, errmax]
        else:
            row[:] = [x, np.nan, np.nan, np.nan]
        val_matrix[i,:] = row[:]
    return val_matrix

def calc_errors_chis_MACS(lst, graphic_function = find_chis_interp):
    xx = lst[4].x*k_B #all energy or temperature vectors are alike. Pick the 5th, but any would do
    val_matrix = np.zeros((xx.size,4))
    for i, x in enumerate(xx):
        chis = []
        vals = []
        row = np.zeros(4)
        counterflag = 0
        for graph in lst:
            if not np.isnan(graph.MACS[i]):
                chis.append(graph.chi2)
                vals.append(graph.MACS[i])
                #it may be that all y[i] are nans. Do a check so that the code only saves data if there actually are values to analyze
                counterflag += 1
        if counterflag > 10:
            index_of_best_fit = np.argwhere(chis==np.min(chis))
            if len(index_of_best_fit)>1:
                index_of_best_fit = index_of_best_fit[0,0]
            else:
                index_of_best_fit = index_of_best_fit.item()
            best_fit = vals[index_of_best_fit]
            errmin, errmax = find_chis_interp(vals,chis)
            row[:] = [x, best_fit, errmin, errmax]
        else:
            row[:] = [x, np.nan, np.nan, np.nan]
        val_matrix[i,:] = row[:]
    return val_matrix