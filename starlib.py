#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:45:17 2022

@author: francesco
"""

import numpy as np
from utils import Name2Z
import os

def import_strengths_idxs():
    strengths_idxs = ['0', 
                      '153B',
                      '153',
                      '0',
                      '128', 
                      '0',
                      '0',
                      '202',
                      '177B',
                      '171B',
                      '153B_202B', #10
                      '0',
                      '0',
                      '0',
                      '153B1',
                      '0',          #15
                      '0',
                      '153B4',
                      '153B3',
                      '153B2',
                      '153B5',      #20
                      '153B6',
                      '153B7',
                      '153B8']
    #strength 10: model 153B up until Bi, then 202 rates
    return strengths_idxs

def strength2name(strength):
    strength_idxs = import_strengths_idxs()
    return strength_idxs[strength]

def read_rates_from_file(ratesfilepath, list_length = 64):
    rates_list = []
    N = 0
    Z = 0
    rate_matrix = np.zeros((95,152,list_length))
    rate_matrix[:] = np.nan
    #print(rate_matrix)
    with open(ratesfilepath, 'r') as ratesfile:
        for n, line in enumerate(ratesfile):
            
            #new reaction
            if line[0] == '!':
                #read new Z and N
                el_name = line[7:9]
                if el_name[1] == ' ':
                    el_name = line[7:8]
                else:
                    el_name = el_name.lower().capitalize()
                
                el_A = line[9:12]
                
                if el_A[0] == ' ':
                    el_A = line[10:12]
                A = int(el_A)
                Z = Name2Z(el_name)
                N = A - Z
            
            #Data line
            if line[2] not in ' T0':
                line_rates = [float(el) for el in line.split()]
                rates_list += line_rates
                
                if len(line.split()) == 4:
                    #save old reaction
                    current_rates = np.array(rates_list)
                    rates_list = []
                    rate_matrix[Z,N,:] = current_rates
                    
    return rate_matrix

def import_fns():
    fns = ['0',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B_00086816',
           '0',
           '0',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta128_00086716',
           '0',
           '0',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta202_00086412',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta177B_00086303',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta171B_00086192',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B_202B_00087565', #10
           '0',
           '0',
           '0',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B1_00087052',
           '0',                                                     #15
           '0',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B4_00086192',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B3_00086489',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B2_00086947',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B5_00086741',     #20
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B6_00086677',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B7_00086587',
           'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta153B8_00086497'
           ]
    return fns

def strength_fn(strength):
    strength_name = strength2name(strength)
    return f'm1.0_FeHm2.5_v3.16i_pm00c_rsi250_ta{strength_name}'

def strength_path(strength):
    return f'../{strength_fn(strength)}'

def rates_path(strength):
    #strength_name = strength2name(strength)
    strength_dir = strength_path(strength)
    for i in os.listdir(strength_dir):
        if i.endswith('.dat'):
            return strength_dir + '/' + i
    #if strength <= 9:
    #    return f'{strength_path(strength)}/vit_mod_ta{strength_name}.dat'
        

def fn_path(strength):
    fns = import_fns()
    return fns[strength]

def read_rates_at_temp(strength_model, temperature_to_compare):
    path = rates_path(strength_model)
    print(path)
    rates = read_rates_from_file(path)
    rates_at_temp = rates[:,:,temperature_to_compare]
    return rates_at_temp























