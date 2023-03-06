#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:15:13 2020

@author: francesco, updated 1st February 2022

Dictionaries for most of the other libraries
"""


from dicts_and_consts import Zdict, isodict, ompdict
from numpy import sqrt, pi

#create useful dictionaries

def Z2Name(Z):
    '''convert Z to corresponding element name, if Z is a number. Otherwise it returns the input string'''
    if isinstance(Z, float):
        Z = int(Z)
    if isinstance(Z, int):
        return isodict[Z]
    else:
        return Z

def Name2Z(Xx):
    '''convert element name to corresponding Z if input is string. Otherwise it returns the input int'''
    if isinstance(Xx, str):
        return Zdict[Xx]
    else:
        return Xx
           
def search_string_in_file(file_name, string_to_search):
    #return the line number where the input string is found
    with open(file_name, 'r') as read_obj:
        for n, line in enumerate(read_obj):
            if string_to_search in line:
                return n
            
def Name2ZandA(inputstr):
    '''
    Translates an input string in the form Xx123 to [Z, A]
    '''
    Xx = ''
    A = ''
    for character in inputstr:
        if character.isalpha():
            Xx += character
        else:
            A += character
    return [Name2Z(Xx.title()), int(A)]

def ZandA2Name(A,Z, invert = False, particle_names = False):
    '''translates A and Z into a XxxNn isotope name string'''
    Astr = str(int(A))
    Zstr = Z2Name(Z)
    
    if Z in [0,1] and particle_names:
        if Z == 0:
            return 'n'
        elif A == 1:
            return 'p'
        elif A == 2:
            return 'd'
        elif A == 3:
            return 't'
        else:
            print('somethings wrong, Z = %d, A = %d'%(Z,A))
            return 0
    
    if invert:
        return Zstr + Astr
    else:
        return Astr + Zstr
    
def lin_interp(x1,y1,x2,y2,x0):
    a = (y2-y1)/(x2-x1)
    b = y2 - a*x2
    return a*x0 + b