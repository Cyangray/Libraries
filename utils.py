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

def ZandA2Name(A,Z):
    '''translates A and Z into a XxxNn isotope name string'''
    Astr = str(A)
    Zstr = Z2Name(Z)
    return Astr + Zstr