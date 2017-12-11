# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:46:59 2017

@author: Lorenzo
"""

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
import os
import io
import unicodedata


dir = os.getcwd();


fh = open("fulldata.txt","w",encoding = "ansi") 
id = 0
for file in sorted(os.listdir(dir)):
    if file.endswith(".txt") and file != "fulldata.txt":
        with io.open(file, encoding="ansi") as f:
            res = f.read()
            res_decoded = unicodedata.normalize('NFC', res)
            fh.write(res_decoded)
        fh.write("\n")

#Reading the files