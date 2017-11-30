# Code used for preparing CSV files to be used in neural network
# for CSC 450 research project
#
# Author: Nick Frogley

import sys
import os
import re
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_csv_lines(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()
    contents_split_nl = contents_raw.split("\n")
    return contents_split_nl

def get_id_status_dict(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()

    contents_split_nl = contents_raw.split("\n")

    d = {}

    for the_line in contents_split_nl:
        line_split_comma = the_line.split(",")
        d[line_split_comma[0]] = line_split_comma[10]  

    return d

def get_status_list(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()

    contents_split_nl = contents_raw.split("\n")

    the_list = list()

    print(contents_split_nl[0])
    print(contents_split_nl[1])

    i = 0

    for the_line in contents_split_nl:
        line_split_comma = the_line.split(",")
        if (len(line_split_comma) > 10):
            the_list.append(line_split_comma[10])
            print(str(i) + " : " + line_split_comma[0] + " : " + line_split_comma[10])
            i += 1
    return the_list


status_list = get_status_list("Nick_sample_info.csv")
print(status_list)

expression_lines = get_csv_lines("Nick_expression.csv")

outfile = open("full_expression_new.csv","w")

status_line = ""

for i in range(0, len(status_list)) :
    status_line += status_list[i] + ","

status_line = status_line[0:-1:]

outfile.write(expression_lines[0] + "\n")
outfile.write(status_line + "\n")

for i in range(1, len(expression_lines)) :
    outfile.write(expression_lines[i] + "\n")



outfile.close()
