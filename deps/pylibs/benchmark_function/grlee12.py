#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/17 10:42
# @Author  : gsunwu@163.com
# @File    : grlee12.py
# @Description:
import numpy as np


def grlee12(x):
    ##########################################################################
    # https://www.sfu.ca/~ssurjano/grlee12.html
    #
    # GRAMACY & LEE (2012) FUNCTION
    #
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    #
    # Copyright 2013. Derek Bingham, Simon Fraser University.
    #
    # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    # derivative works, such modified software should be clearly marked.
    # Additionally, this program is free software; you can redistribute it
    # and/or modify it under the terms of the GNU General Public License as
    # published by the Free Software Foundation; version 2.0 of the License.
    # Accordingly, this program is distributed in the hope that it will be
    # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    # General Public License for more details.
    #
    # For function details and reference information, see:
    # http://www.sfu.ca/~ssurjano/
    #
    ##########################################################################

    term1 = np.sin(10 * np.pi * x) / (2 * x)
    term2 = (x - 1) ** 4

    y = term1 + term2
    return y