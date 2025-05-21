#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/17 10:23
# @Author  : gsunwu@163.com
# @File    : ackley.py
# @Description: function in https://www.sfu.ca/~ssurjano/optimization.html
import numpy as np


def ackley(xx, a=20, b=0.2, c=2 * np.pi):
    ##########################################################################
    #
    # ACKLEY FUNCTION
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
    #
    # INPUTS:
    #
    # xx = [x1, x2, ..., xd]
    # a = constant (optional), with default value 20
    # b = constant (optional), with default value 0.2
    # c = constant (optional), with default value 2*pi
    #
    ##########################################################################

    d = len(xx)

    sum1 = np.sum(np.square(xx))
    sum2 = np.sum(np.cos(c * xx))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    y = term1 + term2 + a + np.exp(1)
    return y

if __name__ == '__main__':
    print(ackley(np.asarray([40,40])))