#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/17 10:27
# @Author  : gsunwu@163.com
# @File    : test.py
# @Description:
import unittest
import logging

log = logging.getLogger(__name__)

from ackley import *
from grlee12 import  *
class TestDatasetLoader(unittest.TestCase):

    def test_plot_ackley(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D



        # Generate grid of points
        x = np.linspace(-32.768, 32.768, 1000)
        y = np.linspace(-32.768, 32.768, 1000)
        X, Y = np.meshgrid(x, y)
        Z = np.array([ackley(np.asarray([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Labels and title
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Ackley Function Value')
        ax.set_title('Ackley Function in 2D')

        plt.show()
    def test_plot_grlee12(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the range for x
        x_values = np.linspace(0.5, 2.5, 400)
        y_values = grlee12(x_values)

        # Plotting the function
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label='GRAMACY & LEE (2012) Function', color='b')
        plt.title('GRAMACY & LEE (2012) Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()