import tkinter as tk
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import threading
import serial
import serial.tools.list_ports
import seaborn as sns
import random
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.neighbors import KernelDensity
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

import Controller
"""
View controls the GUI that the user sees
"""


class View(tk.Tk):

    """
    Initialise the many variables that are required for user interface
    """
    def __init__(self, controller):

        super().__init__()
        self.controller = controller
        self.geometry('1000x750')

        self.title('CSSE4011 Interface')
        self.main_frame = tk.Frame(self)
        self.main_frame.pack()

        self.thread_plot()

    def draw3DRectangle(self, ax, x1, y1, z1, x2, y2, z2):
        # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
        ax.plot([x1, x2], [y1, y1], [z1, z1], color='b')  # | (up)
        ax.plot([x2, x2], [y1, y2], [z1, z1], color='b')  # -->
        ax.plot([x2, x1], [y2, y2], [z1, z1], color='b')  # | (down)
        ax.plot([x1, x1], [y2, y1], [z1, z1], color='b')  # <--

        ax.plot([x1, x2], [y1, y1], [z2, z2], color='b')  # | (up)
        ax.plot([x2, x2], [y1, y2], [z2, z2], color='b')  # -->
        ax.plot([x2, x1], [y2, y2], [z2, z2], color='b')  # | (down)
        ax.plot([x1, x1], [y2, y1], [z2, z2], color='b')  # <--

        ax.plot([x1, x1], [y1, y1], [z1, z2], color='b')  # | (up)
        ax.plot([x2, x2], [y2, y2], [z1, z2], color='b')  # -->
        ax.plot([x1, x1], [y2, y2], [z1, z2], color='b')  # | (down)
        ax.plot([x2, x2], [y1, y1], [z1, z2], color='b')  # <--

    """
    Creates thread to plot data
    """
    def thread_plot(self):

        while(1):
            separated_clusters = self.controller.separated_clusters
            centre_points = self.controller.cluster_points

            print(separated_clusters)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            #### Do all the plotting
            for cluster in separated_clusters:
                df = cluster.drop(cluster.index[0])

                x = df['X'].values
                y = df['Y'].values
                z = df['Z'].values

                data = np.vstack([x, y, z])
                # print(data)

                xmin, xmax, ymin, ymax, zmin, zmax = np.min(data[0, :]), np.max(data[0, :]), np.min(
                    data[1, :]), np.max(data[1, :]), np.min(data[2, :]), np.max(
                    data[2, :])

                ax.scatter(data[0, :], data[1, :], data[2, :], label="centered data")

                ax.set_xlabel('X')
                ax.set_xlim([-2.5, 2.5])
                ax.set_ylabel('Y')
                ax.set_ylim([0, 5])
                ax.set_zlabel('Z')
                ax.set_zlim([-1.5, 3.5])

                ax.legend()

                self.draw3DRectangle(plt, xmin, ymin, zmin, xmax, ymax, zmax)

            plt.show()
