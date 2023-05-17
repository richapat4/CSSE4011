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

import matplotlib.animation as animation

"""
View controls the GUI that the user sees
"""


class View():

    """
    Initialise the many variables that are required for user interface
    """
    def __init__(self, controller):

        # super().__init__()
        self.controller = controller
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatter = self.ax.scatter([0],[0],[0])
        
        self.ax.set_xlabel('X')
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim([0, 5])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim([-1.5, 3.5])

        # self.geometry('1000x750')

        # self.title('CSSE4011 Interface')
        # self.main_frame = tk.Frame(self)
        # self.main_frame.pack()

       

    def draw3DRectangle(self,x1, y1, z1, x2, y2, z2):
        
        # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
        self.ax.plot([x1, x2], [y1, y1], [z1, z1], color='b')  # | (up)
        self.ax.plot([x2, x2], [y1, y2], [z1, z1], color='b')  # -->
        self.ax.plot([x2, x1], [y2, y2], [z1, z1], color='b')  # | (down)
        self.ax.plot([x1, x1], [y2, y1], [z1, z1], color='b')  # <--

        self.ax.plot([x1, x2], [y1, y1], [z2, z2], color='b')  # | (up)
        self.ax.plot([x2, x2], [y1, y2], [z2, z2], color='b')  # -->
        self.ax.plot([x2, x1], [y2, y2], [z2, z2], color='b')  # | (down)
        self.ax.plot([x1, x1], [y2, y1], [z2, z2], color='b')  # <--

        self.ax.plot([x1, x1], [y1, y1], [z1, z2], color='b')  # | (up)
        self.ax.plot([x2, x2], [y2, y2], [z1, z2], color='b')  # -->
        self.ax.plot([x1, x1], [y2, y2], [z1, z2], color='b')  # | (down)
        self.ax.plot([x2, x2], [y1, y1], [z1, z2], color='b')  # <--

    """
    Creates thread to plot data
    """
    def thread_plot(self,frame):

        # while(1)
        separated_clusters = self.controller.separated_clusters
        centre_points = self.controller.cluster_points

        if separated_clusters is not None and centre_points is not None:
            
            print(separated_clusters)
        
            
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

                self.scatter._offsets3D(data[0, :], data[1, :], data[2, :], label="centered data")


                # self.draw3DRectangle(xmin, ymin, zmin, xmax, ymax, zmax)
                self.ax.set_title("Frame{}".format(frame))

        return self.scatter
            


    def animate(self):
        ani = animation.FuncAnimation(self.fig,self.thread_plot, interval = 1000, cache_frame_data=False)
        plt.show()

        # except KeyboardInterrupt:
        #     # self.cli_port.write(('sensorStop\n').encode())
        #     # self.cli_port.close()
        #     # self.data_port.close()
        #     # self.view.destroy()
        #     break


