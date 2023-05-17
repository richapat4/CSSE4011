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

        self.thread_update()

    
    """
    Creates thread to send information over selected coms line to STM32
    """
    def thread_update(self):

        pass

        # thread = threading.Thread(target=self.controller.main_loop)
        # thread.start()
        # thread.join()
        
        # self.after(1, self.thread_update)

        # connected = 0
        # for p in list(serial.tools.list_ports.comports()):
        #     if self.controller.data_port == p.device:
        #         connected = 1
                

        # if connected == 0:

        #     self.controller.data_port = None

        # # Wait for an amount of time before repeating
       

