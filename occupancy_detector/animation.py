import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import pandas as pd

class Controller:
    def __init__(self):
        # Initialize the figure and 3D axes
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')



        # initialize pseudodata of input.
        pseudo_data = {'X': [0,1], 'Y': [0,1], 'Z': [0,1], 'cluster_num': [0,1]}
        pseudo_centroid_data = {'X': [0,1], 'Y': [0,1], 'Z': [0,1], 'label': [0,1]}

        # Create DataFrame
        data = pd.DataFrame(pseudo_data)
        centroid_data = pd.DataFrame(pseudo_centroid_data)

        print(data)
        # self.scatter = self.ax.scatter(data['X'], data['Y'], data['Z'], c=data.cluster_num)
        self.scatter = self.ax.scatter(data['X'], data['Y'], data['Z'], c=data['cluster_num'])
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])

        #self.centroid_scatter = ax.scatter(centroid_data['X'], centroid_data['Y'], centroid_data['Z'], c='blue', marker='s', label='centre', s=100)  # Update cluster centers

    # Update function for the animation
    def update(self, frame):

        if (frame % 2 == 0):
            # initialize pseudodata of input.
            new_pseudo_data = {'X': [2, 3], 'Y': [2, 3], 'Z': [2, 3], 'cluster_num': [1, 0]}
            new_pseudo_centroid_data = {'X': [2,3], 'Y': [2,3], 'Z': [2,3], 'label': [1,0]}

            # Create DataFrame
            new_data = pd.DataFrame(new_pseudo_data)
            new_centroid_data = pd.DataFrame(new_pseudo_centroid_data)

        else:
            # initialize pseudodata of input.
            new_pseudo_data = {'X': [4, 5], 'Y': [4, 5], 'Z': [4, 5], 'cluster_num': [0, 1]}
            new_pseudo_centroid_data = {'X': [4,5], 'Y': [4,5], 'Z': [4,5], 'label': [0,1]}

            # Create DataFrame
            new_data = pd.DataFrame(new_pseudo_data)
            new_centroid_data = pd.DataFrame(new_pseudo_centroid_data)

        # Update the positions of the scatter plot
        self.scatter._offsets3d = (new_data['X'], new_data['Y'], new_data['Z'])
        #self.centroid_scatter._offsets3d = (new_centroid_data['X'], new_centroid_data['Y'], new_centroid_data['Z'])

        # Set the title of the plot with the frame number
        self.ax.set_title('Frame {}'.format(frame))

        return self.scatter, #self.centroid_scatter

if __name__ == "__main__":

    gui = Controller()

    # Create the animation using FuncAnimation
    ani = animation.FuncAnimation(gui.fig, gui.update, interval=1000, cache_frame_data=False)

    # Display the animation
    plt.show()
    time.sleep(2)
