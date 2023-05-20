
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

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Mostly all good, just need Richo to talk me through a bit of the data cleaning in normal_plot

MAX_CSVS = 10

df = pd.read_csv('Data_set6_removed_peak_grouping_standing\\Data_set6_removed_peak_grouping_standing\\Data_0.csv')
# Generate some random data for the scatter plot
# Dropping the first row
df = df.drop(df.index[0])

n = 100  # Number of data points
x = df['X']
y = df['Y']
z = df['Z']


# Create a pca and fit it down to a 2-dimensional system
def PCA_test(x):
    x = StandardScaler().fit_transform(x)

    # Might be worth investigating different configurations for the pca
    # I always found n_components = 'mle' to be the best
    # I see, this is for plotting purposes
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print(principalComponents)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['X', 'Y'])
    return principalDf


# Create a figure and a 3D axis
def plot_PCA():
    fig = plt.figure()
    
    df = pd.read_csv('Data_set6_removed_peak_grouping_standing\\Data_0.csv')
    # Dropping the first row
    df = df.drop(df.index[0])

    n = 100  # Number of data points
    x = df['X']
    y = df['Y']
    z = df['Z']

    #  Plot the 2D scatter plot
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, c='b', marker='o')

    # for a 3d equivalent:
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y,z,c='b', marker='o')


    # Set labels and title
    for i in range(1,MAX_CSVS,1):

        print("plotting dataset {0}".format(i))
        df = pd.read_csv('Data_{}.csv'.format(i))
        # Dropping the first row
        df = df.drop(df.index[0])

        # Take every entry from the x, y, x, velocity columns
        data = df.loc[:,'X':'Velocity']
        df = PCA_test(data)

        x = df['X']
        y = df['Y']

        
        # Extract x and y data from the current data frame
        # Update the scatter plot with new data
        # 2D equivalent
        scatter.remove()
        scatter = ax.scatter(x, y,c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # For the 3d equivalent
        # z = df['Z']
        # scatter.set_data(x, y,z)
        # scatter = ax.scatter(x, y,z,c='b', marker='o')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Scatter Plot')

        # Pause to allow time for the plot to be updated
        plt.pause(1)
        # Update the legend

    plt.show()


# This is for plotting the 3d image. No PCA required
def normal_plot():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df = pd.read_csv('\Data_set6_removed_peak_grouping_standing\\Data_0.csv')
    df = df.drop(df.index[0])

    # Create a PathPatch object for the polygon marker
    center_df = pd.DataFrame({'X': [0],'Y': [1],'Z': [2],'Velocity':[0],'label':1})
            
    n = 100  # Number of data points
    x = df['X']
    y = df['Y']
    z = df['Z']
    df['label'] = [0] * len(df)
    print(df)

    center_points = pd.DataFrame({"X":[0],"Y":[0],"Z":[0], "label":[1],"Velocity":[0]})

    scatter = ax.scatter(df['X'],df['Y'],df['Z'], marker='o',label='data')
    scatter_2 = ax.scatter(center_points['X'],center_points['Y'],center_points['Z'], marker='o',label='centre')

    # Set labels and title
    for i in range(1,MAX_CSVS,1):
        print("iterating over csv {0}".format(i))

        df = pd.read_csv('Data_{}.csv'.format(i))
        df = df.drop(df.index[0])
        
        # Scale the data over a standard distribution
        X = StandardScaler().fit_transform(df)

        # Take the x, y, z columns
        data = df.loc[:,'X':'Z']

        # Create a db clustering algorithm
        db = DBSCAN(eps=0.8, min_samples=20).fit(data)
        labels = db.labels_ 

        # What shape is the dictionary in that allows it to be accessed this way?
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        center_points = []

        for cluster_label in range(n_clusters_):

            # Iterate through, assign each point to a cluster and find the centre point of each
            cluster_points = data[labels == cluster_label]
            center_point = np.mean(cluster_points, axis=0)
            center_points.append(center_point)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        # Generate some random data for the scatter plot
        # Dropping the first row
        # Why?
        df = df.drop(df.index[0])
        df = df.drop(df.columns[0], axis=1)

        x = df['X']
        y = df['Y']
        z = df['Z']

        df['label'] = [0] * len(df)

        # Extract x and y data from the current data frame
        # Update the scatter plot with new data
        # old code; can it be removed???

        # scatter_2.remove()
        # scatter.set_data(x, y,z)
        # if len(scatter)!=0:

        # if len(scatter)!=0:
        #     scatter.remove()
        
        ax.cla()
        if(n_clusters_!=0):
            print("NO")
            center_points = np.array(center_points)
            print(center_points)
            center_points = pd.DataFrame({'X': center_points[:,0],'Y': center_points[:,1],'Z': center_points[:,2],'Velocity':[0]*n_clusters_,'label':np.ones(n_clusters_)})
            center_df = pd.concat([center_df, center_points]).reset_index(drop=True)
            ax.scatter(center_points['X'], center_points['Y'], center_points['Z'],c = 'blue',marker = 's',label='centre',s=200)  # Update cluster centers
            
        # Plot each data point on the graph, and add labes, axes etc.
        ax.scatter(df['X'], df['Y'],df['Z'],c='red', marker='o',label='data')
        
        ax.set_xlabel('X')
        ax.set_xlim([-2,2])
        ax.set_ylabel('Y')
        ax.set_ylim([-2,2])
        ax.set_zlabel('Z')
        ax.set_zlim([-2,2])

        ax.set_title('3D Scatter Plot')
        ax.legend()

        # Pause to allow time for the plot to be updated
        plt.pause(1)

    plt.show()
    # Save all the plot points
    center_df.to_csv('center_points.csv')


if __name__=="__main__":
    # plot_PCA()
    normal_plot()

