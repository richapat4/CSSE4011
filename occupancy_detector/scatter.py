
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



df = pd.read_csv('Data_set6_removed_peak_grouping_sitting/Data_0.csv')
# Generate some random data for the scatter plot
# Dropping the first row
df = df.drop(df.index[0])

n = 100  # Number of data points
x = df['X']
y = df['Y']
z = df['Z']


# plt.scatter(x,y)
# plt.show()
# # z = df['Z']






def PCA_test(x):
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print(principalComponents)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['X', 'Y'])
    return principalDf
    # print(principalDf)
    # finalDf = pd.concat([principalDf, y], axis = 1)
    # finalDf.columns = ['principal component 1', 'principal component 2']
    # Xax = np.array(principalDf['principal component 1'])
    # Yax = np.array(principalDf['principal component 2'])
    # sns.scatterplot(principalDf,x="principal component 1", y = "principal component 2")
    # sns.set_palette("rocket", n_colors=len(label_array))
    # plt.title("PCA Plot of Species Caught by Type")
    # plt.show()


# Create a figure and a 3D axis
def plot_PCA():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot
    df = pd.read_csv('Data_set5_removed_peak_grouping_standing/Data_0.csv')
    # Generate some random data for the scatter plot
    # Dropping the first row
    df = df.drop(df.index[0])

    n = 100  # Number of data points
    x = df['X']
    y = df['Y']
    z = df['Z']
    scatter = ax.scatter(x, y, c='b', marker='o')
    # scatter = ax.scatter(x, y,z,c='b', marker='o')
    # Set labels and title
    for i in range(1,10,1):
        print("here")
        df = pd.read_csv('Data_{}.csv'.format(i))
        # Generate some random data for the scatter plot
        # Dropping the first row
        df = df.drop(df.index[0])
        data = df.loc[:,'X':'Velocity']
        # n = 100  # Number of data points
        df = PCA_test(data)

        x = df['X']
        y = df['Y']

        # z = df['Z']
        # Extract x and y data from the current data frame
        # Update the scatter plot with new data
        scatter.remove()
        # scatter.set_data(x, y,z)
        # scatter = ax.scatter(x, y,z,c='b', marker='o')
        scatter = ax.scatter(x, y,c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Scatter Plot')

        # Pause to allow time for the plot to be updated
        plt.pause(1)
        # Update the legend

    plt.show()




def normal_plot():

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot
    df = pd.read_csv('Data_set5_removed_peak_grouping_standing/Data_0.csv')
    # Generate some random data for the scatter plot
    # Dropping the first row
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
    scatter_2 = ax.scatter(center_points['X'],center_points['Y'],center_points['Z'], marker=path,label='centre')

    # Set labels and title
    for i in range(1,10,1):
        print("here")

        df = pd.read_csv('Data_set5_removed_peak_grouping_standing/Data_{}.csv'.format(i))
        df = df.drop(df.index[0])
        
        X = StandardScaler().fit_transform(df)

        data = df.loc[:,'X':'Z']

        db = DBSCAN(eps=0.2, min_samples=10).fit(data)
        labels = db.labels_ 

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        center_points = []

        for cluster_label in range(n_clusters_):
            cluster_points = data[labels == cluster_label]
            center_point = np.mean(cluster_points, axis=0)
            
            center_points.append(center_point)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        # Generate some random data for the scatter plot
        # Dropping the first row
        df = df.drop(df.index[0])
        df = df.drop(df.columns[0], axis=1)

        # n = 100  # Number of data points
        x = df['X']
        y = df['Y']
        z = df['Z']

        df['label'] = [0] * len(df)

        # Extract x and y data from the current data frame
        # Update the scatter plot with new data
        
        # scatter_2.remove()
        # scatter.set_data(x, y,z)
        # if len(scatter)!=0:

        # if len(scatter)!=0:
        #     scatter.remove()
        
        ax.cla()
        if(n_clusters_!=0):
            print("NO")
            center_points = np.array(center_points)
            center_points = pd.DataFrame({'X': center_points[:,0],'Y': center_points[:,1],'Z': center_points[:,2],'Velocity':[0]*n_clusters_,'label':np.ones(n_clusters_)})
            center_df = pd.concat([center_df, center_points]).reset_index(drop=True)
            ax.scatter(center_points['X'], center_points['Y'], center_points['Z'],c = 'blue',marker = 's',label='centre',s=200)  # Update cluster centers
            
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
    center_df.to_csv('center_points.csv')



# plot_PCA()
normal_plot()




        # ax.scatter(center_points['X'], center_points['Y'], center_points['Z'], c='red', marker='x', 
        #            label='Cluster Centers')

        # scatter = ax.scatter(x, y,c='b', marker='o')
