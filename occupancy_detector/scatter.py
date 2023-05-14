
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


df = pd.read_csv('Data_0.csv')
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







# unique_labels = set(labels)
# core_samples_mask = np.zeros_like(labels, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True

# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


# Create a figure and a 3D axis
def plot_PCA():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot
    df = pd.read_csv('Data_0.csv')
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

    plt.show()




def normal_plot():

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D scatter plot
    df = pd.read_csv('Data_0.csv')
    # Generate some random data for the scatter plot
    # Dropping the first row
    df = df.drop(df.index[0])

    n = 100  # Number of data points
    x = df['X']
    y = df['Y']
    z = df['Z']
    # scatter = ax.scatter(x, y, c='b', marker='o')
    scatter = ax.scatter(x, y,z,c='b', marker='o')
    # Set labels and title
    for i in range(1,10,1):
        print("here")

        df = pd.read_csv('Data_{}.csv'.format(i))
        df = df.drop(df.index[0])
        
        X = StandardScaler().fit_transform(df)

        data = df.loc[:,'X':'Z']

        db = DBSCAN(eps=0.2, min_samples=10).fit(data)
        labels = db.labels_ 
        # model  = KernelDensity(kernel='gaussian', bandwidth= 0.05)
        # model.fit(data)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        # Generate some random data for the scatter plot
        # Dropping the first row
        df = df.drop(df.index[0])

        # n = 100  # Number of data points
        x = df['X']
        y = df['Y']
        z = df['Z']
        # Extract x and y data from the current data frame
        # Update the scatter plot with new data
        scatter.remove()
        # scatter.set_data(x, y,z)
        scatter = ax.scatter(x, y,z,c='b', marker='o')
        # scatter = ax.scatter(x, y,c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_xlim([-2,2])
        ax.set_ylabel('Y')
        ax.set_ylim([-2,2])
        ax.set_zlabel('Z')
        ax.set_zlim([-2,2])

        ax.set_title('3D Scatter Plot')


        # Pause to allow time for the plot to be updated
        plt.pause(1)

    plt.show()



# plot_PCA()
normal_plot()