# from filterpy import KalmanFilter 
# import numpy as np
# from filterpy.common import Q_discrete_white_noise


import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

f = KalmanFilter(6, 3) 

f.x = np.array([0., 0., 0., 0., 0., 0.]) # STATE Vector x, y, z, 

# state matrix based on equations of motion for x,y,z

# A = np.array([[1, 0, 0, dt, 0, 0],
#               [0, 1, 0, 0, dt, 0],
#               [0, 0, 1, 0, 0, dt],
#               [0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1]])

f.F = np.asarray(
    [
        [1., 0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]
    ]
)

#x,y,z position outputs
f.H = np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.]
])

f.R = 5 * np.eye(3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([0], [0], [0], marker='o',label='data')
scatter_2 = ax.scatter([0], [0], [0], marker='o',label='data')

ax.set_xlabel('X')
ax.set_xlim([-10,10])
ax.set_ylabel('Y')
ax.set_ylim([-1,1])
ax.set_zlabel('Z')
ax.set_zlim([-1,1])

ax.set_title('3D Scatter Plot')
ax.legend()


def main():
    for i in range(10):
        ax.cla()
        z = np.array([i, 0.1*i, 0.01*i])
        print('Measured: ', z)
        ax.scatter(z[0],z[1],z[2],c='red', marker='o',label='measured')
        f.predict()
        print('predicted: ', f.x)
        ax.scatter(f.x[0],f.x[1],f.x[2],c='blue', marker='o',label='predicted')
        f.update(z)
        
        ax.set_xlabel('X')
        ax.set_xlim([-10,10])
        ax.set_ylabel('Y')
        ax.set_ylim([-1,1])
        ax.set_zlabel('Z')
        ax.set_zlim([-1,1])

        ax.legend()

        plt.pause(1)

    plt.show()

if __name__ == '__main__':
    main()