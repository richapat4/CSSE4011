from filterpy import KalmanFilter 
import numpy as np

f = KalmanFilter(dim_x = 2, dim_z = 1)


#initial array  x position and x velocity 
f.x = np.array([2.0, 0.0])

#state transition matrix 

f.F = np.array([[1.,1.],
                [0.,1.]])

f.H = np.array([[1.,0.]])  #measurement function 
