import serial
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signal

import queue

import sys

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from flightsql import FlightSQLClient
 
from datetime import datetime
from sklearn.cluster import DBSCAN

import matplotlib.animation as animation
from matplotlib.artist import Artist
from scipy.ndimage import gaussian_filter
import statistics

MAX_CSVS = 10
WRITE_GAP = 0.1
EPSILON = 0.15
RADIUS = 1
SIGMA = 0.3
MIN_SAMPLES = 15
NUM_BOXES = 10
BOX_SIZE = 0.2
FUNC_INTERVAL = 100


# Grafana http://localhost:3000/
# influxdb http://localhost:8086/

# ------------------------------------------------------------------


# Handles exiting upon pressing ctrl+c
def sig_handler(signal, frame):

    if interface.cli_port is not None and interface.data_port is not None:
        interface.cli_port.write(('sensorStop\n').encode())
        interface.cli_port.close()
        interface.data_port.close()

    sys.exit(0)


# Controller class for data reading
class Controller:

    """
    Initialise all controller variables
    """
    def __init__(self):



        # Open the serial ports for the configuration and the data ports

        # Note which system we're using. 
        # David: Linux
        # Richa: Windows
        
        # Linux
        self.cli_port = serial.Serial('/dev/ttyACM0', 115200)
        self.data_port = serial.Serial('/dev/ttyACM1', 921600)
        
        # Windows
        # self.cli_port = serial.Serial('COM14', 115200)
        # self.data_port = serial.Serial('COM15', 921600)

        token = "whl_f4m7pZbnLdO6KHYmNFjFdJaGimywqZXMezcOCwFcwJyUOW0nomnbHXzMdrxf3TeKOGbzpUW4B2rDXgUu5Q=="
        org = "csse4011"
        url = "http://localhost:8086"

        self.bucket="Demo_data"

        self.write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org, debug=False)

        self.write_api = self.write_client.write_api(write_options=SYNCHRONOUS)

        self.duration = 2
        configFileName = 'occupancy_detector/profile_config_0.03_velocity_res.cfg'
        self.count = 0

        # Input buffers for reading into port dicts
        self.byte_buffer = np.zeros(2**15,dtype = 'uint8')
        self.byte_buffer_len = 0

        self.cluster_points = pd.DataFrame({'X': [0], 'Y': [0], 'Z': [0],'cluster_num':[0]}, index=[0]) #Init as dataframe
        self.separated_clusters = pd.DataFrame({'X': [0], 'Y': [0], 'Z': [0],'cluster_num':[0]}, index=[0]) #Init as dataframe
        self.total_lines = []
        self.testData = pd.DataFrame({'X': 0, 'Y': 0, 'Z': 0,'Velocity': 2}, index=[0])
    
        # Configurate the serial port
        self.serialConfig(configFileName)

        # Get the configuration parameters from the configuration file
        self.configParameters = self.parseConfigFile(configFileName)

        self.start_time = time.time()

        self.detObj = {}  
        self.frameData = {}    
        self.currentIndex = 0
        self.dataQueue = queue.Queue(maxsize=5)

        #Plot stuff
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.num_clusters = [0] * 5

        self.grid_lines = [[] for i in range(NUM_BOXES)]

        # Draw the grid for funcanimate
        for i in range(NUM_BOXES):
            self.draw3DRectangle_once(0, 0, 0, 0, 0, 0, i)

        # Start thread for data reading
        signal.signal(signal.SIGINT, self.sig_handler)

        thread = threading.Thread(target=self.main_loop)
        thread.daemon = True
        thread.start()

        #init plot
        self.scatter = self.ax.scatter(self.separated_clusters['X'], self.separated_clusters['Y'],self.separated_clusters['Z'], c=self.separated_clusters['cluster_num'], zorder=1)
        self.scatter_centre = self.ax.scatter(self.cluster_points['X'], self.cluster_points['Y'],self.cluster_points['Z'], zorder=3, marker='s', c='blue', s=200)

        self.ax.set_xlabel('X')
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim([0, 7])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim([-1.5, 3.5])


    # Apply clustering algorithm
    def clustering(self, testData):

        center_df = pd.DataFrame({'X': [0],'Y': [0],'Z': [0], 'label':[0]})

        self.separated_clusters = pd.DataFrame() #clear data frame here for new clusters
            
        if(len(testData) > 0):

            data = testData.copy()
            data = data.dropna()
            x = data['X']
            y = data['Y']
            data = np.vstack((x, y))

            smoothed_data = gaussian_filter(data, sigma=SIGMA, radius=RADIUS)

            x = smoothed_data[0,:]
            y = smoothed_data[1,:]

            smoothed_data = pd.DataFrame({'X': x, 'Y': y})

            # Create a db clustering algorithm
            db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(smoothed_data)
            labels = db.labels_ 

            indices = [count for count, value in enumerate(labels) if value != -1]

            cluster_indices = [labels[index] for index in indices]
            cluster_points = testData.loc[indices]
            cluster_points['cluster_num'] = cluster_indices

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            center_df = pd.DataFrame({'X': [0],'Y': [0],'Z': [0], 'label':[0]})

            for cluster_label in range(n_clusters_):

                # Iterate through, assign each point to a cluster and find the centre point of each
                cluster = cluster_points[cluster_points['cluster_num'] == cluster_label]

                center_point = np.mean(cluster, axis=0)
                entry = pd.Series({'X':center_point['X'], 'Y':center_point['Y'], 'Z':center_point['Z'], 'label':center_point['cluster_num']})
                center_df.loc[len(center_df)] = entry

                # Box dimensions, then update hit boxes
                x1, x2, y1, y2, z1, z2 = entry['X'] -  BOX_SIZE, entry['X'] + BOX_SIZE, \
                        entry['Y'] - BOX_SIZE, entry['Y'] + BOX_SIZE, \
                        0, 1.8
                
                self.draw3DRectangle_update(x1, x2, y1, y2, z1, z2, cluster_label)

            # Remove any extraneous hitboxes
            for i in range(n_clusters_, NUM_BOXES):
                self.draw3DRectangle_update(0, 0, 0, 0, 0, 0, i)
            
            # Set plotting variables
            self.num_clusters.pop(0)
            self.num_clusters.append(n_clusters_)
            self.cluster_points = center_df.drop(center_df.index[0])
            self.separated_clusters = cluster_points


    # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
    def draw3DRectangle_once(self, x1, x2, y1, y2, z1, z2, place):
        
        self.grid_lines[place].append(self.ax.plot([x1, x2], [y1, y1], [z1, z1], color='b'))  # | (up)

        self.grid_lines[place].append(self.ax.plot([x2, x2], [y1, y2], [z1, z1], color='b'))  # -->
        self.grid_lines[place].append(self.ax.plot([x2, x1], [y2, y2], [z1, z1], color='b'))  # | (down)
        self.grid_lines[place].append(self.ax.plot([x1, x1], [y2, y1], [z1, z1], color='b'))  # <--

        self.grid_lines[place].append(self.ax.plot([x1, x2], [y1, y1], [z2, z2], color='b'))  # | (up)
        self.grid_lines[place].append(self.ax.plot([x2, x2], [y1, y2], [z2, z2], color='b'))  # -->
        self.grid_lines[place].append(self.ax.plot([x2, x1], [y2, y2], [z2, z2], color='b'))  # | (down)
        self.grid_lines[place].append(self.ax.plot([x1, x1], [y2, y1], [z2, z2], color='b'))  # <--

        self.grid_lines[place].append(self.ax.plot([x1, x1], [y1, y1], [z1, z2], color='b'))  # | (up)
        self.grid_lines[place].append(self.ax.plot([x2, x2], [y2, y2], [z1, z2], color='b'))  # -->
        self.grid_lines[place].append(self.ax.plot([x1, x1], [y2, y2], [z1, z2], color='b'))  # | (down)
        self.grid_lines[place].append(self.ax.plot([x2, x2], [y1, y1], [z1, z2], color='b'))  # <--


    # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
    def draw3DRectangle_update(self, x1, x2, y1, y2, z1, z2, place):
       
        self.grid_lines[place][0][0].set_data_3d([x1, x2], [y1, y1], [z1, z1])  # | (up)

        self.grid_lines[place][1][0].set_data_3d([x2, x2], [y1, y2], [z1, z1])  # -->
        self.grid_lines[place][2][0].set_data_3d([x2, x1], [y2, y2], [z1, z1])  # | (down)
        self.grid_lines[place][3][0].set_data_3d([x1, x1], [y2, y1], [z1, z1])  # <--

        self.grid_lines[place][4][0].set_data_3d([x1, x2], [y1, y1], [z2, z2])  # | (up)
        self.grid_lines[place][5][0].set_data_3d([x2, x2], [y1, y2], [z2, z2])  # -->
        self.grid_lines[place][6][0].set_data_3d([x2, x1], [y2, y2], [z2, z2])  # | (down)
        self.grid_lines[place][7][0].set_data_3d([x1, x1], [y2, y1], [z2, z2])  # <--

        self.grid_lines[place][8][0].set_data_3d([x1, x1], [y1, y1], [z1, z2])  # | (up)
        self.grid_lines[place][9][0].set_data_3d([x2, x2], [y2, y2], [z1, z2])  # -->
        self.grid_lines[place][10][0].set_data_3d([x1, x1], [y2, y2], [z1, z2])  # | (down)
        self.grid_lines[place][11][0].set_data_3d([x2, x2], [y1, y1], [z1, z2])  # <--
                


    """
    Creates thread to plot data
    """
    def thread_plot(self,frame):
        
        if not self.separated_clusters.empty:
            try:

                self.scatter._offsets3d = (self.separated_clusters['X'], self.separated_clusters['Y'], self.separated_clusters['Z'])
                self.scatter_centre._offsets3d = (self.cluster_points['X'], self.cluster_points['Y'], self.cluster_points['Z'])

                # Need to choose mean, median or mode (whichever works better)
                self.ax.set_title('Number of occupants: {0}'.format((statistics.mode(self.num_clusters))))

            except KeyError:
                pass

        return self.scatter


    # Signal handler to kill entire system
    def sig_handler(self, signal, frame):

        if self.cli_port is not None and self.data_port is not None:
            self.cli_port.write(('sensorStop\n').encode())
            self.cli_port.close()
            self.data_port.close()

        sys.exit(0)

    
    def main_loop(self):
        # So this is all well and good. Just have to figure out where all the outputs go now
        while True:
            try:
                dataOk, self.frameNumber, self.detObj = self.readAndParseData18xx(self.data_port, self.configParameters)
                if dataOk:

                    # Store the current frame into frameData
                    self.frameData[self.currentIndex] = self.detObj
                    self.currentIndex += 1

                    #RECEIVE FROM QUEUE HERE

                    testData = self.dataQueue.get()

                    testDataNew = testData.loc[:, 'X':'Z'].copy()

                    # Implement DB clustering algorithm here
                    self.clustering(testDataNew)

                    # This is where we write the field positioning over to influxdb
                    if((time.time() - self.start_time) > WRITE_GAP):
                        for _, cluster in self.cluster_points.iterrows():

                            point = (
                                Point("Position")
                                .field("X", cluster['X'])
                                .field("Y", cluster['Y'])
                                .field("Z", cluster['Z'])
                                .field("cluster", max(self.num_clusters))
                                .tag("cluster_num", str(cluster['label']))
                                )
                            
                            self.write_api.write(bucket=self.bucket, org="csse4011", record=point)

                        self.count+=1
                        self.start_time = time.time()

            # Stop the program and close everything if Ctrl + c is pressed
            except KeyboardInterrupt:
                self.cli_port.write(('sensorStop\n').encode())
                self.cli_port.close()
                self.data_port.close()
                break
    

    # Function to configure the serial ports and send the data from
    # the configuration file to the radar
    def serialConfig(self, configFileName):

        # Read the configuration file and send it to the board
        # This will boot the board into the mode that it needs to run
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            self.cli_port.write((i+'\n').encode())
            print(i)
            time.sleep(0.01)
            

    # Function to parse the data inside the configuration file
    def parseConfigFile(self, configFileName):
        configParameters = {} # Initialize an empty dictionary to store the configuration parameters
        
        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            
            # Split the line
            splitWords = i.split(" ")
            
            # Hard code the number of antennas, change if other configuration is used
            # We don't have to change this; that's the number for AWR1843Boost
            numRxAnt = 4
            numTxAnt = 3
            
            # Get the information about the profile configuration
            # Don't need to worry about this; happens automatically, and is tweaked via changing the .cfg
            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1
                
                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                    
                digOutSampleRate = int(splitWords[11])
                
            # Get the information about the frame configuration  
            # Same here; don't have to worry about it  
            elif "frameCfg" in splitWords[0]:
                
                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])
                framePeriodicity = float(splitWords[5])

                
        # Combine the read data to obtain the configuration parameters           
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        configParameters["numDopplerBins"] = numChirpsPerFrame // numTxAnt
        configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
        configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
        configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
        configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
        configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
        
        return configParameters
    

    # Funtion to read and parse the incoming data
    # Kinda disgusting parsing, but it works and doesn't require any updating so all good!!!
    def readAndParseData18xx(self, Dataport, configParameters):
    
        
        # Constants
        OBJ_STRUCT_SIZE_BYTES = 12
        BYTE_VEC_ACC_MAX_SIZE = 2**15
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1
        MMWDEMO_UART_MSG_RANGE_PROFILE   = 2
        MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
        MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
        MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
        maxBufferSize = 2**15
        tlvHeaderLengthInBytes = 8
        pointLengthInBytes = 16
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
        
        # Initialize variables
        magicOK = 0 # Checks if magic number has been read
        dataOK = 0 # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}
        
        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
        byteCount = len(byteVec)
        
        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byte_buffer_len + byteCount) < maxBufferSize:
            self.byte_buffer[self.byte_buffer_len:self.byte_buffer_len + byteCount] = byteVec[:byteCount]
            self.byte_buffer_len = self.byte_buffer_len + byteCount
            
        # Check that the buffer has some data
        if self.byte_buffer_len > 16:
            
            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byte_buffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byte_buffer[loc:loc+8]
                if np.all(check == magicWord):
                    startIdx.append(loc)
                
            # Check that startIdx is not empty
            if startIdx:
                
                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < self.byte_buffer_len:
                    self.byte_buffer[:self.byte_buffer_len-startIdx[0]] = self.byte_buffer[startIdx[0]:self.byte_buffer_len]
                    self.byte_buffer[self.byte_buffer_len-startIdx[0]:] = np.zeros(len(self.byte_buffer[self.byte_buffer_len-startIdx[0]:]),dtype = 'uint8')
                    self.byte_buffer_len = self.byte_buffer_len - startIdx[0]
                    
                # Check that there have no errors with the byte buffer length
                if self.byte_buffer_len < 0:
                    self.byte_buffer_len = 0
                    
                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]
                
                # Read the total packet length
                totalPacketLen = np.matmul(self.byte_buffer[12:12+4],word)
                
                # Check that all the packet has been read
                if (self.byte_buffer_len >= totalPacketLen) and (self.byte_buffer_len != 0):
                    magicOK = 1
        
        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Initialize the pointer index
            idX = 0
            
            # Read the header
            magicNumber = self.byte_buffer[idX:idX+8]
            idX += 8
            version = format(np.matmul(self.byte_buffer[idX:idX+4],word),'x')
            idX += 4
            totalPacketLen = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4
            platform = format(np.matmul(self.byte_buffer[idX:idX+4],word),'x')
            idX += 4
            frameNumber = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4
            timeCpuCycles = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4
            numDetectedObj = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4
            numTLVs = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4
            subFrameNumber = np.matmul(self.byte_buffer[idX:idX+4],word)
            idX += 4

            # Read the TLV messages
            for tlvIdx in range(numTLVs):
                
                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]

                # Check the header of the TLV message
                tlv_type = np.matmul(self.byte_buffer[idX:idX+4],word)
                idX += 4
                tlv_length = np.matmul(self.byte_buffer[idX:idX+4],word)
                idX += 4

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                    # Initialize the arrays
                    x = np.zeros(numDetectedObj,dtype=np.float32)
                    y = np.zeros(numDetectedObj,dtype=np.float32)
                    z = np.zeros(numDetectedObj,dtype=np.float32)
                    velocity = np.zeros(numDetectedObj,dtype=np.float32)
                    
                    for objectNum in range(numDetectedObj):
                        # entry = {}
                        
                        # Read the data for each object
                        x[objectNum] = self.byte_buffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        y[objectNum] = self.byte_buffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        z[objectNum] = self.byte_buffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        velocity[objectNum] = self.byte_buffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                
                    # Store the data in the detObj dictionary
                    detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity":velocity}
                    
                    self.testData = pd.DataFrame({"X":x, "Y":y , "Z":z , "Velocity": velocity})
                    self.dataQueue.put(self.testData)

                    dataOK = 1

                elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:

                    # Get the number of bytes to read
                    numBytes = 2*configParameters["numRangeBins"]*configParameters["numDopplerBins"]

                    # Convert the raw data to int16 array
                    payload = self.byte_buffer[idX:idX + numBytes]
                    idX += numBytes
                    rangeDoppler = payload.view(dtype=np.int16)

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(rangeDoppler) > 10000:
                        continue

                    # Convert the range doppler array to a matrix
                    rangeDoppler = np.reshape(rangeDoppler, (configParameters["numDopplerBins"], configParameters["numRangeBins"]),'F') #Fortran-like reshape
                    rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler)/2):], rangeDoppler[:int(len(rangeDoppler)/2)], axis=0)

                    # Generate the range and doppler arrays for the plot
                    rangeArray = np.array(range(configParameters["numRangeBins"]))*configParameters["rangeIdxToMeters"]
                    dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"]/2 , configParameters["numDopplerBins"]/2), configParameters["dopplerResolutionMps"])
                    
            # Remove already processed data
            if idX > 0 and self.byte_buffer_len>idX:
                shiftSize = totalPacketLen
                        
                self.byte_buffer[:self.byte_buffer_len - shiftSize] = self.byte_buffer[shiftSize:self.byte_buffer_len]
                self.byte_buffer[self.byte_buffer_len - shiftSize:] = np.zeros(len(self.byte_buffer[self.byte_buffer_len - shiftSize:]),dtype = 'uint8')
                self.byte_buffer_len = self.byte_buffer_len - shiftSize
                
                # Check that there are no errors with the buffer length
                if self.byte_buffer_len < 0:
                    self.byte_buffer_len = 0         

        return dataOK, frameNumber, detObj


# -------------------------    MAIN   -----------------------------------------  

    
"""
Engage in the main loop
"""
if __name__ == "__main__":

    signal.signal(signal.SIGINT, sig_handler)

    interface = Controller()

    while len(interface.separated_clusters) == 0:
        time.sleep(1)

    ani = animation.FuncAnimation(interface.fig, interface.thread_plot, interval = FUNC_INTERVAL, cache_frame_data=False)
    plt.show()

            
    





