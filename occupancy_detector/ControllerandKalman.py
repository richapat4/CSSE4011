import serial
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signal
import sys

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from flightsql import FlightSQLClient
 
from datetime import datetime
from View2 import View
from sklearn.cluster import DBSCAN

import matplotlib.animation as animation
from matplotlib.artist import 

import queue


MAX_CSVS = 10
WRITE_GAP = 0.2

KALMAN_GAP = 1

# Grafana http://localhost:3000/
# influxdb http://localhost:8086/

# ------------------------------------------------------------------

def sig_handler(signal, frame):

    if interface.cli_port is not None and interface.data_port is not None:
        interface.cli_port.write(('sensorStop\n').encode())
        interface.cli_port.close()
        interface.data_port.close()

    sys.exit(0)

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
        # self.cli_port = serial.Serial('/dev/ttyACM0', 115200)
        # self.data_port = serial.Serial('/dev/ttyACM1', 921600)
        
        # Windows
        self.cli_port = serial.Serial('COM14', 115200)
        self.data_port = serial.Serial('COM15', 921600)

        # token = "whl_f4m7pZbnLdO6KHYmNFjFdJaGimywqZXMezcOCwFcwJyUOW0nomnbHXzMdrxf3TeKOGbzpUW4B2rDXgUu5Q=="
        # org = "csse4011"
        # url = "http://localhost:8086"

        # self.bucket="Demo_data"

        # self.write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org, debug=False)

        # self.write_api = self.write_client.write_api(write_options=SYNCHRONOUS)

        self.duration = 2
        # Change the configuration file name
        configFileName = 'occupancy_detector/test_4_removed_range_peak_grouping.cfg'
        self.count = 0

        # Input buffers for reading into port dicts
        self.byte_buffer = np.zeros(2**15,dtype = 'uint8')
        self.byte_buffer_len = 0

        self.cluster_points = []
        self.separated_clusters = pd.DataFrame({'X': [0], 'Y': [0], 'Z': [0],'cluster_num':[0]}, index=[0]) #Init as dataframe
        self.total_lines = []
        self.testData = pd.DataFrame({'X': 0, 'Y': 0, 'Z': 0,'Velocity': 2}, index=[0])
    
        # Configurate the serial port
        self.serialConfig(configFileName)

        # Get the configuration parameters from the configuration file
        self.configParameters = self.parseConfigFile(configFileName)

        self.start_time = time.time()
        self.kalman_time = time.time()

        # Main loop 
        self.detObj = {}  
        self.frameData = {}    
        self.currentIndex = 0

        #Plot stuff
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.cluster_points = []
        self.num_clusters = 0

        signal.signal(signal.SIGINT, self.sig_handler)

        thread = threading.Thread(target=self.main_loop)
        thread.daemon = True
        thread.start()


        pseudo_data = {'X':[0],'Y':[0],'Z':[0],'cluster_num':[0]}
        data = pd.DataFrame(pseudo_data)

        #init plot
        self.scatter = self.ax.scatter(self.separated_clusters['X'], self.separated_clusters['Y'],self.separated_clusters['Z'], c=self.separated_clusters['cluster_num'])
        
        self.ax.set_xlabel('X')
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim([0, 5])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim([-1.5, 3.5])


        # Apply clustering algorithm
    def clustering(self, testData):

        center_df = pd.DataFrame({'X': [0],'Y': [0],'Z': [0], 'label':[0]})
        separated_cluster = []
        total_cluster = pd.DataFrame({'X': 0, 'Y': 0, 'Z': 0,'cluster_num': 0}, index=[0])

        self.separated_clusters = pd.DataFrame() #clear data frame here for new clusters
            
        if(len(testData) > 0):


            # Create a db clustering algorithm
            db = DBSCAN(eps=0.8, min_samples=20).fit(testData)
            labels = db.labels_ 

            # What shape is the dictionary in that allows it to be accessed this way?
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            separated_cluster = []
            center_df = pd.DataFrame({'X': [0],'Y': [0],'Z': [0], 'label':[0]})

            for cluster_label in range(n_clusters_):

                # Iterate through, assign each point to a cluster and find the centre point of each
                cluster_points = testData[labels == cluster_label]
                cluster_points = cluster_points.assign(cluster_num=cluster_label)

                separated_cluster.append(cluster_points)

                total_cluster = pd.concat([total_cluster, cluster_points], axis=0)

                center_point = np.mean(cluster_points, axis=0)
                entry = pd.Series({'X':center_point['X'], 'Y':center_point['Y'], 'Z':center_point['Z'], 'label':center_point['cluster_num']})
                center_df.loc[len(center_df)] = entry
            
            self.num_clusters = n_clusters_
            self.cluster_points = center_df.drop(center_df.index[0])
            total_cluster = total_cluster.drop(total_cluster.index[0])
            self.separated_clusters = pd.concat([self.separated_clusters, total_cluster], axis=0) 
            # self.separated_clusters = self.separated_clusters.drop(self.separated_clusters.index[0])

      


    """
    Creates thread to plot data
    """
    def thread_plot(self,frame):
        print("HERE")

        # separated_clusters = self.separated_clusters
        print(self.separated_clusters['cluster_num'].unique())
        if not self.separated_clusters.empty:
            # x = separated_clusters['X'].values
            # y = separated_clusters['Y'].values
            # z = separated_clusters['Z'].values

            # data = np.vstack([x, y, z])

            self.scatter._offsets3d = (self.separated_clusters['X'], self.separated_clusters['Y'], self.separated_clusters['Z'])
        
            self.ax.set_title('Number of occupants: {0}'.format(self.num_clusters))

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

                    # Does the writing every 1.5 seconds
                    if((time.time() - self.start_time) > WRITE_GAP):
                        # For some reason this csv writing is breaking the system???

                        # Implement data cleaning algorithm here
                        testDataNew = self.data_cleaning(self.testData.copy())

                        # Implement DB clustering algorithm here
                        self.clustering(testDataNew)
                        

                        
                    
                    

                        # This is where we write the field positioning over to influxdb
                        # This is only going to be useful once we have clustering algorithms

                        # print(self.cluster_points)

                        for index, cluster in self.cluster_points.iterrows():

                            point = (
                                Point("Position")
                                .field("X", cluster['X'])
                                .field("Y", cluster['Y'])
                                .field("Z", cluster['Z'])
                                .field("cluster_num", cluster['label'])
                                )
                            
                            # self.write_api.write(bucket=self.bucket, org="csse4011", record=point)

                        self.testData = pd.DataFrame({'X': 0, 'Y': 0, 'Z': 0,'Velocity': 2}, index=[0])
                        print("evaluation done")
                        self.count+=1
                        self.start_time = time.time()

                if((time.time() - self.kalman_time) > KALMAN_GAP):
                    pass


            # Stop the program and close everything if Ctrl + c is pressed
            except KeyboardInterrupt:
                self.cli_port.write(('sensorStop\n').encode())
                self.cli_port.close()
                self.data_port.close()
                break
                

    # Data cleaning algorithm for applying filters, etc.
    def data_cleaning(self, testData):
        testData = testData.drop(testData.index[0])

        testData = testData.loc[:, 'X':'Z']


        return testData
    

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

                        # if((time.time() - start_time) > 1):
                        # Store another value within testData. This will only store a max of ten values
                        # if(self.count < MAX_CSVS):

                        entry = pd.Series({"X":x[objectNum], "Y":y[objectNum] , "Z":z[objectNum] , "Velocity": velocity[objectNum]})
                        self.testData.loc[len(self.testData)] = entry

                        #enqueue here:

                        


                            # print(self.testData)
                        # count+=1
                        # start_time = time.time()
                    #     entry = pd.DataFrame([x[objectNum], y[objectNum],z[objectNum],velocity[objectNum]],
                    #    columns = ['X','Y','Z','Velocity'])
                
                    # Store the data in the detObj dictionary
                    detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity":velocity}
                    
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
                    
                    # plt.clf()
                    # cs = plt.contourf(rangeArray,dopplerArray,rangeDoppler)
                    # self.fig.colorbar(cs, shrink=0.9)
                    # self.fig.canvas.draw()
                    # plt.pause(0.1)
    
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

    ani = animation.FuncAnimation(interface.fig, interface.thread_plot, interval = 1000, cache_frame_data=False)
    plt.show()

    # interface.view.mainloop()
            
    





