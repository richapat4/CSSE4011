# CSSE4011
CSSE4011 Radar Project

## Instructions for re-configuration

1) The .cfg files for the AWR1843Boost can be adjusted to have it return desired parameters. Accompanying instructions for reconfiguring AWR1843Boost .cfg files can be found [here](https://dev.ti.com/tirex/explore/node?a=VLyFKFf__4.12.1&node=A__AGvTEJkh-csqqwXnVhDbTQ__radar_toolbox__1AslXXD__LATEST&search=config). The mmWave visualiser is a handy tool that can be used for testing and can be found [here](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/)
2) Final_Controller.py is the finalised version of the occupancy detector. The Controller.clustering() function can be altered in to change the clustering and particle filtering algorithms. Controller.thread_plot() can be change to adjust the GUI data displayed.
3) Gaussian particle filtering is used along DBSCAN to clean and cluster the data. The Sigma, Epsilon and minimum_samples variables of these algorithms can be adjusted using the pre-defined constants


## Instructions for Deployment

Accompanying instructions for AWR1843 configuration can be found [here](https://dev.ti.com/tirex/explore/content/radar_toolbox_1_00_01_07/source/ti/examples/Out_Of_Box_Demo/docs/Out_Of_Box_Demo_User_Guide.html)

1) Clone git into local repository
2) In terminal locate CLI and data ports of AWR1843Boost
3) Configure AWR1843 SOP pins for flashing mode
4) Flash Out Of Box demo prebuilt binaries using UniFlash
5) Configure AWR1843 SOP pins for functional mode
7) Press the reset button on AWR1843Boost
8) Set up radar as shown below
<img width="287" alt="radar_setup" src="https://github.com/richapat4/CSSE4011/assets/91168723/db36a11a-4374-49a1-8a03-fc7315b47804">  
10) Run Final_Controller.py from the Git home directory
