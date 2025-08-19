**NOTE:** This effort is a work in progress, and currently only includes modules to handle .DICOM (IVUS images) and .png (lumen mask images) files, process/save/load data, and build/initialize/train/save/load machine learning models. Pulse pressure waveform data is important for deducing metrics of arterial stiffness, but the functionality is missing, as the .DICOM files I currently have access to have no pulse pressure data attached. I plan to add some function that addresses this lack of data cohesion & thereby automates the process of deducing arterial stiffness metrics to the greatest extent possible – but it will require a greater amount of input (and thus, effort) the end-user.  
Initially, a model with a **U-Net** architecture was to employed for the sake of an approach focused on image segmentation (as shown in the image below and demonstrated by this initial commit).  

![U-Net architecture](/assets/images/u-net-architecture.png)

However, the end goal of deducing arterial stiffness metrics only requires a single measurement (cross-sectional area or diameter of the vascular lumen), not a 512x512 mask. Knowledge of the shape or position of the arterial lumen is irrelevant to the task at hand. So, going forward, I intend to build & train a model that produces a lower-dimensional output.  
I believe that some architecture other than a U-Net would be better suited for the task of predicting a single numerical value (regression). Maybe a standard **convolutional neural network (CNN)** – a series of convolutional and pooling layers extract features from the image, then feed the final feature vector into one or more fully connected layers that output a single numerical value.  
I may experiment with **attention mechanisms** to ignore transducer near-field artifact in IVUS images (especially in cases where the ring-down artifact was not subtracted prior to image acquisition), as well as text pertaining to the study & scale crosshairs. I may also look into taking some pre-trained model (such as **VGG** or **ResNet**) and adapting it to this task (perhaps using those first few convolutional layers which have already learned to detect low-level features).  
Also, it has come to my attention that I have yet to implement a robust feature to extract the scale represented by a given IVUS image, and have been proceeding as though each image has the same scale (500px = 14mm) despite the fact that this is not always true.  
*More to come soon. Thanks!*  


# Intravascular Ultrasound (IVUS) Arterial Stiffness Analysis System  

##### This initiative is dedicated to the development of a machine learning-driven system designed for the automated assessment of arterial stiffness. This assessment is derived from the analysis of serial, high-resolution grayscale Intravascular Ultrasound (IVUS) images and concurrently acquired physiological pulse pressure waveform data.  

## System Capabilities  
###### The system incorporates a suite of functionalities designed to facilitate comprehensive cardiovascular analysis  

### DICOM Data Ingestion  
The system is engineered to accurately read and parse .DICOM files, encompassing both the raw IVUS image data and the associated physiological waveform data.  

### IVUS Image Analysis Module  
This module is responsible for the precise identification of end-systolic and end-diastolic phases within the sequential IVUS image series. It performs quantitative measurements of the cross-sectional lumen area of the right carotid artery (RCA) at both end-systole and end-diastole.  

### Pulse Pressure Waveform Analysis Module  
This component extracts critical peak systolic and peak diastolic pulse pressure measurements from the embedded .DICOM Waveform objects.  

### Data Synchronization Mechanism  
A robust mechanism is integrated to synchronize the measurements obtained from IVUS images with their corresponding physiological pulse pressure data.  

### Arterial Stiffness Calculation  
The system calculates various metrics of arterial stiffness, based upon the precisely paired lumen area and pulse pressure measurements. The specific formulas for these computations are formally defined below:  
$A$: the cross sectional area of the vascular lumen  
$A_s$: the cross sectional area of the vascular lumen at peak systole  
$A_d$: the cross sectional area of the vascular lumen at end diastole  
$D$: the diameter of the vascular lumen  
$D_s$: the diameter of the vascular lumen at peak systole  
$D_d$: the diameter of the vascular lumen at end diastole  
$\Delta P$: the difference in peak systolic and end-diastolic arterial pulse pressure  

###### Diameter:  
$$ D = 2 \times \sqrt{\frac{A}{\pi}} $$  

###### Distensibility:  
$$ \text{Dist} = \frac{A_s - A_d}{\Delta P \times A_d} \times 100 $$

###### Compliance:  
$$ \text{Comp} = \frac{A_s - A_d}{\Delta P} $$  

###### Stiffness Index:  
$$ \beta = \frac{\ln(\Delta P)}{\frac{D_s - D_d}{D_d}} \times 10^{-1} $$  


## Implementation and Setup Guide  
###### *Coming soon. I appreciate your patience!*  

#### Prerequisites:  
**OS:**  
**IDE:**  
**Package Manager:**  
**Python Interpreter:**  

#### Setup Procedures  
**Project Repository Initialization:**  
**Workspace Initialization:**  
**Extension Installation:**  
**Virtual Environment Creation:**  
**Dependency Installation:**  

## Project Architecture  
###### The project's architecture is structured into modular components for the sake of clarity, maintainability, and scalability  

ivus_stiffness_analyzer/  
├── .venv/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dedicated Python virtual environment  
├── src/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── dicom_handler.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Manages DICOM file parsing and data extraction operations  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── image_processor.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Implements IVUS image processing  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── waveform_analyzer.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Conducts analysis of pulse pressure waveforms&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{*absent*}  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── synchronizer.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Manages the synchronization of IVUS and waveform data&nbsp;&nbsp;&nbsp;{*absent*}  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── stiffness_calculator.py&nbsp;&nbsp;# Houses the algorithms for arterial stiffness metric computation  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── u_net_model.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Builds, initializes, trains, saves, and loads U-Net models  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── main.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Serves as the primary entry point for executing analysis workflow  
├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── studies-usable/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Designated location of study folders  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── *studyID-DICOMname*/ # *sample study folder*  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── *DICOMname*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# *sample .DICOM file*  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── masks/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Contains masks corresponding to .DICOM file in the same directory  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── processed/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stores the output of intermediate processing stages  
├── notebooks/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Collection of .ipynb files (exploratory analysis and prototyping)  
├── models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Storage for trained machine learning models (.keras)  
├── tests/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Contains unit tests for verifying of individual module functionality  
├── config/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Repository for configuration files (e.g. thresholds and model paths)  
├── .gitignore  
├── requirements.txt  
└── README.md  


## Operational Procedures  
#### DICOM (and mask) File Placement:  
Deposit all serial high-resolution grayscale IVUS .DICOM files pertaining to a given study into a unique directory within the data/studies-usable/ directory.  
*For model training:* Place all corresponding masks within a directory named "masks" which resides within the same unique directory as the .DICOM file.  

#### Execution of Analysis:  
Open the integrated terminal within the IDE. Verify that the virtual environment is active. Subsequently, execute the primary script:  
*python src/main.py*  
The resulting output will be displayed in the console.  


## Prospective Enhancements
###### The following areas (separate from those noted at the top of this document) have been identified for future development and refinement:  

#### Robust & Sophisticated Phase Identification Model  
The development of more sophisticated machine learning models is planned for the precise detection of end-systolic and end-diastolic cardiac phases.  

#### Comprehensive Error Management  
Improvements in error handling mechanisms are anticipated to enhance resilience against corrupted or improperly formatted DICOM files.  

#### Dedicated Visualization Module  
The creation of a specialized module for the visualization of IVUS images, segmented lumens, waveforms, and computed stiffness results is planned.  

#### User Interface Development  
The design and implementation of a streamlined graphical user interface (GUI) are considered to facilitate more intuitive user interaction.  

#### Database Integration for Persistence  
The integration with a persistent and queryable database is proposed for the storage and retrieval of analysis results.

#### General Performance Optimization
Efforts will be directed towards optimizing the codebase to enhance the processing speed for large datasets.
