https://sites.google.com/view/cbm-anomaly-detection



his site contains a dataset for Condition Based Maintenance of Naval Propulsion Systems, which can be adopted for both Supervised and Unsupervised Data Analysis taks, together with some general informations related to how it was collected. As this dataset was submitted to OpenML dataset repository, it is in arff format. Moreover, we report links to the description of previous versions of this dataset.

Downloads
Data submitted to the OpenML dataset repository.

Old Versions of the dataset:

V1

V2

Data Description
Data Set Name: Condition Based Maintenance of Naval Propulsion Plants Data Set

Abstract:

The behavior and interaction of the main components of Ship Propulsion Systems cannot be easily modeled with a priori physical knowledge, considering the large amount of variables influencing them. Data-Driven Models (DDMs), instead, exploit advanced statistical techniques to build models directly on the large amount of historical data collected by the modern on-board automation systems, without requiring any a priori knowledge. DDMs are extremely useful when it comes to continuously monitor the propulsion equipments to avoid Preventive or Corrective Maintenance and take decisions based on the actual condition of the propulsion plant. Unfortunately, DDMs need a large amount of data to achieve satisfying performances. While sensor data are cheap and easy to collect, label them with the actual state of decay of a component can be quite expensive and in some cases unfeasible.

In this paper, the authors investigate the problem of performing Condition-Based Maintenance though the use of DDMs. First, state-of-the-art supervised learning techniques are adopted, which require a large amount of labeled sensor data in order to be deployed. Then, an unsupervised learning approach is developed as it allows to minimize the feedback of the operators in labeling the sensor data.

A Navy vessel, characterised by a combined diesel-electric and gas propulsion plant, has been exploited to show the effectiveness of the proposed approaches.

Source: Numerical simulator of a Navy frigate

Data Type:

The dataset is composed by 30 Double-type features, respectively divided in:

A 25-feature vector containing the vessel relevant features:

Lever (lp) [ ]

Speed [knots]

Gas Turbine shaft torque (GTT) [kN m]

Gas Turbine Speed (GT rpm) [rpm]

Controllable Pitch Propeller Thrust stbd (CPP T stbd)[N]

Controllable Pitch Propeller Thrust port (CPP T port)[N]

Shaft Torque port (Q port) [kN]

Shaft rpm port (rpm port)[rpm]

Shaft Torque stbd (Q stdb) [kN]

Shaft rpm stbd (rpm stbd) [rpm]

HP Turbine exit temperature (T48) [C]

Generator of Gas speed (GG rpm) [rpm]

Fuel flow (mf) [kg/s]

ABB Tic control signal (ABB Tic) []

GT Compressor outlet air pressure (P2) [bar]

GT Compressor outlet air temperature (T2) [C]

External Pressure (Pext) [bar]

HP Turbine exit pressure (P48) [bar]

TCS tic control signal (TCS tic) []

Thrust coefficient stbd (Kt stbd) []

Propeller rps stbd (rps prop stbd) [rps]

Thrust coefficient port (Kt port) []

Propeller rps port (rps prop port) [rps]

Propeller Torque port (Q prop port) [Nm]

Propeller Torque stbd (Q prop stbd) [Nm]

Propeller Thrust decay state coefficient (Kkt)

Propeller Torque decay state coefficient (Kkq)

Hull decay state coefficient (Khull)

GT Compressor decay state coefficient (KMcompr)

GT Turbine decay state coefficient (KMturb)

Task: Regression, Classification, Anomaly Detection

Attribute Type: Double

Area: Condition-Based Maintenance in Naval Domain

Format Type: Each feature vector is a row on the text file (30 elements in each row)

Does your data set contain missing values? No

Number of Instances (records in your data set): 455109

Number of Attributes (fields within each record): 30

Relevant Information:

Use of this dataset in publications must be acknowledged by referencing the following publication.

This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

Attribute Information:

Relevant Papers:

Author: Cipollini,F. and Oneto,L. and Coraddu, A. and Murphy, A.J. and Anguita, D. - Journal: Reliability Engineering & System Safety - Title: Condition-Based Maintenance of Naval Propulsion Systems: Data Analysis with Minimal Feedback - Volume: Under review - Year: 2017.

Author: Cipollini,F. and Oneto,L. and Coraddu, A. and Murphy, A.J. and Anguita, D. - Journal: Ocean Engineering - Title: Condition-Based Maintenance of Naval Propulsion Systems with Supervised Data Analysis - Volume: Under review - Year: 2017.

Author: Coraddu, A. and Oneto, L. and Ghio, A. and Savio, S. and Anguita, D. and Figari, M. - Journal: Proceedings of the Institution of Mechanical Engineers Part M: Journal of Engineering for the Maritime Environment - Number: 1 - Pages: 136-153 - Title: Machine learning approaches for improving condition-based maintenance of naval propulsion plants - Volume: 230 - Year: 2016

Citation Requests / Acknowledgements:

Author: Cipollini,F. and Oneto,L. and Coraddu, A. and Murphy, A.J. and Anguita, D. - Journal: Ocean Engineering - Title: Condition-Based Maintenance of Naval Propulsion Systems with Supervised Data Analysis - Volume: Under review - Year: 2017.

Author: Coraddu, A. and Oneto, L. and Ghio, A. and Savio, S. and Anguita, D. and Figari, M. - Journal: Proceedings of the Institution of Mechanical Engineers Part M: Journal of Engineering for the Maritime Environment - Number: 1 - Pages: 136-153 - Title: Machine learning approaches for improving condition-based maintenance of naval propulsion plants - Volume: 230 - Year: 2016

Francesca Cipollini(1), Luca Oneto(1), Andrea Coraddu(2), Alan J. Murphy(3), Davide Anguita(1)

1 - DIBRIS - University of Genoa

2 - Naval Architecture, Ocean & Marine Engineering, Strathclyde University

3 - School of Marine Science and Technology, Newcastle University
