# Network Traffic Analysis and Machine Learning

## Overview

This project focuses on analyzing network traffic data using Python and various machine learning techniques. It involves converting packet capture (.pcap) files to .csv format, normalizing and encoding the data, inserting safe/unsafe values, and training different types of neural networks for classification.

## Requirements

- Python 3.5
- Libraries: keras, tensorflow, pandas, sklearn, numpy, matplotlib, ipaddress, random
- Wireshark

## Module 1: Converting .pcap file to .csv

### Aim
Convert .pcap file to .csv format.

### Requirements
Wireshark

### Input
'allpacks.pcap'

### Steps
1. Open 'allpacks.pcap' using Wireshark.
2. Select all packets and export as 'csvallpack.csv'.

### Output
'csvallpack.csv'

## Module 2: Normalizing and Encoding the Input .csv (norm.py)

### Aim
Normalize and encode the input .csv.

### Requirements
Python (3.5), pandas, ipaddress, sklearn.preprocessing

### Input
'csvallpack.csv'

### Steps
1. Import required packages.
2. Read 'csvallpack.csv' using pd.read_csv.
3. Normalize columns 'Source', 'Destination', 'Time', 'Length'.
4. Store features 'Time', 'Source', 'Destination', 'Protocol', 'Length'.
5. One-hot encode 'Protocol'.
6. Write output to 'normout.csv'.

### Output
'normout.csv'

## Module 3: Inserting Safe/Unsafe Values into the Data (pseudo.py)

### Aim
Insert safe/unsafe values into the data.

### Requirements
Python (3.5), pandas, random

### Input
'normout.csv'

### Steps
1. Import required packages.
2. Read 'normout.csv' using pd.read_csv.
3. Generate list of random 0s and 1s.
4. Add list to new column 'Safe'.
5. Write output to 'normout.csv'.

### Output
'normout.csv'

## Module 4: Training an ANN (nn.py)

### Aim
Create and train an Artificial Neural Network (ANN).

### Requirements
Python (3.5), pandas, tensorflow, keras, numpy, matplotlib, sklearn.model_selection

### Input
'normout.csv'

### Steps
1. Import required packages.
2. Read 'normout.csv' using pd.read_csv.
3. Split data into training and testing sets.
4. Create ANN model using keras.Sequential([]).
5. Compile model with appropriate settings.
6. Train model using model.fit().
7. Evaluate model performance using model.evaluate().

### Output
Trained model and its performance metrics.

## Module 5: Training a CNN (cnn.py)

### Aim
Create and train a Convolutional Neural Network (CNN).

### Requirements
Similar to Module 4.

### Input
'normout.csv'

### Steps
Similar to Module 4, with adjustments for CNN architecture.

### Output
Trained model and its performance metrics.

## Module 6: Training a RNN (rnn.py)

### Aim
Create and train a Recurrent Neural Network (RNN).

### Requirements
Similar to Module 4.

### Input
'normout.csv'

### Steps
Similar to Module 4, with adjustments for RNN architecture.

### Output
Trained model and its performance metrics.

## Module 7: Training a DBN (dbn.py)

### Aim
Create and train a Deep Belief Network (DBN).

### Requirements
Similar to Module 4.

### Input
'normout.csv'

### Steps
Similar to Module 4, with adjustments for DBN architecture.

### Output
Trained model and its performance metrics.

---

Feel free to adjust the instructions and details based on your specific project requirements and preferences.
