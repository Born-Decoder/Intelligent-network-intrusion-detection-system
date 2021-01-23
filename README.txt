___________________________________________________________________________________________________________________________________________________________________________

Author's note - Hi y'all, I have made this code totally public and feel free to use it

____________________________________________DOCUMENTATION_______________________________________________

Overall Requirements - Python (3.5, keras, tensorflow, pandas, sklearn, numpy, matplotlib, keras,
			tensorflow, pandas, sklearn, numpy, matplotlib, random), Wireshark.


----------------------------------------------Module - 1------------------------------------------------

Aim - Converting .pcap file to .csv

Requirements - Wireshark

Input - 'allpacks.pcap'

Steps:

 1. Open the file 'allpacks.pcap' or the preferred file of a name. 

 2. It is recommended that you use wireshark to open it. If not you can use any other packet analyser
    that allows you to convert the contents of the packet into a .csv file format

 3. Now select all the packets and export as .csv file which we will name as 'csvallpack.csv'

Output - 'csvallpack.csv'

----------------------------------------------Module - 2------------------------------------------------

Aim - to normalize and encode the input .csv (norm.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - ipaddress (Python package)
	     - sklearn with preprocessing (Python package)

Input - 'csvallpack.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing
	- "ipaddress" for handling ipaddress datatypes in the input .csv file
	- "sklearn's preprocessing" for 'one-hot encodiing'

 2. Use pd.read_csv to read the extracted data from 'csvallpack.csv' and store in 'data'

 3. 'nrow' for the number of rows in 'data'

 4. Convert the columns data['Source'] and data['Destination'] of the data frame into int values for 
    future ML use from 'ipaddress'- type values using int(ipaddress.ip_address())

 5. Min - max normalize the columns - 'Source'. 'Destination', 'Time', 'Length' using the formula:
	(x - min)/(max - min). 
    Note that we use min-max normalization here, since ML algorithms generally require the inputs to
    be ranged from 0 to 1.

 6. Store the features 'Time', 'Source', 'Destination', 'Protocol' and 'Length' as a separate data frame

 7. Now we can see that 'Protocol' column is categorical. Hence we use one-hot encoding to get separate
    columns for each category of protocol.
	Use pd.get_dummies() to encode and pd.concat() to concatenate the columns to the dataframe.
 
 8. Write the output data frame - 'df' to 'normout.csv' without indexing using df.to_csv()

Output - 'normout.csv'

----------------------------------------------Module - 3------------------------------------------------

Foreword - Note that pseudorandom values can be manually inserted also. It can be done by creating a 
	   column named 'Safe' in 'normout.csv' and setting 0's and 1's in it. If you do so, this module
	   can be skipped.

Aim - To insert safe/unsafe values into the data (pseudo.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - random (Python package)

Input - 'normout.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing 
	- "random" for generating pseudorandom numbers

 2. Use pd.read_csv to read the data from 'normout.csv' and store in 'data'

 3. 'nrow' for the number of rows in 'data'

 4. Generate a list of random values of 0 or 1 with random.randint() with the range of nrow

 5. Add the list to a new column data['Safe']

 6. Write the data frame 'data' to 'normout.csv' without indexing using data.to_csv() 

Output - 'normout.csv'

----------------------------------------------Module - 4------------------------------------------------

Aim - to create an train an ANN (nn.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - tensorflow (Python package)
	     - keras - tensorflow (Python package)
	     - numpy (Python package)
	     - matplotlib (Python package)
	     - train_test_split - sklearn.model_selection (Python package)

Input - 'normout.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing
	- "sklearn's model_selection" for 'train_test_split'
	- "tensorflow" and "keras" for ANN
	- "numpy" for basic array manips and ops
	- "matplotlib.pyplot" for plots and graphs

 2. Use pd.read_csv to read the data from 'normout.csv' and store in 'data'

 3. Store the required x and y columns in separate data frames

 4. Split the x and y into training x, training y, testing x, testing y in 80/20 portions using the
    function 'train_test_split' 

 5. Create a ANN model keras.Sequential([]) with the required architecture:
	keras.layers.Dense() - to create a NN layer with specified number of nodes and activation fn.
	
	Note: Select an appropriate activation function based on the inputs and outputs that are passed
	into the NN layers. 'relu' is a rectified linear activation function used to boost inputs when 
	they are positive, 'softmax' is a standard and widely used activation function for various 
	layers of NNs, 'sigmoid' is used since the output is basically binary in nature (0/1) and it 
	helps to convert outputs from previous layers into 0/1 values.

 6. Compile the model using 'model.compile()' with required optimizers, loss functions and accuracy 
    metrics
    Note: 'Adam' optimizer is used since it converges quickly than traditional gradient descent
    algorithms. 'binary_crossentropy' is used as the loss function since it is suited for binary
    classification problems.

 7. Train the model using 'model.fit()' and specify the x and y training sets after converting them
    to correct shapes (this is done by using '.as_matrix()'), with the specified epochs parameters

 8. Find the test_loss and test_accuracy by using model.evaluate(), with test_x and test_y as
    parameters

Output - Trained model and its performance metrics

--------------------------------------------------------------------------------------------------------

Aim - to create an train a CNN (cnn.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - tensorflow (Python package)
	     - keras - tensorflow (Python package)
	     - numpy (Python package)
	     - matplotlib (Python package)
	     - train_test_split - sklearn.model_selection (Python package)

Input - 'normout.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing
	- "sklearn's model_selection" for 'train_test_split'
	- "tensorflow" and "keras" including "Conv2D", "Dense" and "Flatten"(if required) for CNN
	- "numpy" for basic array manips and ops
	- "matplotlib.pyplot" for plots and graphs

 2. Use pd.read_csv to read the data from 'normout.csv' and store in 'data'

 3. Store the required x and y columns in separate data frames

 4. Split the x and y into training x, training y, testing x, testing y in 80/20 portions using the
    function 'train_test_split' 

 5. Convert the data frames of test_x, test_y, train_x and train_y into numpy arrays using df.values
    and use array[:,0] to reshape the y values in both the testing and training sets

 6. Create a Sequential model for CNN using Sequential() with the required architecture:
      Use model.add() to add layers in the neural network.
        --Conv2D to add convolutional layers in the NN architecture
        --Dense to add dense ANN layers in the architecture
    Note: Use appropriate activation functions for the layers as specified previously

 7. Compile the model using 'model.compile()' with required optimizers, loss functions and accuracy 
    metrics
    Note: 'Adam' optimizer is used since it converges quickly than traditional gradient descent
    algorithms. 'binary_crossentropy' is used as the loss function since it is suited for binary
    classification problems.

 8. Train the model using 'model.fit()' and specify the x and y training sets after converting them
    to correct shapes, with the specified epochs parameters

 9. Find the test_loss and test_accuracy by using model.evaluate(), with test_x and test_y as
    parameters

Output - Trained model and its performance metrics

--------------------------------------------------------------------------------------------------------

Aim - to create an train a RNN (rnn.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - tensorflow (Python package)
	     - keras - tensorflow (Python package)
	     - numpy (Python package)
	     - matplotlib (Python package)
	     - train_test_split - sklearn.model_selection (Python package)

Input - 'normout.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing
	- "sklearn's model_selection" for 'train_test_split'
	- "tensorflow" and "keras" including "Conv2D", "Dense", "Reshape" and "Flatten" for RNN
	- "numpy" for basic array manips and ops
	- "matplotlib.pyplot" for plots and graphs

 2. Use pd.read_csv to read the data from 'normout.csv' and store in 'data'

 3. Store the required x and y columns in separate data frames

 4. Split the x and y into training x, training y, testing x, testing y in 80/20 portions using the
    function 'train_test_split' 

 5. Convert the data frames of test_x, test_y, train_x and train_y into numpy arrays using
    df.as_matrix()

 6. Create a Sequential model for RNN using Sequential() with the required architecture:
      Use model.add() to add layers in the neural network.
        --Dense to add dense ANN layers in the architecture
	--Reshape() to rehape the inputs to the next layer (if required)
	--LTSM() to add LTSM RNN layers to the architecture
	--Flatten() to flatten the inputs from the LTSM layer to Dense layers
    Note: Use appropriate activation functions for the layers as specified previously

 7. Compile the model using 'model.compile()' with required optimizers, loss functions and accuracy 
    metrics
    Note: 'Adam' optimizer is used since it converges quickly than traditional gradient descent
    algorithms. 'binary_crossentropy' is used as the loss function since it is suited for binary
    classification problems.

 8. Train the model using 'model.fit()' and specify the x and y training sets after converting them
    to correct shapes, with the specified epochs parameters

 9. Find the test_loss and test_accuracy by using model.evaluate(), with test_x and test_y as
    parameters

Output - Trained model and its performance metrics

--------------------------------------------------------------------------------------------------------

Aim - to create an train a DBN (nn.py)

Requirements - Python (3.5)
             - pandas (Python package)
	     - tensorflow (Python package)
	     - keras - tensorflow (Python package)
	     - numpy (Python package)
	     - matplotlib (Python package)
	     - train_test_split - sklearn.model_selection (Python package)

Input - 'normout.csv'

Steps:

 1. Import the required packages for the module:
	- "pandas" for creating dataframes and .csv parsing
	- "sklearn's model_selection" for 'train_test_split'
	- "tensorflow" and "keras" for ANN
	- "numpy" for basic array manips and ops
	- "matplotlib.pyplot" for plots and graphs

 2. Use pd.read_csv to read the data from 'normout.csv' and store in 'data'

 3. Store the required x and y columns in separate data frames

 4. Split the x and y into training x, training y, testing x, testing y in 80/20 portions using the
    function 'train_test_split' 

 5. Convert the data frames of test_x, test_y, train_x and train_y into numpy arrays using df.values
    and use array[:,0] to reshape the y values in both the testing and training sets

 6. Use SupervisedDBNClassification() for architecting DBN with the desired structure, learning rate,
    epochs, batch size and activation functions

 7. Train the model using 'model.fit()' and specify the x and y training sets after converting them
    to correct shapes

 8. Use model.save() to save the model and SupervisedDBNClassification.load() to restore it

 9. Find the test_loss and test_accuracy by using model.predict() and accuracy_scre(), with test_x 
    and test_y as parameters

Output - Trained model and its performance metrics

--------------------------------------------------------------------------------------------------------
________________________________________________________________________________________________________

	








