# Side Circuit
Side Circuit is a circuit optimizer and recommender.  The project has the following goal 
- given a partially defined circuit and requirements predict:
  - optimal circuit component parameters
  - missing circuit components

# Project Structure
- app: contains the front end application (flutter)
- ml: contains the machine learning development (python)

# Problem Statement
How to teach a machine learning model how to be a circuit simulator?

The primary focus of the project is on the machine learning model.  The model must learn to mimic a circuit simulator to some degree.  

# Early Attempt: Directly Model a Circuit Simulator as an Optimizer
An early attempt was made at directly modeling a circuit simulator as in an optimizer (pytorch was used with gradient descent).  The learned parameters being optimized were all voltage and current signals and circuit element parameters (such as resistance).  Any known values had there optimizer parameters frozenThis was a good exercise for my linear algebra and pytorch skills.  However, it had some shortcomings:
   - the model was very sensitive to large differences in values very typical of electronic circuits and signal processing.  For example, 1M ohm resistor and 5 V voltage source attemping to find the a tiny current of 5 uA.  It was difficult to get the model to converge to a solution without some sort of normalization.  Dynamically normalizing as the model optimized was unpredictable.  This will less of a problem with a pretrained model due to having predetermined base values from the dataset for normalization.
   - the model was very slow to train (aka reach a circuit solution)

# Current Approach Being Developed: Pretrain a Model on a Circuit Simulator
The current approach is to pretrain a model to predict values of a circuit.  This will involves
- creating a dataset of circuits and their solutions
  - the dataset will be created by running a circuit simulator on a large number of circuits with a predefined set of parameters and logging the results.
- preprocessing the dataset to be compatible with the model
  - this will involve normalizing the data and creating a graph representation of the circuit
- selecting an appropriate model architecture
  -The model will need to to condider all elements of the circuit simultaneously.  A Graph Transformer may be a good first try.
- training the model on the dataset
- evaluating the model on a test set

