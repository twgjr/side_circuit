# Side Circuit
Side Circuit is a circuit optimizer and recommender.  The project has the following goal 
- given a partially defined circuit and requirements predict:
  - optimal circuit component parameters
  - missing circuit components

# Project Structure
- flutter: contains the front end application
- ml: contains the machine learning development (python)

# Problem Statement
How to teach a machine learning model how to be a circuit simulator?

The primary focus of the project is on the machine learning model.  The model must learn to mimic a circuit simulator to some degree.  

# Modeling a Circuit Simulator as an Optimizer
An early attempt was made at directly modeling a circuit simulator in an optimizer (pytorch was used with gradient descent).

The learned parameters being optimized in the direct optimizer were all voltage and current signals and circuit element parameters (such as resistance).  Any known values had there optimizer parameters frozenThis was a good exercise for my linear algebra and pytorch skills.  However, it had some shortcomings:
   - the model is very sensitive to large differences in values very typical of electronic circuits and signal processing.  For example, 1M ohm resistor and 5 V voltage source attemping to find the a tiny current of 5 uA.  It was difficult to get the model to converge to a solution without some sort of normalization.  Dynamically normalizing as the model optimized was unpredictable.  This will less of a problem with a pretrained model due to having predetermined base values from the dataset for normalization.
   - the model was very slow to train (aka reach a circuit solution)

# How to run the project
- flutter:  Ignore for now. It is run at the console with python.

- ml:
  - the program was run in visual studio code with the python extension and conda environment
  - set working directory to ml
  - install pytorch and networkx in conda
  - run the unittests in the test folder with test_learning.py


