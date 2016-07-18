# Posture Recognition
---------------------
**Keywords: Python, Deep Learning**
  
This will train a model which used in my university graduate project "**Human Posture Recognition Based on Neural Network in Robot Controlling**".  
In 2013, I trained the model that using C++ and OpenNN, and I got **84.04%** classification accuracy.   
Now I train this model again that using Python, Keras, and Theano as backend, then I get a better classification accuracy which is **99.04%**.  
  
  
## Requirement
--------------
1. Python 3.5
2. Keras
3. Theano / TensorFlow (as Keras backend)

## Data
-------
**Input** :  
We got the body coordinate from Kinect, then using Dot Product to find the angle of shoulder and elbow. 
    * angle of left shoulder
    * angle of right shoulder
    * angle of left elbow
    * angle of right elbow
**Output** :  
classified to 9 classes, example:
    * 1 => 1 0 0 0 0 0 0 0 0 
    * 5 => 0 0 0 0 1 0 0 0 0
    * 9 => 0 0 0 0 0 0 0 0 1
