# TrafficLightClassifier

This is a traffic light image classifier.
It takes traffic light images as inputs and outputs the labels(red, yellow or green).
I composed tiny dataset with my codes for you to run quickly. But I used more large dataset to build the classifier. So please feel free to try this. This is Udacity's "Intro to Self-Driving Car" course project. I got more than 95 accuracy(Contained dataset is too small so accuracy is different).


### Explanation
This algorithm has two steps to classify the traffic light image. The first step use brightness feature. The most bright area indicate what color are lighted. The second step use color feature. I convert the RGB image to HSV image, filter the noisy pixel and extract the Hue channel. Hue channel indicate the color. So I use it to predict the traffic light color.
