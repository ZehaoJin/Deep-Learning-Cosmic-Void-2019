# Deep-Learning-Cosmic-Void-2019

This project uses ##Convolutional Neural Network## to spot cosmic voids/dark matter from a given galaxy catalog!

Collaborator: [Joshua Yao-Yu Lin](https://github.com/joshualin24), PhD student in university of illinois urbana champaign
Data from: Arka Banerjee, PhD student in university of illinois urbana champaign

Codes are ran on NCSA [Hardware-Accelerated Learning (HAL) cluster](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster)-This work utilizes resources supported by the National Science Foundation’s Major Research Instrumentation program, grant #1725729, as well as the University of Illinois at Urbana-Champaign


![](counts_result.png)
![](result/demofigure_2.png)


# Procedure
Arka ran cosmology simulations on a 1024 * 1024 * 1024 3D box to simulate an area of cosmos, and generate galaxy catalogs out of them. He then ran the "old-school" astronomy void finder to detect cosmic voids inside that area, so that we have a catalog of galaxys, as well as a corrsponding catalog of voids!

We cut this 1024 * 1024 * 1024 box into about 10 thousands small 224 * 224 * 224 boxs. For each small boxes, we then apply “Cloud-In-Cell” interpolation on the galaxy catalog to generate a real "image" (like convolution) instead of just a scatter plot. Then this small convoluted box is squeezed in Z direction(simply sum along z-axis), results in a 2D 224 * 224 galaxy image. Similiar steps are also applied to the void catalog, giving us a 2D 224 * 224 void image.

Now we have 10 thousand galaxy 224 * 224 image and corresponding 10 thousand void 224 * 244 image ready for training! We seperated 80% of data to be training set, and remaining 20% as test set, and send them into Convolutional Neural Network.
- We mainly use pre-trained ResNet(https://pytorch.org/hub/pytorch_vision_resnet/).
- But we rebuilt ##ResNet##'s last layer's architecture to be 56 * 56 (224 * 224 void image are further resized into 56 * 56 ones befored being trained)
- Therefore the modified ResNet will take in 224 * 224 galaxy image, and finally outputs its prediction on the 56 * 56 void image.
- Compare the loss between predicted void image and ground truth, ResNet is able to "learn" how do to it better next time.

# Result
![](result/demofigure_0.png)
![](result/demofigure_0.png)
![](result/demofigure_0.png)
![](result/demofigure_0.png)


