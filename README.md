# Melon-HDSS

Decision Suport System for detecting the Ripeness of Melons Fruit. This App is a master degree project that aims to develop a system for detecting the correct harvest time. The farmers currently harvest melons based on their experience, and melon cannot maintain their quality after harvesting. Thus, the melon needs to be harvested at the right time that will not affect the maturity of the fruit harvested earlier. Melon harvesting is a daily task; Melon is not mature on the same day even though the melon plant is planted in the same period due to genetics and environment. The decision support system should detect and classify the ripeness level of the fruit on the tree. The system will categorize the maturity level into three categories: Ripe, About to Ripe, or Under Ripe (within a rate from 0 to 10 obtained from fuzzy inference system result). The ripeness levels are confirmed by the expert, depending on the skin color of the fruit. As a result, we get 100% accuracy in classifying each category using phone camera images and video. Furthermore, this decision support system can be implemented in melon’s harvesting robot. The melon used in this study is honeydew Cucumis melo L, var. Alisha F1.


## The System built with Python 3.7, and the GUI using Kivy 2.0.0. Methodology used are as follow:
1. Detecting the melon fruits in the middle of the image or video frame by OpenCv dnn based on the Frozen Tensorflow SSD-Mobilenet Pretrained model.
2. Using HSV color space to segment the detected melon image and extract the melon skin color.
3. Using the extracted skin-color to read the R, G, and B layers, then normalizing the layers pixels intensities.
4. Find the mean for each normalized layer.
5. Using the MEANs in #4 as inputs to Fuzzy-Logic to classify the ripeness level based on reference values.

## Requirments
Packges should be intalled with pip to run the Melon harvesting system:
	"kivy",
	"opencv-python",
	"numpy",
	"matplotlib",
	"filetype",
	"scikit-fuzzy"
 ## Usage 
 After installing the requirments libs you can run the **main.py**
