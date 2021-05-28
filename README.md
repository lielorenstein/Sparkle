Sparkle is a parking lot detection system, upload a picture of a parking lot to Sparkle and it will tell you which spaces are empty and which are occupided. 
Sparkle was created with the following tools: Python, Google Colab, Keras , Tenserflow, Torch/PyTorch, Roboflow, Kaggle, and Coco.
While creating sparkle we developed 3 deep learning / computer vision models with the following algorithms: CNN, R-CNN, and Yolo.

CNN is a class of deep neural networks, most commonly applied to analyze visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation equivariant responses known as feature maps. Counter-intuitively, most convolutional neural networks are only equivariant, as opposed to invariant, to translation. 

CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns to optimize the filters (or kernels) through automated learning, whereas in traditional algorithms these filters are hand-engineered. This independence from prior knowledge and human intervention in feature extraction is a major advantage.

Using CNN was one of our starting points, we developed a simple model that with a dataset of pictures of cars and empty spaces can identify other picutres. With the best version of this model we recieved 86% accuracy.

Another model we built used the Yolo alogorithm. YOLO is a clever convolutional neural network (CNN) for doing object detection in real-time. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
YOLO is popular because it achieves high accuracy while also being able to run in real-time. The algorithm “only looks once” at the image in the sense that it requires only one forward propagation pass through the neural network to make predictions. After non-max suppression (which makes sure the object detection algorithm only detects each object once), it then outputs recognized objects together with the bounding boxes.
With YOLO, a single CNN simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. This model has a number of benefits over other object detection methods:
-YOLO is extremely fast
-YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.
-YOLO learns generalizable representations of objects so that when trained on natural images and tested on artwork, the algorithm outperforms other top detection methods.

In order to run YOLO we found a parking lot dataset from Kaggle (https://www.kaggle.com/duythanhng/parking-lot-database-for-yolo) that was already prepared for YOLO (with bounding boxes). After finding the dataset, we uploaded it to Roboflow in order to prepare it for the algorithm. After uploading the dataset we could already see two different colors of bonudry boxes for empty and occupied spaces.

![image](https://user-images.githubusercontent.com/57219508/119971668-f1b5a380-bfb9-11eb-9ca2-20e7b3ac57d3.png)

Next we took the Google Colab for scaled yolov4 from the Roboflow computer vision library and ran it our dataset. The results were picutres with boundry boxes that had labels of zero(empty) and one (occupided).

The picutres before the algorithm:

![test_batch0_pred](https://user-images.githubusercontent.com/57219508/119972037-65f04700-bfba-11eb-83be-ddbc54b48873.jpg)

The picutes after:

![test_batch0_gt](https://user-images.githubusercontent.com/57219508/119972074-74d6f980-bfba-11eb-833f-e1e658a57142.jpg)

Unfortunately with the time limit we had we weren't able to create a new model to identify all the open and taken spaces using the output weights from the yolov4 that we ran, but it can certainly be done.

The final algroithm that we used was R-CNN. The problem with the CNN approach is that the objects of interest might have different spatial locations within the image and different aspect ratios. Hence, you would have to select a huge number of regions and this could computationally blow up.

Given an input image, R-CNN begins by applying a mechanism called Selective Search to extract regions of interest (ROI), where each ROI is a rectangle that may represent the boundary of an object in image. Depending on the scenario, there may be as many as two thousand ROIs. After that, each ROI is fed through a neural network to produce output features. For each ROI's output features, a collection of support-vector machine classifiers is used to determine what type of object (if any) is contained within the ROI.

We specifically used Mask R-CNN. While previous versions of R-CNN focused on object detection, Mask R-CNN adds instance segmentation. Mask R-CNN also replaced ROIPooling with a new method called ROIAlign, which can represent fractions of a pixel

We used the already built coco map indexing to indetify cars and added the ability to indenify empty spaces (objects that aren't cars).

The input picture:

![image](https://user-images.githubusercontent.com/57219508/119973098-b87e3300-bfbb-11eb-951d-3c95eb8be8fb.png)

The ouput:

![image](https://user-images.githubusercontent.com/57219508/119973148-ccc23000-bfbb-11eb-8f6b-80553d91fb35.png)
