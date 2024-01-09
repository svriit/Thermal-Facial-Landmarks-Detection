# IS Project: Thermal Facial Landmark Dataset with Visual Pairs on Low Resolution Images
The dataset contains 2,556 thermal-visual image pairs of 142 subjects with manually annotated face bounding boxes and 54 facial landmarks. The dataset was constructed from our large-scale [SpeakingFaces dataset](https://github.com/IS2AI/SpeakingFaces).

<img src= "https://raw.githubusercontent.com/IS2AI/thermal-facial-landmarks-detection/main/figures/example.png"> 


## The facial landmarks are ordered as follows:

<img src= "https://raw.githubusercontent.com/IS2AI/thermal-facial-landmarks-detection/main/figures/land_conf.png"> 

## Download the repository:
```
git clone https://github.com/AiYogi1234/Ankit_Ranjan_ISProject.git
```
## Requirements
- imutils
- OpenCV
- NumPy
- Pandas
- dlib
- Tensorflow 2

To install the necessary packages properly, we ask you to walk through these two tutorials:
1. [How to install TensorFlow 2.0 on Ubuntu](https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/).
2. [Install dlib (the easy, complete guide)](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/).

## Data preparation
Download the dataset from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing). Save this dataset to dataset folder in the cloned repository.

- To Generate training, validation, and testing XML files for dlib shape predictor
```
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set train
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set val 
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set test
```

## Training and testing dlib shape predictor
- To manually tune parameters of the model:
```
python train_dlib_predictor.py --training dataset/gray/train/dlib_landmarks_train.xml --validation dataset/gray/val/dlib_landmarks_val.xml
```
- To search optimal parameters via grid search:
```
python dlib_grid_search.py
```
- To optimize parameters via dlib's global optimizer:
```
python dlib_global_optimizer.py
```
- Testing the trained model:
```
python test_dlib_predictor.py --testing dataset/gray/test/dlib_landmarks_test.xml --model models/dlib_landmarks_predictor.dat
```

- Make predictions on images:
```
python dlib_predict_image.py --images PATH_TO_IMAGES --models  models/ --upsample 1
```

- Make predictions on High Resolution images provided in the repository:
```
python dlib_predict_image.py --images test_image --models  models/ --upsample 1
```



- Make predictions on Low Resolution images:
```
python dlib_predict_image.py --images test_image_low_res --models  models/ --upsample 1
```
![image](https://user-images.githubusercontent.com/62781709/231472022-7aa5fe1e-b3ba-4d71-967d-ce603689f35f.png)

- Make predictions on a video:
```
python dlib_predict_video.py --input PATH_TO_VIDEO --models  models/ --upsample 1 --output output.mp4
```

- Make predictions on a video:
```
python dlib_predict_video.py --input video2mp4.mp4 --models  models/ --upsample 1 --output output.mp4
```


4. **U-net model**
```
python unet_predict_image.py --dataset dataset/gray/test --model  models/ 
```


## For dlib face detection model (HOG + SVM)
- Training the model:
```
python train_dlib_face_detector.py --training dataset/gray/train/dlib_landmarks_train.xml --validation dataset/gray/val/dlib_landmarks_val.xml
```
- Make predictions on images:
```
python dlib_face_detector.py --images dataset/gray/test/images --detector models/dlib_face_detector.svm
```

## To visualize dataset
- Thermal images with bounding boxes and landmarks:
```
python visualize_dataset.py --dataset dataset/ --color iron --set train
```

![image](https://user-images.githubusercontent.com/62781709/231471709-88b56c8d-4518-4ffb-98d3-b0fc3d820983.png)


- Thermal-Visual pairs
```
python visualize_image_pairs.py --dataset dataset/ --color iron --set train

```
![image](https://user-images.githubusercontent.com/62781709/231471470-95b9e7bb-b13a-4a5e-a7dd-e2b0d4204682.png)

- Make Patch Detection:
```
python ISP_patches.py

```

![image](https://user-images.githubusercontent.com/62781709/231469947-f68659ed-b684-4c2b-b74c-d62da99981ee.png)
![image](https://user-images.githubusercontent.com/62781709/231470549-21184315-9058-454d-908e-2f91e8cf7b37.png)


### Credits :
'''
Some part of this repository is obtained from the following repository : https://github.com/IS2AI/thermal-facial-landmarks-detection/
'''
