# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


#　--dataset：数据集的路径（利用search_bing_api.py创建的数据集）；
#　--encodings：面部编码将被写到该参数所指的文件中；
#　--detection-method：首先需要检测到图像中的面部，才能对其进行编码。两种面部检测方法为hog或cnn，因此　# 该参数只接受这两个值
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# imagePaths，获取每一张图像的路径
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []


# OpenCV中的颜色通道排列顺序为BGR，但dlib要求的顺序为RGB。对每一个图像进行编码。
 for (i, imagePath) in enumerate(imagePaths):
     # extract the person name from the image path
     print("[INFO] processing image {}/{}".format(i + 1,　len(imagePaths)))
     name = imagePath.split(os.path.sep)[-2]
     # load the input image and convert it from BGR (OpenCV ordering)
     # to dlib ordering (RGB)
   　 image = cv2.imread(imagePath)
    　rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
　　　　　# detect the (x, y)-coordinates of the bounding boxes
　　　　　# corresponding to each face in the input image
     boxes = face_recognition.face_locations(rgb,　model=args["detection_method"])
     # compute the facial embedding for the face
     encodings = face_recognition.face_encodings(rgb, boxes)
     # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()





