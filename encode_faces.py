from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# 对指定路径下的图像人脸进行编码，并存入到磁盘



#　--dataset：  数据集的路径（利用search_bing_api.py创建的数据集）；
#　--encodings：面部编码将被写到该参数所指的文件中；
#　--detection-method：首先需要检测到图像中的面部，才能对其进行编码。两种面部检测方法为hog或cnn，因此该参数只接受这两个值
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# 获取每一张图像的路径
imagePaths = list(paths.list_images(args["dataset"]))

# 把每一张图片的名称和编码放到两个数组中
knownEncodings = []
knownNames = []


# OpenCV中的颜色通道排列顺序为BGR，但dlib要求的顺序为RGB。对每一个图像进行编码。
 for (i, imagePath) in enumerate(imagePaths):
    # 从文件的路径中获取图片的名称
    name = imagePath.split(os.path.sep)[-2]
    
	# 从文件的路径中获取图片的名称
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
	# 在图片中框出人脸，一幅图像可能有多张人脸
    boxes = face_recognition.face_locations(rgb,　model=args["detection_method"])
     
	# 对图片中框处的每张人脸进行编码
    encodings = face_recognition.face_encodings(rgb, boxes)
     
	# 把编码结果放到数组中
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
    
	# 结果存到磁盘
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()





