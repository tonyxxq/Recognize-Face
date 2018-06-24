# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

# --encodings：包含面部编码的pickle文件的路径；
# --image：需要进行面部识别的图像；
# --detection-method：这个选项应该很熟悉了。可以根据系统的能力，选择hog或cnn之一。追求速度的话就选择hog，追求准确度就选择cnn。
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,　help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,　help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",　help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,　 model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []


# 尝试利用face_recognition.compare_faces将输入图像中的每个面部（encoding）对应到已知的编码数据集（保存在data["encodings"]中）上。
# 该函数会返回一个True/False值的列表，每个值对应于数据集中的一张图像。对于我们的侏罗纪公园的例子，数据集中有218张图像，因此返回的列表将包含218个布尔值。
# compare_faces函数内部会计算待判别图像的嵌入和数据集中所有面部的嵌入之间的欧几里得距离。
# 如果距离位于容许范围内（容许范围越小，面部识别系统就越严格），则返回True，表明面部吻合。否则，如果距离大于容许范围，则返回False表示面部不吻合。
for encoding in encodings:
    # attempt to match each face in the input image to our known
    # encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
　　　　# check to see if we have found a match
    if True in matches:
        # find the indexes of all matched faces then initialize a
        # dictionary to count the total number of times each face
        # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)


# 循环每个人的边界盒和名字，然后将名字画在输出图像上以供展示之用：
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    # 如果边界盒位于图像顶端，则将文本移到边界盒下方，否则文本就被截掉了。
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)






