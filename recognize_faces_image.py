import face_recognition
import argparse
import pickle
import cv2


# 找出图像中的人脸，并进行识别




# --encodings：包含面部编码的pickle文件的路径；
# --image：    需要进行面部识别的图像；
# --detection-method：选择hog或cnn之一。追求速度的话就选择hog，追求准确度就选择cnn。
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,　help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,　help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",　help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())



# 加载已经存储的人脸识别编码数据
data = pickle.loads(open(args["encodings"], "rb").read())

# 加载需要识别的图像，并转换为RGB通道
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测人脸的位置，并进行编码
boxes = face_recognition.face_locations(rgb,　 model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# 保存识别出的人脸的名称
names = []


# 尝试利用face_recognition.compare_faces将输入图像中的每个面部（encoding）对应到已知的编码数据集（保存在data["encodings"]中）上。
# 该函数会返回一个True/False值的列表，每个值对应于数据集中的一张图像。对于我们的侏罗纪公园的例子，数据集中有218张图像，因此返回的列表将包含218个布尔值。
# compare_faces函数内部会计算待判别图像的嵌入和数据集中所有面部的嵌入之间的欧几里得距离。
# 如果距离位于容许范围内（容许范围越小，面部识别系统就越严格），则返回True，表明面部吻合。否则，如果距离大于容许范围，则返回False表示面部不吻合。
for encoding in encodings:
    # 当前人脸的编码和库里边的所有的人脸编码进行比对，每一次比对都会返回True，False
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    
	# 如果距离位于容许范围内（容许范围越小，面部识别系统就越严格），则返回True，表明面部吻合。否则，如果距离大于容许范围，则返回False表示面部不吻合。
	name = "Unknown"
    if True in matches:
        # 把匹配到的人脸的index筛选出来
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # 查询出匹配到的人脸的名称
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        # 找到匹配度最高的人的名称，作为最终匹配结果
        name = max(counts, key=counts.get)

    # 添加识别出的人的名称
    names.append(name)


# 循环每个人的边界盒和名字，然后将名字画在输出图像上以供展示之用：
for ((top, right, bottom, left), name) in zip(boxes, names):
	# 画出检测到人脸的矩形框
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
	# 展示人名，如果边界盒位于图像顶端，则将文本移到边界盒下方，否则文本就被截掉了。
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
 

# 展示输出结果
cv2.imshow("Image", image)
cv2.waitKey(0)






