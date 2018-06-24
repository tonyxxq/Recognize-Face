人脸识别，包括图像，摄像头拍到的视频，和视频文件，其中摄像头和视频文件除了输入的视频的方式不一样之外，其他的都都一样。

需要安装依赖库

- dlib（http://dlib.net/）；用户获取图像的编码
- face_recognition（https://github.com/ageitgey/face_recognition）。
- imutils 提供一些遍历的函数

> 由Davis King维护的dlib库包含了“深度度量学习”的实现，用来在实际的识别过程中构建面部嵌入。
>
> Adam Geitgey创建的face_recognition库则封装了dlib的面部识别功能，使之更易用。
>
> 以上的安装都可以使用pip install





原理：通过dlib根据原始图片提取特征编码，存放起来；然后在获取图片使用dlib提取特征编码，和之前的图片去比对，判断到底是属于哪一个人的人脸。



参考文章：https://mp.weixin.qq.com/s/zlfSGIowIJYURRCFkv4qng