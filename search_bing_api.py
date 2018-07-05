# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os
 

# 设置查询参数和图片保存地址
# 运行的时候必须输入参数-q -o
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())


# 设置设置查询的地址， 秘钥，查询最大数量，每一页的数量，执行的时候可能会抛出的异常
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
API_KEY = "1c4cd6595ac64b339a46a81150efad5c" 
MAX_RESULTS = 50
GROUP_SIZE = 5
EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])


# 设置查询的参数
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}


# 执行查询
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()


# 总数量，totalEstimatedMatches：总共查询出的数量，在总数量和设置的最大返回结果中选最小的
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults, term))


# 总共下载的数量，因为有一些会下载失败
total = 0


# 按分组个数，遍历下载图片
for offset in range(0, estNumResults, GROUP_SIZE):
	print("[INFO] making request for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
	params["offset"] = offset
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
	for v in results["value"]:
		# 下载图像且存入指定文件夹，抛异常跳过该图像
		try:
			print("[INFO] fetching: {}".format(v["contentUrl"]))
			r = requests.get(v["contentUrl"], timeout=30)
			ext = v["contentUrl"][v["contentUrl"].rfind("."):]
			p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])
			f = open(p, "wb")
			f.write(r.content)
			f.close()
		except Exception as e:
			if type(e) in EXCEPTIONS:
				print("[INFO] skipping: {}".format(v["contentUrl"]))
				continue

		# 使用cv2验证一下图像，看是否已经下载成功，成功总数量加1
		image = cv2.imread(p)
		if image is None:
			print("[INFO] deleting: {}".format(p))
			os.remove(p)
			continue
		else:
			total += 1

