import os
import config
import Algorithmia
from os.path import basename

client = Algorithmia.client(config.key)
algo = client.algo("cv/FaceRecognition/0.2.0")

files = client.dir("data://jl9612/CommunityFaceClassifiers")


# input = {
# 	"action": "remove_name_space",
# 	"data_collection": "CommunityFaceClassifiers",
#  	"name_space": "cast"
# }

# print (algo.pipe(input).result)

#step 1 add images with labels (names)

train = {
	"action": "add_images",
	"data_collection": "CommunityFaceClassifiers",
	"name_space": "cast",
	"images": []
}

for file in files.files():
	if "test" not in file.path:
		data = {}
		data["url"] = "data://" + file.path
		name = os.path.splitext(basename(file.path))[0]
		data["person"] = ''.join([i for i in name if not i.isdigit()])  # extracts name from path
		train["images"].append(data)

print (algo.pipe(train).result)

#step 2 predict with unlabeled image visualized

test = {
	"name_space": "cast",
	"data_collection": "CommunityFaceClassifiers",
	"action": "predict",
	"images": [
		{
			"url": "data://jl9612/CommunityFaceClassifiers/community_wall1.jpg",
			"output": "data://jl9612/CommunityFaceClassifiers/temp/community_visualized.png"
		}
	]
}

print (algo.pipe(test).result)