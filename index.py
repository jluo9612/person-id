import os
import config
import Algorithmia
from os.path import basename

client = Algorithmia.client(config.key)
algo = client.algo("cv/FaceRecognition/0.2.0")
files = client.dir("data://jl9612/CommunityFaceClassifiers")

# list namespace and people in model
nm = {
	"action": "list_name_spaces",
	"data_collection": "CommunityFaceClassifiers"
}

print (algo.pipe(nm).result)

ppl = {
	"action": "list_people",
	"data_collection": "CommunityFaceClassifiers",
 	"name_space": "cast"
}

print (algo.pipe(ppl).result)

#step 1 add images with labels (names)

# train = {
# 	"action": "add_images",
# 	"data_collection": "CommunityFaceClassifiers",
# 	"name_space": "cast",
# 	"images": []
# }

# for file in files.files():
# 	if "test" not in file.path:
# 		data = {}
# 		data["url"] = "data://" + file.path
# 		name = os.path.splitext(basename(file.path))[0]
# 		data["person"] = ''.join([i for i in name if not i.isdigit()])  # extracts name from path
# 		train["images"].append(data)

# print (train["images"])
# print (algo.pipe(train).result)

#step 2 predict with unlabeled image visualized

test = {
	"name_space": "cast",
	"action": "predict",
	"data_collection": "CommunityFaceClassifiers",
	"images": []
}

for file in files.files():
	if "test" in file.path:
		data = {}
		data["url"] = "data://" + file.path
		test["images"].append(data)

print (algo.pipe(test).result) #kept getting array error