import os
import config
import Algorithmia
from os.path import basename

client = Algorithmia.client(config.key)

files = client.dir("data://jl9612/CommunityFaceClassifiers")

#step 1 add images with labels (names)

input1 = {
	"action": "add_images",
	"data_collection": "CommunityFaceClassifiers",
	"name_space": "cast",
	"images": []
}

for file in files.files():
	if "test" not in file.path:
		data = {}
		data["url"] = file.url 
		name = os.path.splitext(basename(file.path))[0]
		data["person"] = ''.join([i for i in name if not i.isdigit()])  #extracted name
		input1["images"].append(data)

print (input1["images"])
print (client.algo("cv/FaceRecognition/0.2.0").pipe(input1).result)