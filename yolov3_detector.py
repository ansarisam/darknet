import os
import subprocess

image_path="test_images/dog.jpg"
yolov3_weights_path="backup/yolov3.weights"
cfg_path="cfg/yolov3.cfg"
output_path="output_path"

process = subprocess.Popen(['./darknet', 'detect', cfg_path, yolov3_weights_path, image_path],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

std_string = stdout.decode("utf-8")
print(std_string)
std_string = std_string.split(image_path)[1]

count = 0
outputList = []
rowDict = {}
for line in std_string.splitlines():

    if count > 0:
        if count%2 > 0:
            obj_score = line.split(":")
            obj = obj_score[0]
            score = obj_score[1]
            rowDict["object"] = obj
            rowDict["score"] = score
        else:
            bbox = line.split(",")
            rowDict["bbox"] = bbox
            outputList.append(rowDict)
            rowDict = {}
    count = count +1
rowDict["image"] = image_path
rowDict["predictions"] = outputList

print(rowDict)