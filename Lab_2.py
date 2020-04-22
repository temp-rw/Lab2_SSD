# -*- coding: utf-8 -*-


from google.colab import drive
import os
import torch
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random as rnd

drive.mount('/content/gdrive', force_remount = True)

precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
ssd_model.to('cuda')
ssd_model.eval()

classes_to_labels = utils.get_coco_object_dictionary()
trueClasses = ['ignored regions', 'person', 'person', 'bicycle', 'car', 'car', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motorcycle', 'others']  
allClasses = list(dict.fromkeys(classes_to_labels+trueClasses))
dictClasses = {key: [[0,0,0],[0,0,0],[0,0,0]] for key in allClasses}

imglist = os.listdir('VisDrone2019-DET-val/images')
for i, image in enumerate(imglist):
    imglist[i] = 'VisDrone2019-DET-val/images/' + image

annolist = imglist[::]
annolist = [line.replace('.jpg','.txt') for line in annolist]
annolist = [line.replace('images','annotations') for line in annolist]



"""Парсинг аннотаций"""

def Parsing(anno,images):
    annotations = []
    for i in range(len(anno)):
        f = open(anno[i], 'r+')
        lines = [line.split(',') for line in f.readlines()]
        f.close()
        
        boxes = []
        for line in lines:
            box = [int(num) for num in line]
            boxes.append(box)
        im = Image.open(images[i])
        width, height = im.size
        im.close()
        left = (width-height)/2
        right = left + height
        
        j = 0;
        while j < len(boxes):
            if boxes[j][0]+boxes[j][2] <= left or boxes[j][0]>=right:
                del boxes[j]
                j = j-1
            elif boxes[j][0]<left:
                boxes[j][2] = boxes[j][0] + boxes[j][2] - left
                boxes[j][0] = 0
            elif boxes[j][0]+boxes[j][2]>right:
                boxes[j][2] = right-boxes[j][0]
                boxes[j][0] = boxes[j][0] - left
            else:
                boxes[j][0] = boxes[j][0] - left
            j = j+1    

        for box in boxes:
            del box[4]
            del box[5]
            del box[5]
            box[0] = box[0]/height
            box[2] = box[2]/height
            box[1] = box[1]/height
            box[3] = box[3]/height
        annotations.append(boxes)
    return annotations

"""Метрика IoU"""

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA )
    A_boxArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    B_boxArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    IoU = interArea / float(A_boxArea + B_boxArea - interArea)
    return IoU

def EqualClasses(trueClass,predictedClass):
    if trueClass == predictedClass:
        return True
    elif trueClass == 'others':
        return True
    else:
        return False

"""Сбор статискики для подсчета средней точности детектирования объектов
порог IoU | точность | пропущенно | ложные данные |
"""

del imglist[515]
del annolist[515]

batch = 40

thresholds = [0.5,0.75,0.9]

for i in range(3):
    for j in range(14):
        images = [str(image) for image in imglist[j*batch:(j+1)*batch]]
        annotations = [str(annotation) for annotation in annolist[j*batch:(j+1)*batch]]
        trueBoxes = Parsing(annotations,images)

        inputs = [utils.prepare_input(image) for image in images]
        tensor = utils.prepare_tensor(inputs, precision == 'fp16')

        with torch.no_grad():
            detections_batch = ssd_model(tensor)

        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

        for image_idx in range(len(best_results_per_input)):
            predictedBoxes, predictedClasses, c = best_results_per_input[image_idx]
            for idx in range(len(predictedBoxes)):
                flag = False 
                for box in trueBoxes[image_idx]:
                    trueBox = [box[0],box[1],box[0]+box[2],box[1]+box[3]]
                    trueClass = trueClasses[box[4]]
                    if IoU(predictedBoxes[idx],trueBox) > thresholds[i] and EqualClasses(trueClass,classes_to_labels[predictedClasses[idx] - 1]):
                        flag = True
                        break
                if flag:
                    dictClasses[classes_to_labels[predictedClasses[idx] - 1]][i][0] += 1
                else:
                    dictClasses[classes_to_labels[predictedClasses[idx] - 1]][i][1] += 1
            for box in trueBoxes[image_idx]:
              trueClass = trueClasses[box[4]]
              dictClasses[trueClass][i][2] += 1

allTrueBoxes = [0,0,0]
allFalseAlarms = [0,0,0]
allMatches = [0,0,0]

for i in range(3):
    for Class in allClasses:
      allMatches[i] += dictClasses[Class][i][0]
      allFalseAlarms[i] += dictClasses[Class][i][1]
      allTrueBoxes[i] += dictClasses[Class][i][2]

Accuracies = [0,0,0]
Missed = [0,0,0]

for i in range(3):
  Accuracies[i] = 100*allMatches[i]/allTrueBoxes[i]
  Missed[i] = allTrueBoxes[i]-allMatches[i]

summary = pd.DataFrame([Accuracies, allFalseAlarms, Missed], columns=['IoU 0.5','IoU 0.75','IoU 0.9'], index=['Accuracy, %','False Alarms','Missed'])

pd.options.display.float_format = '{:,.2f}'.format
print(summary)

"""1) Вывод данных для качественного анализа (4 снимка)"""

import random as rnd

rnd.seed(10)
start = rnd.randint(0, 20)

for image_idx in range(start, start + 4):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()

for Class in allClasses:
  for i in range(3):
    matches = dictClasses[Class][i][0]
    if matches>0:
      dictClasses[Class][i][0] = matches/dictClasses[Class][i][2]
      dictClasses[Class][i][2] = dictClasses[Class][i][2] - matches

for Class in allClasses:
  for i in range(3):
    matches = dictClasses[Class][i][0]
    if matches>0:
      dictClasses[Class][i][0] = matches*100

allClasses = sorted(allClasses)

list50 = [dictClasses[Class][0] for Class in allClasses]
list75 = [dictClasses[Class][1] for Class in allClasses]
list90 = [dictClasses[Class][2] for Class in allClasses]

pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.max_rows', 50)
ClasswiseStat50 = pd.DataFrame(list50, columns=['Точность','Ложных тревог','Пропущено'], index=allClasses)
ClasswiseStat75 = pd.DataFrame(list75, columns=['Точность','Ложных тревог','Пропущено'], index=allClasses)
ClasswiseStat90 = pd.DataFrame(list90, columns=['Точность','Ложных тревог','Пропущено'], index=allClasses)

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)  
table(ax, ClasswiseStat90)  
plt.savefig('mytable.png')


