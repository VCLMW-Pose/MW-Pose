'''
    Created on Sun Dec 1 00:16 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   :

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import json
import os

joint_name = [ 'nose', 'neck', 'rShoulder',
          'rElbow', 'rWrist', 'lShoulder',
          'lElbow', 'lWrist', 'rHip', 'rKnee',
          'rAnkle', 'lHip', 'lKnee', 'lAnkle',
          'rEye', 'lEye', 'rEar', 'lEar']

def json_reader_alphapose(filename):
    '''
    This function is used for reading json file outputted by AlphaPose.
    And this function will only read annotation for one image
    :return: list of people[
                Every people has a architecture below:
                    dictionary of joints{
                                        key: joint name (e.g. 'head')
                                        value: coordinate[x, y] (e.g. [125, 320])
                                        }
                                ]
    '''
    jsonfile = open(filename, 'r')
    data = json.load(jsonfile, encoding='UTF-8')
    jsonfile.close()
    bodies = data['bodies']
    peoples = []
    for body in bodies:
        people = {}
        for i, joint in enumerate(joint_name):
            people[joint] = body['joints'][i*3: (i+1)*3]
            people[joint][0] = int(people[joint][0])
            people[joint][1] = int(people[joint][1])
        peoples.append(people)
    return peoples

def load_json_anno(path='../../data/annotations'):
    annos = {}
    for root, _, files in os.walk(path):
        for file in files:
            peoples = json_reader_alphapose(os.path.join(root, file))
            annos[file.split('.')[0]+'.jpg'] = peoples
    return annos

if __name__ == "__main__":
    annos = load_json_anno()
    exit()


