# encoding:utf-8

import face_recognition 
import cv2
import os

#存储知道人名列表
known_names=[] 
#存储知道的特征值
known_encodings=[]

def readPersons(path):
    for image_name in os.listdir(path):
        load_image = face_recognition.load_image_file(path+image_name) #加载图片
        image_face_encoding = face_recognition.face_encodings(load_image)[0] #获得128维特征值
        known_names.append(image_name.split(".")[0].split("_")[0])
        known_encodings.append(image_face_encoding)
    #print(known_names)

def face(namePath):
    rgb_frame = face_recognition.load_image_file(namePath)
    face_locations = face_recognition.face_locations(rgb_frame)#获得所有人脸位置
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) #获得人脸特征值
    face_names = [] #存储出现在画面中人脸的名字
    for face_encoding in face_encodings:         
        matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.5)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            name="unknown"
        face_names.append(name)

    img = cv2.imread(namePath)
    print(face_names)
    # 将捕捉到的人脸显示出来
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 1) # 画人脸矩形框
        # 加上人名标签
        #cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
        font = cv2.FONT_HERSHEY_DUPLEX 
        print(name)
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('frame', img)
    cv2.waitKey(0)

if __name__=='__main__':
    readPersons("./persons/") #存放已知图像路径
    face("./find/ping3.jpg")