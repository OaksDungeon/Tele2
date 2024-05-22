import json
from fastapi import FastAPI
from ultralytics import YOLO
import io
from fastapi import FastAPI, File, UploadFile, Path
from pydantic import BaseModel
from typing import Annotated
import tempfile
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware

tags_metadata = [
    {
        "name": "yolo_detection",
        "description": "Обработка фотографии с помощью Yolo8. В случае успешного выполнения работы кода возвращает JSON-строку в которой содержаться id обнаруженного объекта, id типа обнаружения, название обнаружения, координаты формата (x,y) углов в порядке: левый-верхний, левый-нижний, правый-нижний, правый-верхний.",
    },
]
app = FastAPI(openapi_tags=tags_metadata) #запуск FastAPI для возможности создания запросов

origins = [
    "http://localhost:3000",
    "http://192.168.56.1:3000",
    "http://192.168.0.13:58829",
    "http://192.168.0.11:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Detected_Object: #класс, объекты которого содержат информацию о каждом обнаруженном объекте
    def __init__(self, id, clas_id, name, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4):
        self.id = id
        self.clas_id = clas_id
        self.clas_name = name
        self.left_up_x = x_1
        self.left_up_y = y_1
        self.left_down_x = x_2
        self.left_down_y = y_2
        self.right_down_x = x_3
        self.right_down_y = y_3
        self.right_up_x = x_4
        self.right_up_y = y_4



@app.post("/yolo_detection", tags=["yolo_detection"]) #точка, обозначающая POST-запрос
async def yolo_detection(file: Annotated[UploadFile, File(title="Photo", description="Фотография в формате jpg")]): #основная функция
    if not file: #если файл не обнаружен, код вернет информацию об этом
        return {"message": "No upload file sent"}
    else:
        global json_string
        temp_dir = tempfile.mkdtemp() #создание локального временного хранилища для хранения полученного изображения
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        mas_det_obj = [] #массив для хранения всех обнаруженных объектов


        model = YOLO('yolo8_version/best.pt') #выбор используемой модели Yolo8


        results = model(file_path) #непосредственно обнаружение объектов, используя Yolo8
        for result in results:
            # ДЕМОНСТРАЦИОННЫЙ КОД
            # result.show()  # display to screen

            # ДЕМОНСТРАЦИОННЫЙ КОД
            i = 0
            boxes = result.boxes
            for box in boxes:

                coordinates = box.xyxy
                xyxy_boxes = coordinates.squeeze().tolist()
                x1, y1 = xyxy_boxes[0], xyxy_boxes[1]  #правый нижний угол
                x2, y2 = xyxy_boxes[2], xyxy_boxes[3]  #левый верхний угол
                x3, y3 = x2, y1  #левый нижний угол
                x4, y4 = x1, y2  #правый верхний угол

                clas_1 = box.cls
                clas = clas_1.squeeze().tolist()
                clas_name = ""
                match clas:
                    case 0.0:
                        clas_name = "cash"
                    case 1.0:
                        clas_name = "corner"
                    case 2.0:
                        clas_name = "door"
                    case 3.0:
                        clas_name = "shelving"
                    case 4.0:
                        clas_name = "showcase"
                    case 5.0:
                        clas_name = "sign"
                    case 6.0:
                        clas_name = "window"
                det_obj = Detected_Object(i, clas, clas_name, x4, y4, x1, y1, x3, y3, x2, y2) #создание объекта класса Detected_Object
                mas_det_obj.append(det_obj) #добавление объекта в массив
                i += 1

            det_obj_slov = [] #инициализация словаря для хранения информации об обнаруженных объектах
            for obj in mas_det_obj:
                det_obj_slov.append(obj.__dict__)
            json_string = json.dumps(det_obj_slov) #создание json-строки хранящей информацию обо всех обнаруженных объектах
        return {"json_string": json_string} #возвращение клиенту полученную строку