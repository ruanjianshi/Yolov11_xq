from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO('yolo11n.pt')

    model.train(data = 'data/data.yaml',epochs=50,imgsz=640,batch=32,workers=4)

#yolo train model='yolo11n.pt' data='test/data.yaml' epochs=50 batch=32