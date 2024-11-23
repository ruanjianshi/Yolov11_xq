from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('best.pt')

    results = model(r"data\test\images\79_jpg.rf.3276808c1eb279588f0cb05ae4f6960d.jpg")

    results[0].show()

    pass

#yolo predict model='best.pt' source="ultralytics\ultralytics-main\ultralytics\assets"