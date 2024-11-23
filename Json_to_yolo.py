import json
import os
 
# 为了提高计算效率和简化模型处理，用数字表示类别
label_translate_dict = {
    "car": "0",
    "fox": "1"
}
 
 
# 框xyxy的坐标转换为xywh，因为yolo读取的时xywh坐标
def xyxy2xywh(size, box):  # (xmin,ymin)左上角，(xmax,ymax)右下角
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return [x, y, w, h]  # 返回的都是标准化后的值
 
 
def json2yolo(path, save_path):
    txt_name = os.path.basename(path)  # 返回path最后文件名
    txt_name = txt_name.replace('.json', '.txt')
    with open(path, 'r') as f:
        data = json.load(f)  # 读取到这个文件所有内容，并返回字典dict格式
        h = data['imageHeight']
        w = data['imageWidth']
        res = []  # 创建用于写入的列表
        for item in data['shapes']:  # (打开json文件就可以明白)
            label = item['label']
            points = item['points']
            if len(points) == 2:  # json的框
                xmin = min(points[0][0], points[1][0])
                ymin = min(points[0][1], points[1][1])
                xmax = max(points[0][0], points[1][0])
                ymax = max(points[0][1], points[1][1])
                box = [float(xmin), float(ymin), float(xmax), float(ymax)]
                # 将x1, y1, x2, y2转换成yolo所需要的x, y, w, h格式
                bbox = xyxy2xywh((w, h), box)
                if label in label_translate_dict:
                    class_id = label_translate_dict[label]
                    res.append(f"{class_id} {' '.join(str(x) for x in bbox)}")
                else:
                    print(f"Warning: Label '{label}' not found in label_translate_dict")
 
        # 写入目标文件中，格式为 id x y w h
        with open(os.path.join(save_path, txt_name), 'w') as out_file:
            for line in res:
                out_file.write(line + '\n')
        out_file.close()
 
 
if __name__ == '__main__':
    # json格式数据路径
    path = r'D:\BaiduNetdiskDownload\yolov11\yolov11_project\test\label\Json\73_jpg.rf.4a2e03f9868a026dc1e1e7d166b59340.json'  # 转换成txt的json文件，替换成自己的
    save_path = r'D:\BaiduNetdiskDownload\yolov11\yolov11_project\test\label'  # txt保存根目录，替换成自己的


    if not os.path.isfile(path):
        print(f"文件不存在: {path}")
    else:
        json_names = os.listdir(path)
        for json_name in json_names:
            json2yolo(os.path.join(path, json_name), save_path)  # json转换yolo格式的坐标