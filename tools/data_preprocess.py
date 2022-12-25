import json
import os
import cv2

classname_id = {'embedded': 1, 'isolated': 2}

class Math2Txt:
    def __init__(self, img_path, txt_path, dst_path):
        self.img_path = img_path
        self.txt_path = txt_path
        self.dst_path = dst_path
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)

    def convert2Txt(self):
        for file in os.listdir(self.txt_path):
            img_path = os.path.join(self.img_path, file.replace('color_', '').replace('txt', 'jpg'))
            txt_path = os.path.join(self.txt_path, file)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            g = open(os.path.join(self.dst_path, file.replace('color_', '')), 'w')
            with open(txt_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i < 4:
                        continue
                    line = line.replace(' ', '')
                    line = line.strip('\n').split('\t')
                    label = line[-1]
                    x1 = int(float(line[0]) * w / 100)
                    y1 = int(float(line[1]) * h / 100)
                    x2 = int(x1 + float(line[2]) * w / 100) - 1
                    y2 = y1
                    x3 = x2
                    y3 = int(y1 + float(line[3]) * h / 100) - 1
                    x4 = x1
                    y4 = y3
                    data = str(x1) + '\t' + str(y1) + '\t' + str(x2) + '\t' + str(y2) + '\t' \
                           + str(x3) + '\t' + str(y3) + '\t' + str(x4) + '\t' + str(y4) + '\t' + label + '\n'
                    # print(data)
                    g.write(data)
                g.close()


class Txt2CoCo:
    def __init__(self, txt_path, img_path):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.txt_path = txt_path
        self.img_path = img_path

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)

    def convert2Coco(self):
        self._init_categories()
        print(self.txt_path)
        for im in os.listdir(self.img_path):
            im_name = im.split('.')[0]
            self.images.append(self._image(im))
            txt = im_name + '.txt'
            with open(os.path.join(self.txt_path, txt), 'r', encoding='utf-8') as f:
                for i, ann in enumerate(f.readlines()):
                    annotation, flag = self._annotation(ann)
                    if not flag:
                        continue
                    else:
                        self.annotations.append(annotation)
                        self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'COCO form created'
        instance['license'] = 'MIT'
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self, im_name):
        img = cv2.imread(os.path.join(img_path, im_name))
        try:
            H, W = img.shape[:-1]
        except:
            raise ValueError(im_name)
        image = {}
        image['height'] = H
        image['width'] = W
        image['id'] = self.img_id
        image['file_name'] = im_name
        return image

    def _annotation(self, ann):
        flag = False
        annotation = {}
        ann = ann.strip('\n').split('\t')
        label = int(ann[-1]) + 1
        annotation['category_id'] = label

        flag = True
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['segmentation'] = [self.str2int(ann[:-1])]
        annotation['bbox'] = self.segm2Bbox(annotation['segmentation'])
        annotation['iscrowd'] = 0
        annotation['area'] = self.getArea(annotation['bbox'])
        return annotation, flag

    def str2int(self, msg):
        coord = []
        for num in msg:
            coord.append(int(num))
        return coord

    def segm2Bbox(self, segm):
        x_min = min(segm[0][0::2])
        x_max = max(segm[0][0::2])
        y_min = min(segm[0][1::2])
        y_max = max(segm[0][1::2])
        bbox = [x_min, y_min, x_max-x_min+1, y_max-y_min+1]
        return bbox

    def getArea(self, bbox):
        return float(bbox[2] * bbox[3])


if __name__ == '__main__':
    img_path = '/platform_tech/zhongyuxiang/dataset/IBEM_dataset/Va01/img'  # Your split image path
    txt_path = '/platform_tech/zhongyuxiang/dataset/IBEM_dataset/Va01/gt'   # Your split txt label path
    dst_path = '/platform_tech/zhongyuxiang/dataset/IBEM_dataset/Va01/gt_icdar' # icdar txt label path
    train_path = '/platform_tech/zhongyuxiang/dataset/IBEM_dataset/Va01'    # coco format json destination folder

    toIcdar = Math2Txt(img_path, txt_path, dst_path)
    toIcdar.convert2Txt()

    toCoco = Txt2CoCo(dst_path, img_path)
    instance = toCoco.convert2Coco()
    toCoco.save_coco_json(instance, os.path.join(train_path, 'train_coco.json'))