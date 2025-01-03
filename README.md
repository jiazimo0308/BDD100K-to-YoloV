# BDD100K-to-YoloV
BDD100k标签数据(JSON)转为YOLOV5格式

### 在代码里写清注释了，请看代码。

    #YOLO数据集的格式： class、x_center/img_width、y_center/img_height、w/img_width、h/img_height

    '''
    class               ：目标类别
    x_center/img_width  ：归一化中心列坐标
    y_center/img_height ：归一化中心行坐标
    w/img_width         ：归一化宽
    h/img_height        ：归一化高
    '''
    
    #引用库
    import os
    import cv2 as cv
    import shutil
    import json
    
    #全局路径
    data_root = r"/************************/bdd100k/"
    #标签路径
    label_ori = r"/***********************/bdd100k/label"
    
    def ConventDATA(which):
        '''which:可以为train，也可为val'''
        #图片位置
        img_root = data_root + "100k/"+str(which)
        #数据存放的位置
        label_root = data_root + 'ConventLabel/'+str(which)
        #数据读取的位置
        jsonpath = os.path.join(label_ori, 'bdd100k_labels_images_'+str(which)+'.json')
        #检测路径是否存在
        #如果没有就生成
        if not os.path.isdir(label_root):
            os.makedirs(label_root)
        else:
            # 如果之前已经生成过: 递归删除目录和文件, 重新生成目录
            shutil.rmtree(label_root)
            os.makedirs(label_root)
        #读取标签文件，并加载
        jsonfile = open(jsonpath, "rb")
        fileJson = json.load(jsonfile)
        #根据长度判断图片和标签的长度是否相等（从个数上进行初步判断）直接
        imgs = os.listdir(img_root)
        img_count = len(imgs)
        json_label_count = len(fileJson)
        if json_label_count!=img_count:
            print('数据个数不匹配存在问题！！！！！')
        #查找具体少哪一个图片，输出图片名和字典
        fileJson_imgs = []
        for i in range(len(fileJson)):
            imgdict = fileJson[i]
            fileJson_imgs.append(imgdict['name'])
            if 'labels' not in imgdict.keys():
                print('json {} not labels!'.format(i))
                print('imgdict: ', imgdict)
        # 判断是否再两者文件夹中都能相互找到对方
        imgs_diff_jsonfile = list(set(imgs).difference(set(fileJson_imgs)))
        jsonfile_diff_imgs = list(set(fileJson_imgs).difference(set(imgs)))
        print('in imgs but not in jsonfile: ', imgs_diff_jsonfile)
        print('in jsonfile but not in imgs: ', jsonfile_diff_imgs)
        #对不对的图片数据进行删除
        for del_img in imgs_diff_jsonfile:
            del_img_path = os.path.join(img_root, del_img)
            #如果路径存在，则删除图片
            if os.path.exists(del_img_path):
                os.remove(del_img_path)
        #关键词提取
        used_names = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider','traffic sign', 'traffic light']
        #关键词编辑
        category2id = {
            'car': 0,
            'bus': 1,
            'person': 2,
            'bike': 3,
            'truck': 4,
            'motor': 5,
            'train': 6,
            'rider': 7,
            'traffic sign': 8,
            'traffic light': 9
        }
        #fileJson为[{}]
        #count为处理总数
        count = 0
        empty_count=0
        #根据fileJson中的数据进行提取
        for imgdict in fileJson:
            #提取照片名改为.txt
            txtfile = imgdict['name'].replace('.jpg', '.txt')
            #存入到目标路径中
            #目标路径
            txtpath = os.path.join(label_root, txtfile)
            #对图像进行数据度量
            #图片路径
            imgpath = os.path.join(img_root, imgdict['name'])
            #对数据进行cv读取
            img = cv.imread(imgpath)
            #数据尺寸提取
            img_height, img_width, _ = img.shape
            #再对数据进行检测
            if 'labels' not in imgdict.keys():
                t=open(txtpath,'a')
                t.close()
                empty_count+=1
                print('创建空文本文件',txtfile)
                continue
        #对数据进行提取
            for label in imgdict['labels']:
                category = label['category']
                if category in used_names:
                    x1 = label['box2d']['x1']
                    x2 = label['box2d']['x2']
                    y1 = label['box2d']['y1']
                    y2 = label['box2d']['y2']
                    #中心行坐标
                    x_center = (x1 + x2) / 2
                    #中心列坐标
                    y_center = (y1 + y2) / 2
                    #宽
                    w = x2 - x1
                    #高
                    h = y2 - y1
                    #数据归一化的处理和存储格式
                    label_str= '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    #分别对应
                    category2id[category],
                    #归一化
                    x_center / img_width,
                    y_center / img_height,
                    w / img_width,
                    h / img_height)
                # 以追加的方式添加每一帧的label
                    with open(txtpath, 'a') as f:
                        f.write(label_str)
        count+=1
        if count%200==0:
            print('图片{}处理完'.format(count))
        #处理结果
        print('########################################')
        print('labels txt file count: ', len(os.listdir(label_root)))
        print('empty txt file count: ', empty_count)
        print('images count: ', len(os.listdir(img_root)))
        print('\nAll image dealt! Done!')
    
    ConventDATA('val')
