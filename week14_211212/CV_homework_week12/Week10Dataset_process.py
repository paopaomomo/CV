# coding:utf-8
import os
import glob
import numpy as np
import torch
import cv2
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET



def parse_rec(xml_file, pic_path):
	tree = ET.parse(xml_file) # 解析读取xml函数
	# root = tree.getroot()
	# size = root.find('size')
	# print(size.text)
	#objects = [obj_struc1,obj_struc2,...]
    objects = []
    # img_dir= [xxx/xxx/xxx/img1.jpg,xxx/xxx/img2.jpg,xxx/xxx/img3.jpg]
	img_dir = []
	for xml_name in tree.findall('filename'):
		img_path = os.path.join(pic_path, xml_name.text)
		img_dir.append(img_path)
	for obj in tree.findall('object'):
		obj_struct = {}
        # 类别： 百岁山
		obj_struct['name'] = obj.find('name').text
		# obj_struct['pose'] = obj.find('pose').text
		# obj_struct['truncated'] = int(obj.find('truncated').text)
		# obj_struct['difficult'] = int(obj.find('difficult').text)
		bbox = obj.find('bndbox')
        # 框
		obj_struct['bbox'] = [int(bbox.find('xmin').text),
							  int(bbox.find('ymin').text),
							  int(bbox.find('xmax').text),
							  int(bbox.find('ymax').text)]
		objects.append(obj_struct)
	return objects, img_dir


def get_image_bbox(annotations_path, img_path):
	folders = os.listdir(annotations_path)
	objects = []
	img_dirs = []
	for folder_name in folders:
		# print('folder_name: ', folder_name)

		if folder_name == '.DS_Store':
			pass
		else:
			folder_path = os.path.join(annotations_path, folder_name)
			xml_pathes = glob.glob(folder_path + '/*.xml')
			# print('xml_pathes: ', xml_pathes)
			for xml_path in xml_pathes:
				pic_path = os.path.join(img_path, folder_name)

				object, img_dir = parse_rec(xml_path, pic_path)
				img_dirs.append(img_dir)
				objects.append(object)
	return img_dirs, objects


def show_img(idx, img_dirs, objects):
	obj_struct = objects[idx][0]
	img_path = img_dirs[idx][0]

	(xmin, ymin, xmax, ymax) = obj_struct['bbox']
	w = xmax - xmin
	h = ymax - ymin
	label = obj_struct['name']
	image = cv2.imread(img_path)
	cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 2)
	cv2.namedWindow(label, cv2.WINDOW_NORMAL)
	cv2.imshow(label, image)
	cv2.waitKey(1000)  # 持续时长


def get_train_val_txt(val_percent, img_dirs, objects, txt_path, classes_list):
	'''
	txt描述文件 image_name.jpg xmin ymin xmax ymax class
	'''
	# train_percent = 1 - val_percent
	random.seed(0)
	number = len(img_dirs)
	print('number:', number)
	all_idx = range(number)
	print('all_idx: ', all_idx)
	val_number = int(number * val_percent)
	print('val_number: ', val_number)
	# train_number = int(val_number*train_percent)
	# print('train_number: ', train_number)
	val_idx = random.sample(all_idx, val_number)
	print('val_idx: ', val_idx)
	# train = random.sample(trainval, train_number)
	# print('train: ', train)
	train_txt = 'train.txt'
	train_file = open(os.path.join(txt_path, train_txt), 'w')
	val_txt = 'val.txt'
	val_file = open(os.path.join(txt_path, val_txt), 'w')

	for idx in all_idx:
		obj_struct_bbox = objects[idx][0]['bbox']
		obj_struct_name = objects[idx][0]['name']
		img_path = img_dirs[idx][0]
		label_idx = classes_list.index(obj_struct_name)
		if idx in val_idx:
			val_file.write(img_path + ' ' + str(obj_struct_bbox[0])
						   + ' ' + str(obj_struct_bbox[1])
						   + ' ' + str(obj_struct_bbox[2])
						   + ' ' + str(obj_struct_bbox[3])
						   + ' ' + str(label_idx))
			val_file.write('\n')
		# print(obj_struct_bbox)
		# print(img_path)
		else:
			train_file.write(img_path + ' ' + str(obj_struct_bbox[0])
							 + ' ' + str(obj_struct_bbox[1])
							 + ' ' + str(obj_struct_bbox[2])
							 + ' ' + str(obj_struct_bbox[3])
							 + ' ' + str(label_idx))
			train_file.write('\n')

	val_file.close()
	train_file.close()


def draw_img_rects(img_name, xmin, ymin, xmax, ymax):
	w = xmax - xmin
	h = ymax - ymin
	image = cv2.imread(img_name)
	cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmin) + int(w), int(ymin) + int(h)), (0, 255, 0), 2)
	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	cv2.imshow('img', image)
	cv2.waitKey(1000)  # 持续时长


def check_file_rects(file_name, img_path):
	'''
	file_name: train,txt
	img_path: 'week10_dataset/image/'
	'''
	file_path = os.path.join(img_path, file_name)
	with open(file_path) as f:
		lines = f.readlines()  # xxx.jpg xx xx xx xx class

	for line in lines:
		splited = line.strip().split()
		img_name = splited[0]  # 存储图片的地址+图片名称
		num_boxes = (len(splited) - 1) // 5  # 每一幅图片里面有多少个bbox
		# box = []
		# label = []
		for i in range(num_boxes):
			xmin = float(splited[1 + 5 * i])
			ymin = float(splited[2 + 5 * i])
			xmax = float(splited[3 + 5 * i])
			ymax = float(splited[4 + 5 * i])
			label_idx = int(splited[5 + 5 * i])
			# box.append([xmin, ymin, xmax, ymax])
			# label.append(label_idx)
			draw_img_rects(img_name, xmin, ymin, xmax, ymax)


def get_mean_std(file_name, img_path):
	'''
	file_name: train,txt
	img_path: 'week10_dataset/image/'
	146
	normMean = [140.7888129  131.87180331 126.43424442]
	normStd = [53.84969082 54.91440049 56.4051085 ]
	'''
	means = [0, 0, 0]
	stdevs = [0, 0, 0]
	num_imgs = 0

	file_path = os.path.join(img_path, file_name)
	with open(file_path) as f:
		lines = f.readlines()  # xxx.jpg xx xx xx xx class

	for line in lines:
		splited = line.strip().split()
		img_name = splited[0]  # 存储图片的地址+图片名称

		num_imgs += 1
		img = cv2.imread(img_name)  # cv2默认为bgr顺序
		img = np.asarray(img)
		# print(img.shape)
		img = img.astype(np.float32)
		for i in range(3):
			means[i] += img[:, :, i].mean()
			stdevs[i] += img[:, :, i].std()
	print(num_imgs)
	means.reverse()
	stdevs.reverse()

	means = np.asarray(means) / num_imgs
	stdevs = np.asarray(stdevs) / num_imgs

	print("normMean = {}".format(means))
	print("normStd = {}".format(stdevs))
	return means, stdevs


class myDataset(DataLoader):
	image_size = 448

	def __init__(self, img_path, file_name, train, transform):
		self.img_path = img_path
		self.file_name = file_name
		self.file_path = os.path.join(self.img_path, self.file_name)
		self.train = train
		self.transform = transform
		self.all_img_name = []
		self.boxes = []
		self.labels = []
		self.S = 14  # grid number 14*14 normally
		self.B = 2  # bounding box number in each grid
		self.classes_num = 14  # how many classes
		self.mean = (141, 132, 126)  # RGB

		with open(self.file_path) as f:
			lines = f.readlines()  # xxx.jpg xx xx xx xx class

		for line in lines:
			splited = line.strip().split()
			img_name = splited[0]  # 存储图片的地址+图片名称
			num_boxes = (len(splited) - 1) // 5  # 每一幅图片里面有多少个bbox
			box = []
			label = []
			for i in range(num_boxes):
				xmin = float(splited[1 + 5 * i])
				ymin = float(splited[2 + 5 * i])
				xmax = float(splited[3 + 5 * i])
				ymax = float(splited[4 + 5 * i])
				label_idx = int(splited[5 + 5 * i])
				box.append([xmin, ymin, xmax, ymax])
				label.append(label_idx)
			self.all_img_name.append(img_name)
			self.boxes.append(torch.Tensor(box))
			self.labels.append(torch.LongTensor(label))
		self.num_samples = len(self.boxes)

	def __getitem__(self, idx):
		img_name = self.all_img_name[idx]
		img = cv2.imread(img_name)
		# print(type(img))
		boxes = self.boxes[idx].clone()
		labels = self.labels[idx].clone()
		if self.train:
			# torch自带的transform会造成bbox的坐标,需要自己来定义数据增强
			pass
			# img = self.RandomBrightness(img)
			# img, boxes = self.random_flip(img, boxes)
		h, w, _ = img.shape
		boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 坐标归一化处理，为了方便训练
		img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
		img = self.subMean(img, self.mean)  # 减去均值
		img = cv2.resize(img, (self.image_size, self.image_size))  # 将所有图片都resize到指定大小\
		target = self.encoder(boxes, labels)  # 将图片标签编码到7x7*14的向量

		for t in self.transform:
			img = t(img)

		return img, target

	def __len__(self):
		return self.num_samples


	def subMean(self, bgr, mean):
		mean = np.array(mean, dtype=np.float32)
		bgr = bgr - mean
		return bgr


	def BGR2RGB(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


	def encoder(self, boxes, labels):
		'''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 14x14x24
        '''
		grid_num = 14
		yolo_output = 24  # 5*2+14
		target = torch.zeros((grid_num, grid_num, yolo_output))
		cell_size = 1. / grid_num  # 每个格子的大小
		# 右下坐标        左上坐标
		# x2,y2           x1,y1
		x2y2 = boxes[:, 2:]
		x1y1 = boxes[:, :2]

		wh = x2y2 - x1y1
		cxcy = (x2y2 + x1y1) / 2

		for i in range(cxcy.size()[0]):
			# 物体中心坐标
			cxcy_sample = cxcy[i]
			# 指示落在那网格，如[0,0]
			ij = (cxcy_sample / cell_size).ceil() - 1  # # 中心点对应格子的坐标
			#    0 1    2 3   4      5 6   7 8   9
			# [中心坐标,长宽,置信度,中心坐标,长宽,置信度, 14个类别] x 7x7   因为一个框预测两个物体
			
			# 第一个框的置信度
			target[int(ij[1]), int(ij[0]), 4] = 1
			# 第二个框的置信度
			target[int(ij[1]), int(ij[0]), 9] = 1
			# 类别
			target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
			# xy为归一化后网格的左上坐标---->相对整张图
			xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
			# 物体中心相对左上的坐标 ---> 坐标x,y代表了预测的bounding
			# box的中心与栅格边界的相对值
			delta_xy = (cxcy_sample - xy) / cell_size  # 其实就是offset

			# (1) 每个小格会对应B(2)个边界框，边界框的宽高范围为全图，表示以该小格为中心寻找物体的边界框位置。
			# (2) 每个边界框对应一个分值，代表该处是否有物体及定位准确度
			# (3) 每个小格会对应C个概率值，找出最大概率对应的类别P(Class|object)，并认为小格中包含该物体或者该物体的一部分。

			# 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例

			target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
			target[int(ij[1]), int(ij[0]), :2] = delta_xy
			# 每一个网格有两个边框
			target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # 长宽
			# 中心坐标偏移
			# 由此可得其实返回的中心坐标其实是相对左上角顶点的偏移，因此在进行预测的时候还需要进行解码
			target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
		return target



if __name__ == '__main__':
	annotations_path = 'week10_dataset/annotations/'
	img_path = 'week10_dataset/image/'
	classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
					"可口可乐", "农夫山泉", "恒大冰泉", "其他"]
	classes_num = len(classes_list)
	# class_to_idx
	# class_idx = classes_list.index('其他')
	# print('class_idx: ', class_idx) # 13

	# img_dirs, objects = get_image_bbox(annotations_path, img_path)

	''' step 01 检查标注'''
	# for i in range(len(img_dirs)):
	# 	show_img(i, img_dirs, objects)


	'''step 02 划分数据集，准备 train.txt、val.txt'''
	# val_percent = 0.2
	# get_train_val_txt(val_percent, img_dirs, objects, txt_path=img_path, classes_list=classes_list)

	'''step 03 检查划分数据集之后的标注'''
	# file_name = 'val.txt'
	# check_file_rects(file_name, img_path)

	'''step 04 获得训练集的均值和方差'''
	# file_name = 'train.txt'
	# means, stdevs = get_mean_std(file_name, img_path)
	# normMean = [141, 132, 126]

	'''step 05 编写 class myDataset(DataLoader)'''
	train_dataset = myDataset(img_path=img_path, file_name='train.txt', train=True, transform=[transforms.ToTensor()])
	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

	for i, sample in enumerate(train_loader):
		img, target = sample[0], sample[1]
		print(" ********* img_%d *********" % (i))
		print(img.shape)
		print(img)

		print("******** target_%d ********" % (i))
		print(target.shape)
		print(target)
		if i == 1:
			break




