import logging
import argparse
import mxnet as mx
import os
import sys
import cv2
import numpy as np
from symbol import mobilenet
DEBUG = True
def parse_args():
	parser = argparse.ArgumentParser(description='train a highway recognition model using mobilenet')
	parser.add_argument('--train-list',dest='train_list',help='train.lst to use',default='rec_file/data_train.lst',type=str)
	parser.add_argument('--test-list',dest='test_list',help='test.lst to use',default='rec_file/data_test.lst',type=str)
	parser.add_argument('--train-rec',dest='train_rec',help='train.rec to use',default='rec_file/data_train.rec',type=str)
	parser.add_argument('--test-rec',dest='test_rec',help='test.rec to use',default='rec_file/data_test.rec',type=str)
	parser.add_argument('--root',dest='root',help='root path of dataset',default='/mnt/data-1/data/jiajie.tang/highway/dataset/',type=str)
	parser.add_argument('--batch-size',dest='batch_size',help='batch size',default=256,type=int)
	parser.add_argument('--resume',dest='resume',help='resume training from epoch n',default=-1,type=int)
	parser.add_argument('--prefix',dest='prefix',help='new model prefix',default=os.path.join(os.getcwd(),'output','exp3','mobilenet'),type=str)
	parser.add_argument('--gpus',dest='gpus',help='gpu device to train with',default='0,1,2,3',type=str)
	parser.add_argument('--begin-epoch',dest='begin_epoch',help='begin epoch of training',default=0,type=int)
	parser.add_argument('--end-epoch',dest='end_epoch',help='end epoch of training',default=240,type=int)
	parser.add_argument('--frequent',dest='frequent',help='frequency of logging',default=20,type=int)
	parser.add_argument('--data-shape',dest='data_shape',help='set image shape',default=360,type=int)
	parser.add_argument('--lr',dest='learning_rate',help='learning rate',default=0.001,type=float)
	parser.add_argument('--log-file',dest='log_file',help='log file name',default='train.log',type=str)
	parser.add_argument('--alpha',dest='alpha',help='mobilenet factor',default=0.5,type=float)
	args = parser.parse_args()
	return args

		
def main():
	args = parse_args()
	ctx =  [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]

	logging.basicConfig()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	log_file_path = os.path.join(os.path.dirname(args.prefix),args.log_file)
	fh = logging.FileHandler(log_file_path)
	logger.addHandler(fh)
	

	#train_data = mx.image.ImageIter(batch_size = args.batch_size,
	#			data_shape=(3,224,224),
        #                       label_width = 1,
        #                        path_imglist = args.train_list,
        #                        path_root=args.root,
        #                        aug_list =[mx.image.ForceResizeAug((224,224))])
	#test_data = mx.image.ImageIter(batch_size = args.batch_size,
#				data_shape=(3,224,224),
#                                label_width = 1,
#                                path_imglist = args.test_list,
#                                path_root=args.root,
#                                aug_list =[mx.image.ForceResizeAug((224,224))])

	sym,arg_params, aux_params = mx.model.load_checkpoint('output/exp2/mobilenet',30)
	texec = sym.simple_bind(ctx = mx.gpu(),data=(1,3,224,224),label = (1,))
	texec.copy_params_from(arg_params,aux_params)
	img = cv2.imread("/mnt/data-1/data/jiajie.tang/highway/dataset/1/7.11-hiv00028-0136.jpg")
	if img.size == 0:
		print("open image failed")
	
	img = img.astype(np.float32)
	img[:] = img[:,:,::-1]
	img = cv2.resize(img,(224,224))
	img = mx.nd.array(img)
	img = img.transpose(axes = (2,0,1))
	img -= 123
	img = mx.nd.expand_dims(img,0)
	print(texec.forward(is_train = False,data = img,softmax_label = mx.nd.array([1])))
			
if __name__ == "__main__":
	main()
