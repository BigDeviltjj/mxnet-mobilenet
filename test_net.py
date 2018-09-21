import logging
import argparse
import mxnet as mx
import numpy as np
import os
import sys
from symbol import mobilenet
DEBUG = True
def parse_args():
	parser = argparse.ArgumentParser(description='train a highway recognition model using mobilenet')
	parser.add_argument('--test-list',dest='test_list',help='test.lst to use',default='rec_file/data_test.lst',type=str)
	parser.add_argument('--test-rec',dest='test_rec',help='test.rec to use',default='rec_file/data_test.rec',type=str)
	parser.add_argument('--root',dest='root',help='root path of dataset',default='/mnt/data-1/data/jiajie.tang/highway/dataset/',type=str)
	parser.add_argument('--prefix',dest='prefix',help='model prefix',default=os.path.join(os.getcwd(),'output','exp1','mobilenet'),type=str)
	parser.add_argument('--batch-size',dest='batch_size',help='batch size',default=256,type=int)
	parser.add_argument('--gpus',dest='gpus',help='gpu device to test with',default='0,1,2,3',type=str)
	parser.add_argument('--frequent',dest='frequent',help='frequency of logging',default=20,type=int)
	parser.add_argument('--log-file',dest='log_file',help='log file name',default='test.log',type=str)
	parser.add_argument('--epoch',dest='epoch',help='resume epoch',default=185,type=int)
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
	
	test_data = mx.io.ImageRecordIter(
			path_imgrec=args.test_rec,
			data_shape=(3,224,224),
			batch_size =args.batch_size,
			resize = 126,
			pad = 51,
			mean_r = 123,
			mean_g = 123,
			mean_b = 123,
			preprocess_threads = 16,
			fill_value = 123)
	
	logger.info("test using model {}-{}.params".format(args.prefix,str(args.epoch)))
	sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix,args.epoch)
	mod = mx.mod.Module(sym,data_names=('data',),label_names = ('softmax_label',), logger = logger, context = ctx)
	mod.bind(data_shapes=test_data.provide_data,label_shapes=test_data.provide_label)
	mod.set_params(arg_params, aux_params)

	if DEBUG:
		print(sym.list_arguments())
		arg_shape, out_shape,_ = sym.infer_shape(data=(256,3,224,224))
		print(arg_shape)
		print(out_shape)
		print(test_data.provide_data)
		print(test_data.provide_label)

	
	if DEBUG:
		y = mod.predict(test_data,num_batch = 1).asnumpy()
		score = y.max(axis=1)
		pred = y.argmax(axis=1)
		print(pred)
		print(pred.shape)
		test_data.reset()
		gt = test_data.next()
		gt = gt.label[0].asnumpy()
		print(gt)
		ret = np.vstack([score,pred,gt])
		print(ret.T)
	score = mod.score(test_data,'acc')
	print(score)
	
	
			
if __name__ =='__main__':
	main()
