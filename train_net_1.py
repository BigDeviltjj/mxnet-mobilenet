import logging
import argparse
import mxnet as mx
import os
import sys
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
	parser.add_argument('--prefix',dest='prefix',help='new model prefix',default=os.path.join(os.getcwd(),'output','exp2','mobilenet'),type=str)
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

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,num_examples,batch_size,begin_epoch):
	iter_refactor = lr_refactor_step
	if lr_refactor_ratio >= 1:
		return (learning_rate, None)
	else:
		lr = learning_rate
		epoch_size = num_examples //batch_size
		for s in iter_refactor:
			if begin_epoch >= s:
				lr *= lr_refactor_ratio
		if lr != learning_rate:
			logging.getLogger().info("adjusted learning rate to {} for epoch{}".format(lr, begin_epoch))
		steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
		if not steps:
			return (lr,None)
		lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step = steps, factor = lr_refactor_ratio)
		return (lr,lr_scheduler)
def get_optimizer_params(optimizer='sgd',learning_rate=None,momentum=0.9,weight_decay=0.0005,lr_scheduler=None,ctx=None,logger=None):
	if optimizer.lower() =='sgd':
		opt = 'sgd'
		optimizer_params={'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
	return opt, optimizer_params
		
def main():
	args = parse_args()
	ctx =  [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]

	logging.basicConfig()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	log_file_path = os.path.join(os.path.dirname(args.prefix),args.log_file)
	fh = logging.FileHandler(log_file_path)
	logger.addHandler(fh)
	
	with open(args.train_list,'r') as f:
		lines = f.readlines()
		
	num_examples = len(lines)
	logger.info("totally {} train examples".format(num_examples))

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
	train_data = mx.io.ImageRecordIter(
			path_imgrec=args.train_rec,
			data_shape=(3,224,224),
			batch_size =args.batch_size,
			mean_r = 123,
			mean_g = 123,
			mean_b = 123,
			preprocess_threads = 16,
			resize_mode = 'force',
			shuffle = True,
			rand_mirror=True)
	test_data = mx.io.ImageRecordIter(
			path_imgrec=args.test_rec,
			data_shape=(3,224,224),
			batch_size =args.batch_size,
			mean_r = 123,
			mean_g = 123,
			mean_b = 123,
			preprocess_threads = 16,
			resize_mode = 'force')


	sym = mobilenet.get_symbol(2,args.alpha)
	if DEBUG:
		train_data.reset()
		it = train_data.next()
		print(it.data[0])
		print(mx.nd.sum(it.data[0]).asscalar())
		print(it.data[0].shape)
		print(sym.list_arguments())
		arg_shape, out_shape,_ = sym.infer_shape(data=(256,3,224,224))
		print(arg_shape)
		print(out_shape)
	begin_epoch = 0
	if args.resume > 0:
		logger.info("Resume training with gpu {} from epoch {}",format(args.gpus,args.resume))
		_,net_args,net_auxs = mx.model.load_checkpoint(args.prefix,resmue)
		begin_epoch = resume
	else:
		logger.info("starting training from scratch with gpu {}".format(args.gpus))
		net_args = None
		net_auxs = None
	mod = mx.mod.Module(sym,label_names = ('softmax_label',), logger = logger, context = ctx)

	batch_end_callback = []
	print(args.frequent)
	batch_end_callback.append(mx.callback.Speedometer(train_data.batch_size, frequent = args.frequent))
	lr_refactor_ratio = [int(args.end_epoch * i / 3) for i in range(1,3)]
	logger.info("learning rate decay at epoch{}".format(lr_refactor_ratio))
	lr,lr_scheduler = get_lr_scheduler(args.learning_rate, lr_refactor_ratio,0.1, num_examples,args.batch_size,begin_epoch)
	opt, opt_params = get_optimizer_params(optimizer = 'sgd', learning_rate = lr, momentum = 0.9, weight_decay = 0.0005, lr_scheduler = lr_scheduler,ctx = ctx, logger = logger)
	mod.fit(train_data,test_data, 
		eval_metric='acc',
		batch_end_callback=batch_end_callback,
		epoch_end_callback=[mx.callback.do_checkpoint(args.prefix,period = 5)],
		optimizer = opt,
		optimizer_params = opt_params,
		begin_epoch = args.begin_epoch,
		num_epoch = args.end_epoch,
		initializer = mx.init.Xavier(),
		arg_params=net_args,
		aux_params=net_auxs,
		allow_missing = False)
			
if __name__ =='__main__':
	main()
