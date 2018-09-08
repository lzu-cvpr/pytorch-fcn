import torch
import sys
import caffe

# wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
caffe_model_path = './fcn8s-heavy-pascal.caffemodel'

# wget https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn8s/deploy.prototxt
caffe_prototxt = './deploy.prototxt'

caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)

model_torch_dict={}
for name, p in caffe_model.params.iteritems():
	model_torch_dict[name+'.weight'] = torch.from_numpy(p[0].data)
	if len(p) == 2:
		model_torch_dict[name+'.bias'] = torch.from_numpy(p[1].data)
torch.save(model_torch_dict, './fcn8s_from_caffe.pth')


#____validation____
torch_model = torch.load('./fcn8s_from_caffe.pth')
c_data = caffe_model.params['conv1_1'][0].data
p_data = torch_model['conv1_1.weight']

print('conv1_1 weight')
print( 'caffe_shape:', c_data.shape,
       'pytorch_shape:', p_data.shape,
       'caffe_max: [%.30f]' % c_data.max(),
       'pytorch_max: [%.30f]' % p_data.max(),
       'caffe type',type(c_data),
       'pytorch type', type(p_data))
