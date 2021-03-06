
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

def str2intlist(v):
	lst = v.split(",")
	lst_parsed = []
	for l in lst:
		try:
			lst_parsed += [int(l)]
		except ValueError:
			lst_parsed += [float(l)]
	return lst_parsed

def str2list(v):
	return v.split(",")

def str2bool2list(v):
	if 'true' in v.lower() or 'false' in v.lower():
		return str2bool(v)
	return str2intlist(v)
	

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_scale_size', type=int, default=64,
                     help='input image will be resized with the given value as width and height')
net_arg.add_argument('--conv_hidden_num', type=int, default=128, help='n in the paper')
net_arg.add_argument('--z_num', type=int, default=128)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='CelebA')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--load_attributes', type=str, default=None)
data_arg.add_argument('--use_channels', type=str2bool2list, default=None)
data_arg.add_argument('--num_worker', type=int, default=12)

# Data preprocessing/augmentation
prep_arg = add_argument_group('Preprocessing')
prep_arg.add_argument('--flips', type=str2bool, default=False)
prep_arg.add_argument('--rotate_angle', type=int, default=0)
prep_arg.add_argument('--take_log', type=str2bool2list, default=False)
prep_arg.add_argument('--normalize', type=str2bool, default=False)
prep_arg.add_argument('--normalize_channels', type=str2bool, default=False)
prep_arg.add_argument('--filter_by_pop', type=str2list, default=None)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--model_type', type=str, default='began')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=500000)
train_arg.add_argument('--lr_update_step', type=int, default=10000)
train_arg.add_argument('--L2', type=float, default=0.0)
train_arg.add_argument('--L1', type=float, default=0.0)
train_arg.add_argument('--lr_G', type=float, default=0.0001)
train_arg.add_argument('--lr_D', type=float, default=0.0001)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--comment', type=str, default=None, help="short comment to explain or identify experiment purpose/characteristics")
data_arg.add_argument('--src_names', type=str, default=None)
data_arg.add_argument('--save_image_channels', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--gpu_ids', type=str2intlist, default=[])
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
