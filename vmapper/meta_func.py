# coding: utf-8
import numpy as np

def union(inp):
	if all([e == inp[0] for e in inp[1:]]):
		char = inp[0]
	else:
		char = 'Mixed'
	return char

def add_meta_info(
	meta_name,
	meta_data,
	func,
	id_set,
	id_light=False
):
	info  = ''
	if func  == 'list':
		info_temp = '%s' % str(meta_data[id_set][0])
		for i in range(len(meta_data[id_set])-1):
			info_temp += ', %s' % str(meta_data[id_set][i+1])
		if id_light:
			info += info_temp
		else:
			info += '\"%s\"' % info_temp
	elif func  == 'list-percent':
		meta_list = meta_data[id_set]
		meta_list_uni = np.unique(meta_list)
		meta_list_uni_count = [np.count_nonzero(meta_list==meta_list_uni[i]) for i in range(len(meta_list_uni))]
		percent = ['{:.0%}'.format(meta_list_uni_count[i]/len(meta_list)) for i in range(len(meta_list_uni))]
		info_set = '%s' % str(meta_list_uni[0])
		info_per = '%s' % str(percent[0])
		for i in range(len(meta_list_uni)-1):
			info_set += ', %s' % str(meta_list_uni[i+1])
			info_per += ', %s' % str(percent[i+1])
		if id_light:
			info += '%s / %s' % (info_set,info_per)
		else:
			info += '\"%s / %s\"' % (info_set,info_per)
	elif func  == 'set':
		meta_list = meta_data[id_set]
		meta_list_uni = np.unique(meta_list)
		meta_list_uni_count = [np.count_nonzero(meta_list==meta_list_uni[i]) for i in range(len(meta_list_uni))]
		if id_light:
			info += '%s' % (str(list(meta_list_uni))[1:-1])
		else:
			info += '\"%s\"' % (str(list(meta_list_uni))[1:-1])
	elif func  == 'dominant':
		meta_list = meta_data[id_set]
		meta_list_uni = np.unique(meta_list)
		meta_list_uni_count = [np.count_nonzero(meta_list==meta_list_uni[i]) for i in range(len(meta_list_uni))]
		max_id = np.array(np.where(meta_list_uni_count==np.max(meta_list_uni_count))[0])
		info_temp = '%s' % str(meta_list_uni[max_id][0])
		for i in range(len(meta_list_uni[max_id])-1):
			info_temp += ', %s' % str(meta_list_uni[max_id][i+1])
		if id_light:
			info += info_temp
		else:
			info += '\"%s\"' % info_temp
	elif func  == 'dominant-percent':
		meta_list = meta_data[id_set]
		meta_list_uni = np.unique(meta_list)
		meta_list_uni_count = [np.count_nonzero(meta_list==meta_list_uni[i]) for i in range(len(meta_list_uni))]
		max_id = np.array(np.where(meta_list_uni_count==np.max(meta_list_uni_count))[0])
		max_percent = ['{:.0%}'.format(meta_list_uni_count[max_id[i]]/len(meta_list)) for i in range(len(max_id))]
		if id_light:
			info += '%s / %s' % (str(list(meta_list_uni[max_id]))[1:-1],str(list(max_percent))[1:-1])
		else:
			info += '\"%s / %s\"' % (str(list(meta_list_uni[max_id]))[1:-1],str(list(max_percent))[1:-1])
	elif func[:7]  == 'average':
		info += '%0.4f' % (np.mean(np.array(meta_data[id_set],dtype=float),axis=0))
	elif func[:4]  == 'mean':
		info += '%0.4f' % (np.mean(np.array(meta_data[id_set],dtype=float),axis=0))
	elif func[:6]  == 'median':
		info += '%0.4f' % (np.median(np.array(meta_data[id_set],dtype=float),axis=0))
	elif func[:3]  == 'max':
		info += '%0.4f' % (np.max(np.array(meta_data[id_set],dtype=float),axis=0))
	elif func[:3]  == 'min':
		info += '%0.4f' % (np.min(np.array(meta_data[id_set],dtype=float),axis=0))
	elif func  == 'union':
		meta_data[id_set]
		if id_light:
			info += '%s' % (union(meta_data[id_set]))
		else:
			info += '\"%s\"' % (union(meta_data[id_set]))
	elif func[:9]  == 'inclusion':
		if func[10:] in meta_data[id_set]:
			info += 'true'
		else:
			info += 'false'
	return info
