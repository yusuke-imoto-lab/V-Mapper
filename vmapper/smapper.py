# coding: utf-8
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial.distance
import scipy.sparse
from sklearn import cluster
from sklearn.decomposition import PCA
import sklearn
import vmapper.color
import vmapper.meta_func
from sklearn.manifold import TSNE


## [Simple Mapper]
class smapper():
	def __init__(
		self,
		data,
		num_cube=10,
		d_proj = 2,
		adaptive=True
	):
		overlap=0.5
		overlap += overlap*1.0e-10
		color_id_max = len(vmapper.color.color_dict)
		n,d = data.shape
		data_pca = PCA().fit_transform(data)
		d_pca = data_pca.shape[1]
		data_p = data_pca[:,:d_proj]
		data_p_min = np.min(data_p,axis=0)
		data_p_max = np.max(data_p,axis=0)
		data_p1_min = np.min(data_p.T[0])
		data_p1_max = np.max(data_p.T[0])
		cube_size = (data_p1_max-data_p1_min)/(1+(num_cube-1)*(1-overlap))
		cube_min = np.array([[j*cube_size*overlap+data_p1_min for j in range(num_cube)] for i in range(d_proj)])
		cube_max = np.array([[cube_min[i,j]+cube_size for j in range(num_cube)] for i in range(d_proj)])
		if adaptive == True:
			for i in range(d_proj):
				sorted_val = np.sort(data_p.T[i])
				num_cell_pt = int(n/num_cube)
				bd = np.empty(num_cube+1)
				bd[0] = data_p_min[i]
				bd[num_cube] = data_p_max[i]
				bd[1:-1] = [0.5*(sorted_val[(j+1)*num_cell_pt]+sorted_val[(j+1)*num_cell_pt+1]) for j in range(num_cube-1)]
				cube_min[i,0] = data_p_min[i]-1.0e-8
				cube_max[i,-0] = data_p_max[i]+1.0e-8
				for j in range(num_cube-1):
					cube_max[i,j] = 0.5*(bd[j+1]+bd[j+2])
					cube_min[i,j+1] = 0.5*(bd[j]+bd[j+1])
		data_sub_id_1d = np.empty([d_proj,num_cube],dtype=list)
		data_sub_id_1d = np.array([[np.where((data_p.T[i_dim]>=cube_min[i_dim,i_cube]) & (data_p.T[i_dim]<=cube_max[i_dim,i_cube]))[0] for i_cube in range(num_cube)] for i_dim in range(d_proj)])
		num_cube_all = num_cube**d_proj
		data_sub_id = np.empty(num_cube_all,dtype=list)
		data_sub_id = [data_sub_id_1d[0,i%num_cube] for i in range(num_cube_all)]
		for i in range(num_cube_all):
			for i_dim in range(d_proj-1):
				j = int(i/(num_cube**(i_dim+1))%num_cube)
				data_sub_id[i] = np.intersect1d(data_sub_id[i],data_sub_id_1d[i_dim+1,j])
		data_sub = np.empty(num_cube_all,dtype=list)
		data_sub = [np.array(data_pca[data_sub_id[i]]) for i in range(num_cube_all)]
		data_sub_clstr = np.empty(num_cube_all,dtype=list)
		data_sub_clstr_set = np.empty(num_cube_all,dtype=list)
		data_sub_clstr_num = np.empty(num_cube_all,dtype=int)
		rad = np.empty(num_cube_all,dtype=float)
		rad[:] = cube_size
		if adaptive == True:
			for i in range(num_cube_all):
				i_1d = i%num_cube
				rad[i] = (cube_max[0,i_1d]-cube_min[0,i_1d])
				for i_dim in range(d_proj-1):
					j = int(i/(num_cube**(i_dim+1))%num_cube)
					r = (cube_max[i_dim+1,j]-cube_min[i_dim+1,j])
					if rad[i] < r:
						rad[i] = r
		for i in range(num_cube_all):
			if len(data_sub[i]) == 0:
				data_sub_clstr[i] = []
				data_sub_clstr_num[i] = 0
			else:
				clusterer = cluster.DBSCAN(eps=rad[i], min_samples=1)
				clstr = clusterer.fit(data_sub[i])
				data_sub_clstr[i] = np.array(clusterer.fit_predict(data_sub[i]),dtype=int)
				data_sub_clstr_set[i] = np.unique(data_sub_clstr[i])
				data_sub_clstr_num[i] = len(data_sub_clstr_set[i])
		num_node_all = np.sum(data_sub_clstr_num)
		node_id_set = np.empty(num_node_all,dtype=object)
		node_color_id = np.empty(num_node_all,dtype=int)
		node_name = np.empty(num_node_all,dtype=object)
		num_node = 0
		i_cube = 0
		for i in range(num_cube_all):
			if data_sub_clstr_num[i] > 0:
				i_cube += 1
				i_clstr = 0
				for j in list(data_sub_clstr_set[i]):
					node_id_set[num_node] = np.array(data_sub_id[i][np.where(data_sub_clstr[i]==j)[0]],dtype=int)
					id_dupl = 0
					if True in [str(node_id_set[num_node]) == str(node_id_set[k]) for k in range(num_node)]:
						id_dupl = 1
						break
					node_color_id[num_node] = i_cube%color_id_max
					node_name[num_node] = '%d_%d' % (i_cube,i_clstr+1)
					if id_dupl == 0:
						num_node += 1
						i_clstr += 1
		self.num_node = num_node
		self.node_id_set = node_id_set[:num_node]
		self.node_color_id = node_color_id[:num_node]
		self.node_name = node_name[:num_node]
		self.adjcy_mat = np.empty([self.num_node,self.num_node],dtype=int)
		self.adjcy_mat = np.array([[len(np.intersect1d(self.node_id_set[i],self.node_id_set[j])) for i in range(self.num_node)] for j in range(self.num_node)],dtype=int)
	## ** Visualization by Graphviz
	def out_png(
		self,
		out_dir='.',
		out_file='smapper',
		fig_title='',
		fig_layout='fdp'
	):
		G = Digraph(format="png")
		G.attr('node')
		color_id_max = len(vmapper.color.color_dict)
		for i in range(self.num_node):
			G.node(str(i),label='',fillcolor='%s' % vmapper.color.color_dict[self.node_color_id[i]][0],width='0.075',shape='circle',style='filled')
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					G.edge(str(j),str(i),arrowhead='none',color='darkslategray')
		G.attr(fontsize='8',label=fig_title, dpi='200',layout=fig_layout)
		out_file_png = '%s/%s' % (out_dir,out_file)
		G.render(out_file_png)
		os.remove('%s' % (out_file_png))
	# ** Visualization by Cytoscape 
	def out_cytoscape(
		self,
		sample_name,
		meta_data,
		meta_func,
		out_dir='.',
		out_file='smapper',
		fig_title='',
	):
		out_cyjs = open('%s/%s.cyjs' % (out_dir,out_file),"w")
		data_info = '\"data\":{\"name\":\"%s\"}' % (fig_title)
		node_info = '\"nodes\":['
		sample_name = np.array(sample_name)
		meta_data   = np.array(meta_data)
		num_meta = meta_func.shape[1]
		for i in range(self.num_node):
			node_info += '{\"data\" : {\"id\":\"%s\",\"member\":\"%s\",\"count\":%d' % (self.node_name[i],str(list(sample_name[self.node_id_set[i]])),len(self.node_id_set[i]))
			for k in range(num_meta):
				if meta_func[0,k] in meta_data[0]:
					meta_func_name = meta_func[0,k]
					if meta_func[1,k][-4:] == 'RECODE':
						meta_func_name += '-RECODE'
					meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
					node_info += vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],self.node_id_set[i])
			node_info += '}},'
		node_info = node_info[:-1] + ']'
		edge_info = '\"edges\":['
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					edge_idx_set = np.intersect1d(self.node_id_set[i],self.node_id_set[j])
					edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"%s\","count":%d,\"annotation\":\"strict\"' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],str(list(sample_name[edge_idx_set])),self.adjcy_mat[i,j])
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_func_name = meta_func[0,k]
							if meta_func[1,k][-4:] == 'RECODE':
								meta_func_name += '-RECODE'
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
							edge_info += vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set)
					edge_info += '}},'
		edge_info = edge_info[:-1] + ']'
		out_cyjs.write('{%s,\"elements\":{%s,%s}}' % (data_info,node_info,edge_info))
