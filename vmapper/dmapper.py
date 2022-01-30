# coding: utf-8
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial.distance
import scipy.sparse
import vmapper.mapper as mapper
from sklearn import cluster
from sklearn.decomposition import PCA
import sklearn
import vmapper.color
import vmapper.meta_func
from sklearn.manifold import TSNE


class dmapper(mapper):
	def __init__(
		self,
		data,
		data_p,
		flow_data,
		num_cube=10,
		overlap=0.5,
		clusterer=cluster.DBSCAN(eps=3.0, min_samples=1),
		adaptive=True
	):
		mp = mapper(data,data_p,num_cube,overlap,clusterer,adaptive=adaptive)
		num_clstr = mp.num_node
		clstr_id_set = np.array(mp.node_id_set)
		clstr_sub_name = np.array(mp.node_name)
		clstr_sub_set = np.empty(num_clstr,dtype=list)
		clstr_sub_num = np.empty(num_clstr,dtype=int)
		clstr_sub_set = [np.unique(flow_data[clstr_id_set[i]]) for i in range(num_clstr)]
		clstr_sub_num[:] = len(clstr_sub_set[:])
		num_node = np.sum(clstr_sub_num)
		node_id_set = np.empty(num_node,dtype=list)
		node_clstr_id = np.empty(num_node,dtype=list)
		node_flow_id = np.empty(num_node,dtype=object)
		node_name = np.empty(num_node,dtype=object)
		i_node = 0
		i_cube = 0
		for i_clstr in range(num_clstr):
			i_clstr_sub = 0
			for j in list(clstr_sub_set[i_clstr]):
				node_id_set[i_node] = clstr_id_set[i_clstr][np.where(flow_data[clstr_id_set[i_clstr]]==j)[0]]
				id_dupl = 0
				clstr_list = [str(node_id_set[i_node]) == str(node_id_set[k]) for k in range(i_node)]
				if True in clstr_list:
					id_dupl = 1
					j_node = clstr_list.index(True)
					node_clstr_id[j_node].append(i_clstr)
				if id_dupl == 0:
					node_clstr_id[i_node] = [i_clstr]
					node_flow_id[i_node] = j
					node_name[i_node] = '%s_%d' % (str(clstr_sub_name[i_clstr]),i_clstr_sub+1)
					i_node += 1
					i_clstr_sub += 1
		num_node = i_node
		self.num_node = i_node
		self.adjcy_mat = np.empty([num_node,num_node],dtype=int)
		self.adjcy_mat = np.array([[len(np.intersect1d(node_id_set[i],node_id_set[j])) for i in range(num_node)] for j in range(num_node)],dtype=int)
		self.node_clstr_id = node_clstr_id
		self.node_flow_id = node_flow_id
		self.node_name = node_name
		self.node_id_set = node_id_set
	## ** Visualization by Graphviz
	def out_png(
		self,
		flow_data,
		flow_graph_node,
		flow_graph_dis,
		out_dir='.',
		out_file='dmapper',
		fig_title='',
		fig_layout='fdp'
	):
		color_id_max = len(vmapper.color.color_dict)
		flow_data_set = np.unique(flow_data)
		G = Digraph(format="png")
		G.attr('node')
		for i in range(self.num_node):
			i_color = np.where(flow_graph_node==self.node_flow_id[i])[0][0]%color_id_max + 1
			G.node(str(i),label='',fillcolor='%s' % vmapper.color.color_dict[i_color][0],width='0.075',shape='circle',style='filled')
		for i in range(self.num_node):
			for j in range(self.num_node):
				if self.adjcy_mat[i,j] > 0 and i>j:
					i_color = np.where(flow_graph_node==self.node_flow_id[i])[0][0]%color_id_max + 1
					G.edge(str(i),str(j),dir='both',arrowtail='none',arrowhead='none',arrowsize='0.2',color='%s' % vmapper.color.color_dict[i_color][0])
				elif len(np.intersect1d(self.node_clstr_id[i],self.node_clstr_id[j])):
					i_graph = np.where(flow_graph_node==self.node_flow_id[i])[0][0]
					j_graph = np.where(flow_graph_node==self.node_flow_id[j])[0][0]
					dis = flow_graph_dis[i_graph,j_graph]
					if dis == 1:
						G.edge(str(i),str(j),arrowhead='normal',arrowsize='0.4',color='black')
					elif dis == np.inf:
						G.edge(str(i),str(j),arrowhead='none',arrowsize='0.2',color='darkslategray',style='dashed')
					elif dis > 1:
						G.edge(str(i),str(j),arrowhead='empty',arrowsize='0.4',color='gray')
		G.attr(fontsize='9',label=fig_title, dpi='200',layout=fig_layout)
		out_file_png = '%s/%s' % (out_dir,out_file)
		G.render(out_file_png)
		os.remove('%s' % (out_file_png))
	# ** Visualization by Cytoscape 
	def out_cytoscape(
		self,
		sample_name,
		meta_data,
		meta_func,
		flow_data,
		flow_graph_node,
		flow_graph_dis,
		out_dir='.',
		out_file='dmapper',
		fig_title='',
	):
		out_cyjs = open('%s/%s.cyjs' % (out_dir,out_file),"w")
		data_info = '\"data\":{\"name\":\"%s\"}' % (fig_title)
		node_info = '\"nodes\":['
		sample_name = np.array(sample_name)
		meta_data   = np.array(meta_data)
		num_meta = meta_func.shape[1]
		for i in range(self.num_node):
			node_info += '{\"data\" : {\"id\":\"%s\",\"member\":\"%s\",\"count\":%d,\"flow_id\":\"%s\"' % (self.node_name[i],str(list(sample_name[self.node_id_set[i]])),len(self.node_id_set[i]),str(self.node_flow_id[i]))
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
			for j in range(self.num_node):
				if self.adjcy_mat[i,j] > 0 and i>j:
					edge_idx_set = np.intersect1d(self.node_id_set[i],self.node_id_set[j])
					edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"%s\","count":%d,\"annotation\":\"strict\",\"flow_id\":\"%s\",\"distance\":0' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],str(list(sample_name[edge_idx_set])),self.adjcy_mat[i,j],str(self.node_flow_id[i]))
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_func_name = meta_func[0,k]
							if meta_func[1,k][-4:] == 'RECODE':
								meta_func_name += '-RECODE'
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
							edge_info += vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set)
					edge_info += '}},'
				elif len(np.intersect1d(self.node_clstr_id[i],self.node_clstr_id[j])):
					i_graph = np.where(flow_graph_node==self.node_flow_id[i])[0][0]
					j_graph = np.where(flow_graph_node==self.node_flow_id[j])[0][0]
					dis = flow_graph_dis[i_graph,j_graph]
					if dis == 1:
						edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"[]\","count":0,\"annotation\":\"arrow\",\"flow_id\":\"%s-%s\",\"distance\":%d' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],str(self.node_flow_id[i]),str(self.node_flow_id[j]),int(dis))
					elif dis == np.inf:
						edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"[]\","count":0,\"annotation\":\"dashed\",\"flow_id\":\"%s-%s\",\"distance\":null' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],str(self.node_flow_id[i]),str(self.node_flow_id[j]))
					elif dis > 1:
						edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"[]\","count":0,\"annotation\":\"two_arrow\",\"flow_id\":\"%s-%s\",\"distance\":%d' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],str(self.node_flow_id[i]),str(self.node_flow_id[j]),int(dis))
					else:
						continue
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func[0,k])[0][0]]
							edge_idx_set = np.hstack((self.node_id_set[i],self.node_id_set[j]))
							edge_info += vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set)
					edge_info += '}},'
		edge_info = edge_info[:-1] + ']'
		out_cyjs.write('{%s,\"elements\":{%s,%s}}' % (data_info,node_info,edge_info))

def plot_flow(
		flow_data,
		out_dir='.',
		file_name='flow',
		fig_title=''
	):
	flow_data_mat = np.array(np.where(flow_data[1:,1:]=='',0,flow_data[1:,1:]),dtype=int)
	flow_data_name = np.array(flow_data[0,1:],dtype=object)
	num_node = len(flow_data_name)
	flow_data_set = np.unique(flow_data)
	G = Digraph(format="png")
	G.attr('node')
	color_id_max = len(vmapper.color.color_dict)
	for i in range(num_node):
		i_color = i%color_id_max
		G.node(str(i),label='%s' % (flow_data_name[i]),fillcolor='%s' % vmapper.color.color_dict[i+1][0],fontcolor='%s' % vmapper.color.color_dict[i+1][1],width='0.075',shape='square',style='filled')
	for i in range(num_node):
		for j in range(num_node):
			if flow_data_mat[i,j] == 1:
				G.edge(str(i),str(j),arrowhead='normal',arrowsize='1',color='black')
	G.attr(fontsize='9',label=fig_title, dpi='200')
	out_file = '%s/%s' % (out_dir,file_name)
	G.render(out_file)
	os.remove('%s' % (out_file))

def flow_dis(adj_mat):
	n = adj_mat.shape[0]
	graph_dis = np.zeros([n,n],dtype=float)
	for i in range(n):
		for j in range(n):
			if adj_mat[i,j] == '':
				graph_dis[i,j] = np.inf
			else:
				graph_dis[i,j] = float(adj_mat[i,j])
	dis = 2
	while(dis<n):
		for i in range(n):
			for j in range(n):
				if graph_dis[i,j] != np.inf:
					for k in range(n):
						if graph_dis[j,k] != np.inf:
							if graph_dis[i,k] > graph_dis[i,j]+graph_dis[j,k]:
								graph_dis[i,k] =  graph_dis[i,j]+graph_dis[j,k]
		dis += 1
	
	for i in range(n):
		for j in range(n):
			if graph_dis[i,j] == np.inf and graph_dis[j,i] != np.inf:
				graph_dis[i,j] = -graph_dis[j,i]
	
	for i in range(n):
		if adj_mat[i,i] == '1':
			graph_dis[i,i] = 1
		else:
			graph_dis[i,i] = 0
	
	return graph_dis
