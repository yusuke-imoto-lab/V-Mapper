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

## [Mapper]
class mapper():
	def __init__(self,
		data,
		data_p,
		num_cube=10,
		overlap=0.5,
		clusterer=cluster.DBSCAN(eps=3.0, min_samples=1),
		adaptive=True
		):
		""" 
		V-Mapper (Resolution of curse of dimensionality in single-cell data analysis). A noise reduction method for single-cell sequencing data. 
		
		Parameters
		----------
		data : ndarray/anndata of shape (n_samples, n_features)
			Tranceforming single-cell sequencing data matrix (row:cell, culumn:gene/peak).
		
		Attributes
		----------
		log_ : dict
			Running log.
		
		"""
		color_id_max = len(vmapper.color.color_dict)
		Num,Dim = data_p.shape
		data_p_min = np.min(data_p,axis=0)
		data_p_max = np.max(data_p,axis=0)
		cube_size  = np.empty(Dim)
		overlap += overlap*1.0e-5
		cube_size[:] = (data_p_max[:]-data_p_min[:])/(1+(num_cube-1)*(1-overlap))
		cube_min = np.array([[j*cube_size[i]*overlap+data_p_min[i] for j in range(num_cube)] for i in range(Dim)])
		cube_max = np.array([[cube_min[i,j]+cube_size[i] for j in range(num_cube)] for i in range(Dim)])
		if adaptive == True:
			for i in range(Dim):
				sorted_val = np.sort(data_p.T[i])
				num_cell_pt = int(Num/num_cube)
				bd = np.empty(num_cube+1)
				bd[0] = data_p_min[i]
				bd[1:-1] = [0.5*(sorted_val[(j+1)*num_cell_pt]+sorted_val[(j+1)*num_cell_pt+1]) for j in range(num_cube-1)]
				bd[num_cube] = data_p_max[i]
				cube_min[i,0] = data_p_min[i]-1.0e-10
				cube_max[i,num_cube-1] = data_p_max[i]+1.0e-10
				for j in range(num_cube-1):
					cube_max[i,j] = 0.5*(bd[j+1]+bd[j+2])
					cube_min[i,j+1] = 0.5*(bd[j]+bd[j+1])
		data_sub_id_1d = np.empty([Dim,num_cube],dtype=list)
		data_sub_id_1d = np.array([[np.where((data_p.T[i_dim]>=cube_min[i_dim,i_cube]) & (data_p.T[i_dim]<=cube_max[i_dim,i_cube]))[0] for i_cube in range(num_cube)] for i_dim in range(Dim)],dtype=object)
		num_cube_all = num_cube**Dim
		data_sub_id = np.empty(num_cube_all,dtype=list)
		data_sub_id = [data_sub_id_1d[0,i%num_cube] for i in range(num_cube_all)]
		for i in range(num_cube_all):
			for i_dim in range(Dim-1):
				j = int(i/(num_cube**(i_dim+1))%num_cube)
				data_sub_id[i] = np.intersect1d(data_sub_id[i],data_sub_id_1d[i_dim+1,j])
		data_sub = np.empty(num_cube_all,dtype=list)
		data_sub = [np.array(data[data_sub_id[i]]) for i in range(num_cube_all)]
		data_sub_clstr = np.empty(num_cube_all,dtype=list)
		data_sub_clstr_set = np.empty(num_cube_all,dtype=list)
		data_sub_clstr_num = np.empty(num_cube_all,dtype=int)
		for i in range(num_cube_all):
			if len(data_sub[i]) == 0:
				data_sub_clstr[i] = []
				data_sub_clstr_num[i] = 0
			else:
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
		self.Num = Num
		self.num_node = num_node
		self.node_id_set = node_id_set[:num_node]
		self.node_color_id = node_color_id[:num_node]
		self.node_name = node_name[:num_node]
		self.adjcy_mat = np.empty([self.num_node,self.num_node],dtype=int)
		self.adjcy_mat = np.array([[len(np.intersect1d(self.node_id_set[i],self.node_id_set[j])) for i in range(self.num_node)] for j in range(self.num_node)],dtype=int)
		self.data_p = data_p
		self.data_p_min = data_p_min
		self.data_p_max = data_p_max
		self.cube_min = cube_min
		self.cube_max = cube_max
	## ** Visualization by Graphviz
	def out_graph_graphviz(
		self,
		out_dir='.',
		out_file='mapper',
		fig_title='',
		fig_fmt='png',
		fig_layout='fdp'
	):
		G = Digraph(format=fig_fmt)
		G.attr('node')
		color_id_max = len(vmapper.color.color_dict)
		for i in range(self.num_node):
			i_color = i%color_id_max
			G.node(str(i),label='',fillcolor='%s' % vmapper.color.color_dict[self.node_color_id[i_color]][0],width='0.075',shape='circle',style='filled')
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					G.edge(str(j),str(i),arrowhead='none',color='darkslategray')
		G.attr(fontsize='8',label=fig_title, dpi='200',layout=fig_layout)
		out_file = '%s/%s_graph' % (out_dir,out_file)
		G.render(out_file)
		os.remove('%s' % (out_file))
	## ** Visualization by PCA
	def out_graph_pca(
		self,
		data,
		out_dir='.',
		out_file='mapper',
		fig_title='',
		fig_fmt='png'
	):
		n,d = data.shape
		data_node = np.empty([self.num_node,d],dtype=float)
		for i in range(self.num_node):
			data_node[i] = np.mean(data[self.node_id_set[i]],axis=0)
		data_node_pca = PCA(n_components=2).fit_transform(data_node)
		#
		n_edge = np.sum(np.where(self.adjcy_mat>0,1,0))
		data_edge_pca = np.empty([n_edge,4],dtype=float)
		i_edge = 0
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					data_edge_pca[i_edge,0:2] = data_node_pca[i]
					data_edge_pca[i_edge,2:4] = data_node_pca[j]-data_node_pca[i]
					i_edge += 1
		data_edge_pca = data_edge_pca[:i_edge]   
		#
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(1,1,1)
		ax.scatter(data_node_pca[:,0],data_node_pca[:,1],zorder=1,s=10,color='black')
		ax.quiver(data_edge_pca[:,0],data_edge_pca[:,1],data_edge_pca[:,2],data_edge_pca[:,3],color='gray',angles='xy',scale_units='xy',width=0.002,scale=1,alpha=0.5,zorder=0,headwidth=0,headlength=0,headaxislength=0)
		ax.set_title('Mapper (PCA coordinate)')
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		plt.savefig('%s/%s_pca.%s' % (out_dir,out_file,fig_fmt))
	## ** Visualization by tSNE
	def out_graph_tsne(
		self,
		data,
		out_dir='.',
		out_file='mapper',
		fig_title='',
		fig_fmt='png'
	):
		n,d = data.shape
		data_node = np.empty([self.num_node,d],dtype=float)
		for i in range(self.num_node):
			data_node[i] = np.mean(data[self.node_id_set[i]],axis=0)
		data_node_tsne = TSNE(n_components=2).fit_transform(data_node)
		#
		n_edge = np.sum(np.where(self.adjcy_mat>0,1,0))
		flow = np.empty([n_edge,4],dtype=float)
		i_edge = 0
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					flow[i_edge,0:2] = data_node_tsne[i]
					flow[i_edge,2:4] = data_node_tsne[j]-data_node_tsne[i]
					i_edge += 1
		flow = flow[:i_edge]   
		#
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(1,1,1)
		ax.scatter(data_node_tsne[:,0],data_node_tsne[:,1],zorder=1,s=10,color='black')
		ax.quiver(flow[:,0],flow[:,1],flow[:,2],flow[:,3],color='gray',angles='xy',scale_units='xy',width=0.002,scale=1,alpha=0.5,zorder=0,headwidth=0,headlength=0,headaxislength=0)
		ax.set_title('Mapper (tSNE coordinate)')
		ax.set_xlabel('tSNE1')
		ax.set_ylabel('tSNE2')
		plt.savefig('%s/%s_tsne.%s' % (out_dir,out_file,fig_fmt))
	# ** Visualization by Cytoscape 
	def out_cytoscape(
		self,
		data,
		sample_name,
		meta_data,
		meta_func,
		out_dir='.',
		out_file='mapper',
		fig_title='',
	):
		n,d = data.shape
		data_node = np.empty([self.num_node,d],dtype=float)
		for i in range(self.num_node):
			data_node[i] = np.mean(data[self.node_id_set[i]],axis=0)
		data_node_pca = PCA(n_components=2).fit_transform(data_node)
		data_node_tsne = TSNE(n_components=2).fit_transform(data_node)
		out_cyjs = open('%s/%s.cyjs' % (out_dir,out_file),"w")
		data_info = '\"data\":{\"name\":\"%s\"}' % (fig_title)
		node_info = '\"nodes\":['
		sample_name = np.array(sample_name)
		meta_data   = np.array(meta_data)
		num_meta = meta_func.shape[1]
		for i in range(self.num_node):
			member = '%s' % sample_name[self.node_id_set[i]][0]
			for j in range(len(sample_name[self.node_id_set[i]])-1):
				member += ', %s' % sample_name[self.node_id_set[i]][j+1] 
			node_info += '{\"data\" : {\"id\":\"%s\",\"member\":\"%s\",\"count\":%d,\"PCA1\":%.5f,\"PCA2\":%.5f,\"tSNE1\":%.5f,\"tSNE2\":%.5f' % (self.node_name[i],member,len(self.node_id_set[i]),data_node_pca[i,0],data_node_pca[i,1],data_node_tsne[i,0],data_node_tsne[i,1])
			for k in range(num_meta):
				if meta_func[0,k] in meta_data[0]:
					meta_func_name = meta_func[0,k]
					if meta_func[1,k][-4:] == 'RECODE':
						meta_func_name += '-RECODE'
					meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
					node_info += ',\"%s[%s]\":%s' % (meta_func[0,k],meta_func[1,k],vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],self.node_id_set[i]))
			node_info += '}},'
		node_info = node_info[:-1] + ']'
		edge_info = '\"edges\":['
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					edge_idx_set = np.intersect1d(self.node_id_set[i],self.node_id_set[j])
					member = '%s' % sample_name[edge_idx_set][0]
					for k in range(len(sample_name[edge_idx_set])-1):
						member += ', %s' % sample_name[edge_idx_set][k+1]
					edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"%s\","count":%d,\"annotation\":\"strict\"' % (self.node_name[i],self.node_name[j],self.node_name[i],self.node_name[j],member,self.adjcy_mat[i,j])
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_func_name = meta_func[0,k]
							if meta_func[1,k][-4:] == 'RECODE':
								meta_func_name += '-RECODE'
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
							edge_info += ',\"%s[%s]\":%s' % (meta_func[0,k],meta_func[1,k],vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set))
					edge_info += '}},'
		edge_info = edge_info[:-1] + ']'
		out_cyjs.write('{%s,\"elements\":{%s,%s}}' % (data_info,node_info,edge_info))
	def out_cytoscape_light(
		self,
		sample_name,
		meta_data,
		meta_func,
		out_dir='.',
		out_file='mapper',
		fig_title='',
	):
		sample_name = np.array(sample_name)
		meta_data   = np.array(meta_data)
		num_meta = meta_func.shape[1]
		num_edge = int(0.5*(np.sum(np.sign(self.adjcy_mat))-self.num_node))
		n_default = 3
		
		sif_data = np.empty([self.num_node+num_edge,1],dtype=object)
		
		meta_data_out_node = np.empty([self.num_node+1,num_meta+n_default],dtype=object)
		meta_data_out_node[0,0] = 'name'
		meta_data_out_node[0,1] = 'member'
		meta_data_out_node[0,2] = 'count'
		meta_data_out_node[0,n_default:] = ['%s[%s]' % (meta_func[0][i],meta_func[1][i]) for i in range(num_meta)]
		
		
		meta_data_out_edge = np.empty([num_edge+1,num_meta+n_default],dtype=object)
		meta_data_out_edge[0,0] = 'name'
		meta_data_out_edge[0,1] = 'member'
		meta_data_out_edge[0,2] = 'count'
		meta_data_out_edge[0,n_default:] = ['%s[%s]' % (meta_func[0][i],meta_func[1][i]) for i in range(num_meta)]
		
		for i in range(self.num_node):
			sif_data[i] = self.node_name[i]
			meta_data_out_node[i+1,0] = self.node_name[i]
			member = '%s' % sample_name[self.node_id_set[i]][0]
			for j in range(len(sample_name[self.node_id_set[i]])-1):
				member += ', %s' % sample_name[self.node_id_set[i]][j+1] 
			meta_data_out_node[i+1,1] = member
			meta_data_out_node[i+1,2] = len(self.node_id_set[i])
			
			for k in range(num_meta):
				if meta_func[0,k] in meta_data[0]:
					meta_func_name = meta_func[0,k]
					if meta_func[1,k][-4:] == 'RECODE':
						meta_func_name += '-RECODE'
					meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
					meta_data_out_node[i+1,k+n_default] = vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],self.node_id_set[i],id_light=True)
		i_edge = 0
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					edge_idx_set = np.intersect1d(self.node_id_set[i],self.node_id_set[j])
					sif_data[self.num_node+i_edge] = '%s - %s' % (self.node_name[i],self.node_name[j])
					meta_data_out_edge[i_edge+1,0] = '%s (-) %s' % (self.node_name[i],self.node_name[j])
					member = '%s' % sample_name[edge_idx_set][0]
					for k in range(len(sample_name[edge_idx_set])-1):
						member += ', %s' % sample_name[edge_idx_set][k+1]
					meta_data_out_edge[i_edge+1,1] = member
					meta_data_out_edge[i_edge+1,2] = self.adjcy_mat[i,j]
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_func_name = meta_func[0,k]
							if meta_func[1,k][-4:] == 'RECODE':
								meta_func_name += '-RECODE'
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
							meta_data_out_edge[i_edge+1,k+n_default] = vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set,id_light=True)
					i_edge += 1
		np.savetxt('%s/%s.sif' % (out_dir,out_file),sif_data,delimiter = ",", fmt = "%s")
		np.savetxt('%s/%s_node.tsv' % (out_dir,out_file),meta_data_out_node,delimiter="\t", fmt = "%s")
		np.savetxt('%s/%s_edge.tsv' % (out_dir,out_file),meta_data_out_edge,delimiter="\t", fmt = "%s")
	def out_fig_cube_docomp(
		self,
		out_dir='.',
		out_file='mapper',
		fig_title=''
	):
		file_name = '%s/%s' % (out_dir,out_file)
		i=0
		
		## - PCA3D for Original Data 1 & 2 
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.scatter(self.data_p.T[0],self.data_p.T[1],c='blue',marker='.')
		ptb1 = 5e-3*(self.data_p_max[i]-self.data_p_min[i])
		ptb2 = 5e-3*(self.data_p_max[i+1]-self.data_p_min[i+1])
		color_id_max = len(vmapper.color.color_dict)
		for j in range(len(self.cube_min[i])):
			c = vmapper.color.color_dict[j][0]
			l = self.cube_min[i,j]#-ptb1
			u = self.cube_max[i,j]#+ptb2
			h = 0.5*(u+l)
			ap = self.data_p_min[i+1]-(1+0.75*(j%2))*2e-2*(self.data_p_max[i+1]-self.data_p_min[i+1])
			ax.axvline(l, ls = "--", linewidth=1,color=c)
			ax.axvline(u, ls = "--", linewidth=1,color=c)
			ax.annotate('',xy=(l,ap),xytext=(h,ap),
				arrowprops=dict(shrink=0, width=0.5,headwidth=5,headlength=5,
					connectionstyle='arc3',facecolor=c, edgecolor=c))
			ax.annotate('',xy=(u,ap),xytext=(h,ap),
				arrowprops=dict(shrink=0, width=0.5,headwidth=5,headlength=5,
					connectionstyle='arc3',facecolor=c, edgecolor=c))
			l = self.cube_min[i+1,j]#-ptb1
			u = self.cube_max[i+1,j]#+ptb2
			h = 0.5*(u+l)
			ap = self.data_p_min[i]-(1+0.75*(j%2))*2e-2*(self.data_p_max[i]-self.data_p_min[i])
			ax.axhline(l, ls = "--", linewidth=1, color=c)
			ax.axhline(u, ls = "--", linewidth=1, color=c)
			ax.annotate('',xy=(ap,l),xytext=(ap,h),
				arrowprops=dict(shrink=0, width=0.5,headwidth=5,headlength=5,
					connectionstyle='arc3',facecolor=c, edgecolor=c))
			ax.annotate('',xy=(ap,u),xytext=(ap,h),
				arrowprops=dict(shrink=0, width=0.5,headwidth=5,headlength=5,
					connectionstyle='arc3',facecolor=c, edgecolor=c))
		plt.title(fig_title)
		ax.set_xlabel('%d' % (i+1))
		ax.set_ylabel('%d' % (i+2))
		plt.savefig('%s_cube_%1d%1d.png' % (file_name,i+1,i+2))
		plt.close()
	## ** Visualization with velocity by Graphviz
	def out_graph_graphviz_V(
		self,
		data,
		data_vel,
		out_dir='.',
		out_file='vmapper',
		fig_title='',
		fig_fmt='png',
		fig_layout='fdp'
	):
		G = Digraph(format=fig_fmt)
		G.attr('node')
		color_id_max = len(vmapper.color.color_dict)
		for i in range(self.num_node):
			i_color = i%color_id_max
			G.node(str(i),label='',fillcolor='%s' % vmapper.color.color_dict[self.node_color_id[i_color]][0],width='0.05',shape='circle',style='filled')
		for i in range(self.num_node):
			for j in range(i+1,self.num_node):
				if self.adjcy_mat[i,j] > 0:
					idx_i = self.node_id_set[i]
					idx_j = self.node_id_set[j]
					idx_ij = np.intersect1d(idx_i,idx_j)
					mean_i = np.mean(data[idx_i],axis=0)
					mean_j = np.mean(data[idx_j],axis=0)
					rltv_vel = mean_j-mean_i
					vel = np.dot(rltv_vel,np.mean(data_vel[idx_ij],axis=0))
					if vel>0:
						G.edge(str(i),str(j),arrowhead='normal',arrowsize='0.2',color='darkslategray')
					else:
						G.edge(str(j),str(i),arrowhead='normal',arrowsize='0.2',color='darkslategray')
		G.attr(fontsize='8',label=fig_title, dpi='200',layout=fig_layout)
		out_file_png = '%s/%s_graph' % (out_dir,out_file)
		G.render(out_file_png)
		os.remove('%s' % (out_file_png))
	def out_graph(
			self,
			data,
			data_vel,
			method='PCA',
			out_file='graph',
			title='',
			label=['',''],
	):
		n, d = data.shape
		n_node = self.num_node
		n_edge = int(
			0.5 * (np.sum(np.where(self.adjcy_mat > 0, 1, 0)) - n_node))
		data_node = np.empty([n_node, d], dtype=float)
		for i in range(n_node):
			data_node[i] = np.mean(data[self.node_id_set[i]], axis=0)

		if method == "PCA":
			data_node_pca = PCA(n_components=2).fit_transform(data_node)
		elif method == "TSNE":
			data_node_pca = TSNE(n_components=2).fit_transform(data_node)
		#
		data_edge_pca = np.empty([n_edge, 4], dtype=float)
		edge_flow = np.zeros([n_edge], dtype=float)
		i_edge = 0
		for i in range(n_node):
			for j in range(i + 1, n_node):
				if self.adjcy_mat[i, j] > 0:
					idx_i = self.node_id_set[i]
					idx_j = self.node_id_set[j]
					idx_ij = np.intersect1d(idx_i, idx_j)
					mean_i = np.mean(data[idx_i], axis=0)
					mean_j = np.mean(data[idx_j], axis=0)
					rltv_vel = mean_j - mean_i
					edge_vel = np.mean(data_vel[idx_ij], axis=0)
					vel = np.dot(rltv_vel, edge_vel)
					if vel > 0:
						data_edge_pca[i_edge, 0:2] = data_node_pca[i]
						data_edge_pca[i_edge, 2:4] = data_node_pca[j] - data_node_pca[i]
					else:
						data_edge_pca[i_edge, 0:2] = data_node_pca[j]
						data_edge_pca[i_edge, 2:4] = data_node_pca[i] - data_node_pca[j]
					mean_vel = np.mean(data_vel[idx_ij], axis=0)
					flow = np.dot(mean_vel, rltv_vel) / \
						np.linalg.norm(rltv_vel)
					edge_flow[i_edge] = flow
					i_edge += 1
		data_edge_pca = data_edge_pca[:i_edge]
		#
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1, 1, 1)
		ax.scatter(data_node_pca[:, 0], data_node_pca[:,
					1], zorder=1, s=5, color='gray')
		clim_min = np.percentile(edge_flow, 20)
		clim_max = np.percentile(edge_flow, 80)
		clim_delta = clim_max - clim_min
		ax.quiver(data_edge_pca[:, 0], data_edge_pca[:, 1], data_edge_pca[:, 2], data_edge_pca[:, 3], edge_flow, cmap='coolwarm', clim=(
			clim_min, clim_max), angles='xy', scale_units='xy', width=0.002, scale=1, alpha=0.8, zorder=0, headwidth=5, headlength=5, headaxislength=5)
		ax.set_title(title)
		ax.set_xlabel(label[0])
		ax.set_ylabel(label[1])
		plt.savefig(out_file)
	## ** Visualization with velocity by PCA
	def out_graph_pca_V(
		self,
		data,
		data_vel,
		out_dir='.',
		out_file='vmapper',
		fig_title='',
		fig_fmt='png'
	):
		file='%s/%s_pca.%s' % (out_dir, out_file, fig_fmt)
		self.out_graph(data,data_vel,"PCA",file,"V-Mapper (PCA coordinate)",["PC1","PC2"])

	## ** Visualization with velocity by tSNE
	def out_graph_tsne_V(
			self,
			data,
			data_vel,
			out_dir='.',
			out_file='vmapper',
			fig_title='',
			fig_fmt='png'
	):
		file='%s/%s_tsne.%s' % (out_dir, out_file, fig_fmt)
		self.out_graph(data,data_vel,"TSNE",file,"V-Mapper (tSNE coordinate)",["tSNE1","tSNE2"])
	def out_cytoscape_V(
		self,
		data,
		data_vel,
		sample_name,
		meta_data,
		meta_func,
		out_dir='.',
		out_file='vmapper',
		fig_title='',
		flow_type='mean'
	):
		n,d = data.shape
		n_node = self.num_node
		n_edge = int(0.5*(np.sum(np.where(self.adjcy_mat!=0,1,0))-n_node))
		## Helmholtz-Hodge decomposition
		grad_mat = np.zeros([n_edge,n_node],dtype=int)
		flow_mat = np.zeros([n_node,n_node],dtype=float)
		flow_vec = np.zeros([n_edge],dtype=float)
		idx_edge = np.zeros([n_edge,2],dtype=int)
		i_edge = 0
		if flow_type != 'mean' and flow_type != 'particle':
			print('Worning: There is no flow type \'%s\'' % (flow_type))
			flow_type = 'mean'
			print('         flow type is changed as \'%s\'' % (flow_type))
		if flow_type == 'mean':
			for i in range(n_node):
				for j in range(i+1,n_node):
					if self.adjcy_mat[i,j] > 0:
						idx_i = self.node_id_set[i]
						idx_j = self.node_id_set[j]
						idx_ij = np.intersect1d(idx_i,idx_j)
						mean_i = np.mean(data[idx_i],axis=0)
						mean_j = np.mean(data[idx_j],axis=0)
						rltv_vel = mean_j-mean_i
						mean_vel = np.mean(data_vel[idx_ij],axis=0)
						if np.linalg.norm(rltv_vel) > 1.0e-5:
							flow = np.dot(mean_vel,rltv_vel)/np.linalg.norm(rltv_vel)
						else:
							flow = 0
						grad_mat[i_edge,i] = -1
						grad_mat[i_edge,j] = 1
						flow_vec[i_edge] = flow
						flow_mat[i,j] = flow
						flow_mat[j,i] = -flow
						idx_edge[i_edge] = i,j
						i_edge += 1
		elif flow_type == 'particle':
			for i in range(n_node):
				for j in range(i+1,n_node):
					if self.adjcy_mat[i,j] > 0:
						idx_i = self.node_id_set[i]
						idx_j = self.node_id_set[j]
						idx_ij = np.intersect1d(idx_i,idx_j)
						mean_i = np.mean(data[idx_i],axis=0)
						mean_j = np.mean(data[idx_j],axis=0)
						rltv_vel = mean_j-mean_i
						v_e = np.dot(data_vel[idx_ij],rltv_vel)
						flow =np.sum(v_e/np.linalg.norm(data_vel[idx_ij],axis=1)/np.linalg.norm(rltv_vel))
						grad_mat[i_edge,i] = -1
						grad_mat[i_edge,j] = 1
						flow_vec[i_edge] = flow
						flow_mat[i,j] = flow
						flow_mat[j,i] = -flow
						idx_edge[i_edge] = i,j
						i_edge += 1
		potential = -np.dot(np.linalg.pinv(grad_mat),flow_vec)
		divergence = np.sum(flow_mat,axis=0)
		grad_flow = -np.dot(grad_mat,potential)
		div0_flow = flow_vec-grad_flow
		diffusion = -np.dot(grad_mat.T,grad_flow)
		# cycle score
		d_adjcy_mat = np.array(np.where(flow_mat>0,1,0),dtype=int)
		n_comp_cycle = n_node
		d_adjcy_mat_i = d_adjcy_mat
		d_adjcy_mat_csr = scipy.sparse.csr_matrix(d_adjcy_mat)
		cycle_score_flow = np.zeros(n_node,dtype=int)
		cycle_temp  = np.zeros(n_node,dtype=int)
		n_cycle_temp = n_node
		for i in range(n_comp_cycle):
				d_adjcy_mat_csr_i = scipy.sparse.csr_matrix(d_adjcy_mat_i)
				d_adjcy_mat_i = np.sign((d_adjcy_mat_csr*d_adjcy_mat_csr_i).toarray())
				idx_cycle = np.where(np.diag(d_adjcy_mat_i)==1)[0]
				d_adjcy_mat_i = np.where(cycle_temp==1,0,d_adjcy_mat_i)
				cycle_temp = np.sign(cycle_temp+d_adjcy_mat_i)
				if len(idx_cycle)>0:
					cycle_score_flow[idx_cycle] = i+2
				if np.sum(cycle_temp)==n_cycle_temp:
					break
				n_cycle_temp = np.sum(cycle_temp)
		#
		div0_flow_mat = np.zeros([n_node,n_node],dtype=float)
		for i in range(n_edge):
			div0_flow_mat[idx_edge[i,0],idx_edge[i,1]] = div0_flow[i]
			div0_flow_mat[idx_edge[i,1],idx_edge[i,0]] = -div0_flow[i]
		d_adjcy_mat = np.array(np.where(div0_flow_mat>0,1,0),dtype=int)
		n_comp_cycle = n_node
		d_adjcy_mat_i = d_adjcy_mat
		d_adjcy_mat_csr = scipy.sparse.csr_matrix(d_adjcy_mat)
		cycle_score_div0 = np.zeros(n_node,dtype=int)
		cycle_temp  = np.zeros(n_node,dtype=int)
		n_cycle_temp = n_node
		for i in range(n_comp_cycle):
				d_adjcy_mat_csr_i = scipy.sparse.csr_matrix(d_adjcy_mat_i)
				d_adjcy_mat_i = np.sign((d_adjcy_mat_csr*d_adjcy_mat_csr_i).toarray())
				idx_cycle = np.where(np.diag(d_adjcy_mat_i)==1)[0]
				d_adjcy_mat_i = np.where(cycle_temp==1,0,d_adjcy_mat_i)
				cycle_temp = np.sign(cycle_temp+d_adjcy_mat_i)
				if len(idx_cycle)>0:
					cycle_score_div0[idx_cycle] = i+2
				if np.sum(cycle_temp)==n_cycle_temp:
					break
				n_cycle_temp = np.sum(cycle_temp)
		#
		data_node = np.empty([n_node,d],dtype=float)
		for i in range(n_node):
			data_node[i] = np.mean(data[self.node_id_set[i]],axis=0)
		data_node_pca = PCA(n_components=2).fit_transform(data_node)
		data_node_tsne = TSNE(n_components=2).fit_transform(data_node)
		out_cyjs = open('%s/%s.cyjs' % (out_dir,out_file),"w")
		data_info = '\"data\":{\"name\":\"%s\"}' % (fig_title)
		node_info = '\"nodes\":['
		sample_name = np.array(sample_name)
		meta_data   = np.array(meta_data)
		num_meta = meta_func.shape[1]
		for i in range(self.num_node):
			member = '%s' % sample_name[self.node_id_set[i]][0]
			for j in range(len(sample_name[self.node_id_set[i]])-1):
				member += ', %s' % sample_name[self.node_id_set[i]][j+1] 
			idx_i = self.node_id_set[i]
			vel_mag = np.linalg.norm(np.mean(data_vel[idx_i],axis=0))
			node_info += '{\"data\" : {\"id\":\"%s\",\"member\":\"%s\",\"count\":%d,\"divergence\":%.5f,\"potential\":%.5f,\"diffusion\":%.5f,\"cycle_score(flow)\":%d,\"cycle_score(div0 flow)\":%d,\"PCA1\":%.5f,\"PCA2\":%.5f,\"tSNE1\":%.5f,\"tSNE2\":%.5f' % (self.node_name[i],member,len(self.node_id_set[i]),divergence[i],potential[i],diffusion[i],cycle_score_flow[i],cycle_score_div0[i],data_node_pca[i,0],data_node_pca[i,1],data_node_tsne[i,0],data_node_tsne[i,1])
			for k in range(num_meta):
				if meta_func[0,k] in meta_data[0]:
					meta_func_name = meta_func[0,k]
					if meta_func[1,k][-4:] == 'RECODE':
						meta_func_name += '-RECODE'
					meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
					node_info += ',\"%s[%s]\":%s' % (meta_func[0,k],meta_func[1,k],vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],self.node_id_set[i]))
			node_info += '}},'
		node_info = node_info[:-1] + ']'
		edge_info = '\"edges\":['
		i_edge = 0
		for i in range(n_node):
			for j in range(i+1,n_node):
				if self.adjcy_mat[i,j] > 0:
					idx_i = self.node_id_set[i]
					idx_j = self.node_id_set[j]
					edge_idx_set = np.intersect1d(idx_i,idx_j)
					mean_i = np.mean(data[idx_i],axis=0)
					mean_j = np.mean(data[idx_j],axis=0)
					rltv_vel = mean_j-mean_i
					rltv_vel_norm = np.linalg.norm(rltv_vel)
					mean_vel = np.mean(data_vel[edge_idx_set],axis=0)
					mean_vel_norm = np.linalg.norm(mean_vel)
					vel = np.dot(rltv_vel,mean_vel)
					flow_mag = np.abs(flow_mat[i,j])
					vel_prob = 0
					if rltv_vel_norm>1.0e-5 and mean_vel_norm>1.0e-5:
						vel_prob = np.abs(vel)/mean_vel_norm/rltv_vel_norm
					v_e = np.dot(data_vel[edge_idx_set],rltv_vel)
					ind_vel_prob =2*np.sum(np.abs(v_e[np.sign(v_e)==np.sign(vel)]))/np.sum(np.abs(v_e))-1
					weight = np.log(len(edge_idx_set)+1)
					flow_score = ind_vel_prob*vel_prob*weight
					pt_flow = 0
					if np.linalg.norm(rltv_vel)>0:
						for k in edge_idx_set:
							if np.linalg.norm(data_vel[k])>1.0e-5:
								pt_flow += np.dot(data_vel[k],rltv_vel)/np.linalg.norm(data_vel[k])
						pt_flow = np.abs(pt_flow/np.linalg.norm(rltv_vel))
					hetero_prob = 1-ind_vel_prob
					hetero_score = hetero_prob*weight
					s_idx = i
					t_idx = j
					if flow_mat[i,j]<0:
						s_idx = j
						t_idx = i
					if np.abs(grad_flow[i_edge])<1.0e-5:
						grad_flow_source = 'None'
						grad_flow_target = 'None'
						grad_flow_consist = 'false'
					elif flow_mat[i,j]*grad_flow[i_edge]>0:
						grad_flow_source = 'None'
						grad_flow_target = 'Arrow'
						grad_flow_consist = 'true'
						potential_source = potential[s_idx]
						potential_target = potential[t_idx]
					else:
						grad_flow_source = 'Arrow'
						grad_flow_target = 'None'
						grad_flow_consist = 'false'
						potential_source = potential[t_idx]
						potential_target = potential[s_idx]
					if np.abs(div0_flow[i_edge])<1.0e-5:
						div0_flow_source = 'None'
						div0_flow_target = 'None'
						div0_flow_consist = 'false'
					elif flow_mat[i,j]*div0_flow[i_edge]>0:
						div0_flow_source = 'None'
						div0_flow_target = 'Arrow'
						div0_flow_consist = 'true'
					else:
						div0_flow_source = 'Arrow'
						div0_flow_target = 'None'
						div0_flow_consist = 'false'
					member = '%s' % sample_name[edge_idx_set][0]
					for k in range(len(sample_name[edge_idx_set])-1):
						member += ', %s' % sample_name[edge_idx_set][k+1]
					edge_info += '{\"data\" : {\"source\":\"%s\",\"target\":\"%s\",\"name\":\"%s-%s\",\"member\":\"%s\","count":%d,\"annotation\":\"Arrow\",\"pt_flow\":%.5f,\"flow_mag\":%.5f,\"vel_prob\":%.5f,\"ind_vel_prob\":%.5f,\"hetero_prob\":%.5f,\"flow_score\":%.5f,\"hetero_score\":%.5f,\"grad_flow\":%.5f,\"grad_flow_source\":\"%s\",\"grad_flow_target\":\"%s\",\"grad_flow_arrow_consist\":%s,\"div0_flow\":%.5f,\"div0_flow_source\":\"%s\",\"div0_flow_target\":\"%s\",\"div0_flow_arrow_consist\":%s,\"potential_mean\":%.5f,\"potential_source\":%.5f,\"potential_target\":%.5f' % (self.node_name[s_idx],self.node_name[t_idx],self.node_name[s_idx],self.node_name[t_idx],member,self.adjcy_mat[i,j],pt_flow,flow_mag,vel_prob,ind_vel_prob,hetero_prob,flow_score,hetero_score,np.abs(grad_flow[i_edge]),grad_flow_source,grad_flow_target,grad_flow_consist,np.abs(div0_flow[i_edge]),div0_flow_source,div0_flow_target,div0_flow_consist,0.5*(potential[s_idx]+potential[t_idx]),potential[s_idx],potential[t_idx])
					i_edge += 1
					for k in range(num_meta):
						if meta_func[0,k] in meta_data[0]:
							meta_func_name = meta_func[0,k]
							if meta_func[1,k][-4:] == 'RECODE':
								meta_func_name += '-RECODE'
							meta_data_sec = meta_data[1:,np.where(meta_data[0]==meta_func_name)[0][0]]
							edge_info += ',\"%s[%s]\":%s' % (meta_func[0,k],meta_func[1,k],vmapper.meta_func.add_meta_info(meta_func[0,k],meta_data_sec,meta_func[1,k],edge_idx_set))
					edge_info += '}},'
		edge_info = edge_info[:-1] + ']'
		out_cyjs.write('{%s,\"elements\":{%s,%s}}' % (data_info,node_info,edge_info))