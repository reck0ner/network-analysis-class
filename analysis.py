import db
import config as c
import numpy as np
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv



def time_ranges(min_year,max_year,year_slices):
	for i in range(min_year, max_year, year_slices):
		time_ranges.append([i, i + year_slices-1])
	if time_ranges[-1][1] != max_year:
		time_ranges[-1][1] = max_year

def generate_network(author_mat,node_labels,threshold,weighted=False,**kwargs):
	time_ranges = time_ranges(1980,2015,5)
	print time_ranges
	g = nx.Graph()
	with open("author_collab.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        
	
	# 
	# conn,cursor = db.dbConnect()
	# sql = db.get_author_collab()
	# cursor.execute(sql)
	# rows = cursor.fetchall()


	# g.add_nodes_from(node_labels)
	# n = sim_mat.shape[0]
	# for (x,y) in  zip(np.triu_indices(n,k=1)[0],np.triu_indices(n,k=1)[1]):
	# 	if sim_mat[x][y]>= threshold:
	# 		# if g.has_edge(node_labels[x],node_labels[y]):
	# 		#print "has edge",node_labels[x],node_labels[y]
	# 		if weighted:
	# 			g.add_edge(node_labels[x],node_labels[y],weight=sim_mat[x][y])
	# 		else:
	# 			g.add_edge(node_labels[x],node_labels[y])
	# print "nodes:",g.number_of_nodes(),"edges",g.number_of_edges()
	# if kwargs.get("save",False):
	# 	to_pajek(g,kwargs.get("gname","network"))
	# return g

# def to_pajek(g,gname):
	
# 	fname = "net/pajek_"
# 	if weighted:
# 		fname=fname+"weighted_"
# 	else:
# 		fname=fname+"unweighted_"
# 	fname=fname+gname+".net"
# 	nx.write_pajek(g,fname)
	
# def plot_degree_dist(g):
# 	degree_sequence=sorted(nx.degree(g).values(),reverse=True) # degree sequence
# 	#print "Degree sequence", degree_sequence
# 	dmax=max(degree_sequence)

# 	plt.loglog(degree_sequence,'b-',marker='o')
# 	plt.title("Degree rank plot")
# 	plt.ylabel("degree")
# 	plt.xlabel("rank")

# 	# draw graph in inset
# 	plt.axes([0.45,0.45,0.45,0.45])
# 	gcc=sorted(nx.connected_component_subgraphs(g), key = len, reverse=True)[0]
# 	pos=nx.spring_layout(gcc)
# 	plt.axis('off')
# 	nx.draw_networkx_nodes(gcc,pos,node_size=20)
# 	nx.draw_networkx_edges(gcc,pos,alpha=0.4)

# 	plt.savefig("degree_histogram.png")
# def network_measures(g):
# 	node_labels = g.nodes()
# 	# trans = nx.transitivity(g)
# 	# print "transitivity",trans

# 	clus_coeff = nx.clustering(g)
# 	#print "avg clustering coefficients",np.mean(clus_coeff.values())

# 	'''
# 	The degree centrality for a node v is the fraction of nodes it is connected to.
# 	'''
# 	degree_cent = nx.degree_centrality(g)
# 	#print "avg degree_centrality",np.mean(degree_cent.values())
	
# 	'''
# 	Closeness centrality of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes. Higher values of closeness indicate higher centrality.
# 	'''
# 	closeness_cent = nx.closeness_centrality(g)
# 	#print "avg closeness_centrality",np.mean(degree_cent.values())
	
# 	'''
# 	Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
# 	'''
# 	betw_cent = nx.betweenness_centrality((g))
# 	#print "avg betweenness_centrality",np.mean(betw_cent.values())

# 	'''
# 	Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
# 	'''
# 	eigenv_cent = nx.eigenvector_centrality(g)
# 	#print "avg eigenvector_centrality",np.mean(eigenv_cent.values())
# 	'''
# 	PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links. It was originally designed as an algorithm to rank web pages.
# 	'''
# 	pgrank = nx.pagerank(g)
# 	'''
# 	f = open("net/network_measures_"+gname+".csv","w")
# 	f.write("topic,clustering_coefficient,degree_centrality,closeness_centrality,betweenness_centrality,eigenvector_centrality\n")
# 	# degree.intra
# 	# betweenness.intra
# 	# colseness.intra
# 	# eigenvector.intra
# 	# clustcoeff.intra
# 	# pagerank.intra
# 	'''
# 	'''
# 	for n in node_labels:
# 		f.write(str(n)+","+str(clus_coeff.get(n,"NaN"))+","+str(degree_cent.get(n,"NaN"))+","+str(closeness_cent.get(n,"NaN"))+","+str(betw_cent.get(n,"NaN"))+","+str(eigenv_cent.get(n,"NaN"))+"\n")
# 	'''
# 	#print node_labels
# 	df = pd.DataFrame({ 'domain':[n.split("_")[0] for n in node_labels] ,'label': [n.split("_")[1] for n in node_labels],'clustcoeff':[clus_coeff.get(n,"NaN") for n in node_labels],'degree':[degree_cent.get(n,"NaN") for n in node_labels] ,'betweenness':[betw_cent.get(n,"NaN") for n in node_labels] ,'eigenvector':[eigenv_cent.get(n,"NaN") for n in node_labels], 'colseness':[closeness_cent.get(n,"NaN") for n in node_labels], 'pagerank':[pgrank.get(n,"NaN") for n in node_labels] })
	
# 	return df

