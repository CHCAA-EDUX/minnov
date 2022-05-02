# imports
import pandas as pd 
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

# step 2.3 visualize network # 
def prepare_network(distance_matrix): 
    '''
    distance is the matrix (e.g. distance_threads or distance_topics)
    '''
    
    # distance between topics or threads
    dist = distance_matrix[distance_matrix != 0]
    dist = pd.DataFrame(dist, columns = ["dist"])

    # source and target 
    sources, targets = distance_matrix.nonzero()
    df_edgelist = pd.DataFrame(
        zip(sources.tolist(), targets.tolist()),
        columns = ["src", "trg"])
    
    # put it together
    df_concat = pd.concat([df_edgelist.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)
    df_concat = df_concat[df_concat["src"] < df_concat["trg"]]
    df_concat["weight"] = [1-x for x in df_concat["dist"]]
    
    # return 
    return df_concat


# attributes: nodes # 
def set_node_attributes(G): 
    
    # node size:
    node_size_list = list(dict(G.degree(weight = "weight")).values()) # degree: size of nodes (topics)

    # node color:
    node_color_list = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']

    # node labels
    node_label_dict = {
        0: 'bags & boxes',
        1: 'models & instructions',
        2: 'games & figures',
        3: 'calendar & day',
        4: 'app store'
    }
    
    return node_size_list, node_color_list, node_label_dict

# 

# assign 
def set_edge_attributes(G): 
    
    # edge width
    edgeattr_weight = nx.get_edge_attributes(G, "weight") # weight: extract weight (size of connections)
    edge_width_list = list(edgeattr_weight.values()) # weight in proper format (list)

    # edge color
    dct_edgecolor = {
        (0, 0): "tab:blue", #1f77b4
        (0, 1): "#268C70", # blue, green
        (0, 2): "#8F7B61", # blue, orange
        (0, 3): "#7B4F6E", # blue, red
        (0, 4): "#5A6FB9", # blue, purple
        (1, 1): "tab:green", #2ca02c
        (1, 2): "#96901D", # green, orange
        (1, 3): "#81642A", # green, red
        (1, 4): "#608475", # green, purple
        (2, 2): "tab:orange", #ff7f0e
        (2, 3): "#EB531B", # orange, red
        (2, 4): "#CA7366", # orange, purple
        (2, 5): "#C66B2D", # orange, brown
        (3, 3): "tab:red", #d62728
        (3, 4): "#B54773", # red, purple
        (4, 4): "tab:purple", #9467bd
    }
    
    # assign color 
    for i, j in G.edges():

        # sort the values
        comm_low, comm_high = sorted([i, j])

        # find the color 
        comm_col = dct_edgecolor.get((comm_low, comm_high))

        # assign color
        G[i][j]['color'] = comm_col

    ## extract it 
    edgeattr_color = nx.get_edge_attributes(G, "color")
    edge_color_list = list(edgeattr_color.values())
    
    return edge_width_list, edge_color_list

# draw network
def draw_network(G, node_size_list, node_color_list, edge_width_list, edge_color_list, node_label_dict): 
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.axis("off")

    pos = nx.spring_layout(
        G = G,
        seed = 125) # reproducibility

    ## draw nodes ##
    nx.draw_networkx_nodes(
            G, 
            pos,
            node_size = [x*10000 for x in node_size_list],
            node_color = node_color_list)

    ## draw edges ##
    nx.draw_networkx_edges(
        G, 
        pos,
        width = [x*100 for x in edge_width_list],
        edge_color = edge_color_list,
        alpha = 0.3) 

    ## draw labels ##
    nx.draw_networkx_labels(
        G, 
        pos, 
        labels = node_label_dict,
        font_size = 15);