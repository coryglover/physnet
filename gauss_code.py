import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
import copy
import sympy as sy

class LinearGaussCode:
    """
    This class connects a linear physical network with its Gauss code.

    Attributes:
        g (networkx) - network
        gauss_code (ndarray) - Gauss code of given network
        node_pos (ndarray) - position of nodes for plotting
        edges (ndarray) - edges of network
    """
    def __init__(self,g=None,gauss_code=None,node_pos=None,node_order=None,orient=True):
        # Initialize attributes
        self.g = g
        if self.g is not None:
            if not nx.is_directed(self.g):
                if orient:
                    self.g = self.orient_network()
        self.gauss_code = gauss_code
        if self.g is None and self.gauss_code is None:
            raise ValueError("NetworkX Graph or Gauss Code must be provided.")
        if self.g is None:
            self.g = nx.Graph()
            self.g.add_edges_from(list(self.gauss_code.keys()))
            self.g = self.orient_network()
        self.split_code = None
        self.nodes = list(self.g.nodes())
        self.node_pos = node_pos
        self.node_order = node_order
        self.edges = np.array(list(self.g.edges()))
        self.edge_idx = {tuple(e): i for i, e in enumerate(self.edges)}
        self.map = None
        self.R = None
    
    def generate_gauss_code(self,store_crossings=True):
        """
        This function aims to generate the Gauss code of a given network.
        It assumes that the network is projected to the xplane.
        """

        # Check for node order
        if self.node_order is None:
            self.generate_node_order()

        self.gauss_code = {tuple(self.edges[i]):[] for i in range(len(self.edges))}
        self.small_gauss_code = {tuple(self.edges[i]):[] for i in range(len(self.edges))}
        crossing_dictionary = {tuple(self.edges[i]):[] for i in range(len(self.edges))}
        if store_crossings:
            self.crossings = {}

        # Crossing counter
        k = 0
        # Loop through each pair of edges
        for i in range(len(self.edges)):
            # Compare with each other edge
            e = self.edges[i]
            for j in range(i+1,len(self.edges)):
                f = self.edges[j]

                # Check that the edges can intersect
                if e[0] == f[0] or e[0] == f[1] or e[1] == f[0] or e[1] == f[1]:
                    continue
                # Find intersection
                try:
                    # Find intersection
                    X = np.array([self.node_pos[self.node_order[e[0]]][:2],self.node_pos[self.node_order[e[1]]][:2]])
                    Y = np.array([self.node_pos[self.node_order[f[0]]][:2],self.node_pos[self.node_order[f[1]]][:2]])
                    lhs = np.array([X[1]-X[0],Y[0]-Y[1]]).T
                    rhs = np.array([Y[0]-X[0]]).T
                    t, s = np.linalg.solve(lhs,rhs)

                    # Check that intersection occurs in network
                    if t >=1 or t <= 0 or s >= 1 or s <= 0:
                        continue

                    # Save crossing and edges
                    # Check which edge is above
                    g = lambda t: (1-t)*self.node_pos[self.node_order[e[0]]]+t*self.node_pos[self.node_order[e[1]]]
                    h= lambda s: (1-s)*self.node_pos[self.node_order[f[0]]]+s*self.node_pos[self.node_order[f[1]]]

                    if g(t)[2] > h(s)[2]:
                        crossing_dictionary[tuple(e)].append((k,t,1,f))
                        crossing_dictionary[tuple(f)].append((k,s,-1,e))

                        # Store crossing
                        if store_crossings:
                            self.crossings[k] = [g(t),e,t]

                    else:
                        crossing_dictionary[tuple(e)].append((k,t,-1,f))
                        crossing_dictionary[tuple(f)].append((k,s,1,e))

                        # Store crossing
                        if store_crossings:
                            self.crossings[k] = [g(t),f,s]

                    k += 1
                # Skip parallel edges
                except:
                    continue

        # Create gauss code
        for e in self.edges:
            # Collect crossing information
            crossing_info = list(crossing_dictionary[tuple(e)])
            # Sort based on traversal
            if len(crossing_info) == 0:
                self.gauss_code[tuple(e)] = list(e)
                continue
            parameters = np.zeros(len(crossing_info))
            # Get parameters
            for i in range(len(crossing_info)):
                parameters[i] = crossing_info[i][1]
            mask = np.argsort(np.array(parameters))
            self.gauss_code[tuple(e)].append(e[0])
            for m in mask:
                self.gauss_code[tuple(e)].append((list(crossing_info[m])[0],list(crossing_info[m])[2],list(crossing_info[m])[3]))
            self.gauss_code[tuple(e)].append(e[1])

        pass

    def generate_network(self):
        """
        This function aims to generate network from a given Gauss code.
        It only creates the non-spatial network.
        """
        # Check that gauss code exists
        if self.gauss_code is None:
            self.generate_gauss_code()

        # Intialize network
        g = nx.Graph()

        # Add edges
        for k in list(self.gauss_code.keys()):
            g.add_edge(k[0],k[1])

        # Save network as attribute
        self.g = g
        pass

    def label_crossings(self):
        """
        Label each crossing as either a positive or negative crossing.
        These labels are saved in a label dictionary.
        """
        if self.map is None:
            self.generate_gauss_code_map()
        if self.node_pos is None:
            self.generate_embedding()
        # Initialize variables
        crossing_labels = dict()
        
        # Label each crossing
        for c in self.map.keys():
            # Get edges in crossing
            edges = self.map[c]
            # Get vectors
            v1 = self.node_pos[edges[0][1]] - self.node_pos[edges[0][0]]
            v2 = self.node_pos[edges[1][1]] - self.node_pos[edges[1][0]]
            # Make comparison vector
            u = np.array([v1[1],-v1[0]])
            # Get angle
            theta = np.dot(v2,u)
            # Create label
            if theta < 0:
                crossing_labels[c] = -1
            else:
                crossing_labels[c] = 1
        
        # Create attribute
        self.crossing_labels = crossing_labels
        pass
    
    def update_crossing_matrix(self):
        """
        This function updates a crossing dictionary with the original Gauss code.
        """
        # Check that gauss code exists
        if self.gauss_code is None:
            self.generate_gauss_code()

        # Intialize matrix
        self.R = np.zeros((len(self.edges),len(self.edges)))
        for i, e in enumerate(self.edges):
            # Get idx of edge
            e_idx = i
            code = self.gauss_code[tuple(e)]
            # Check gauss code
            for j in range(1,int(len(code)-1)):
                crossing = code[j]
                # Get index of edge
                f_idx = self.edge_idx[tuple(crossing[2])]
                # Update crossing
                self.R[e_idx,f_idx] = crossing[1]

        pass

    def generate_gauss_code_map(self):
        """
        This function creates a Gauss code map for quickly navigating the gauss code.
        The map is a dictionary where the keys are crossings and the value is a tuples of edges
        which are involved in that crossing.
        """
        # Check that gauss code exists
        if self.gauss_code is None:
            self.generate_gauss_code()

        # Intialize map
        self.map = {}
        # Loop through each pair of edges
        for i, e in enumerate(self.edges):
            if i != len(self.edges) - 1:
                for j, f in enumerate(self.edges[i+1:]):
                    # Get crossings involving both edges
                    e_crossings = np.array([[k[0],k[1]] for k in self.gauss_code[tuple(e)][1:-1]])
                    f_crossings = np.array([[k[0],k[1]] for k in self.gauss_code[tuple(f)][1:-1]])
                    if len(e_crossings) == 0 or len(f_crossings) == 0:
                        continue
                    # Get joint crossings
                    crossings = set(e_crossings[:,0]).intersection(set(f_crossings[:,0]))
                    # Update Gauss code map
                    for c in crossings:
                        # Find crossing in e_crossing
                        idx = np.where(e_crossings[:,0]==c)[0][0]
                        if e_crossings[idx,1] == -1:
                            self.map[c] = (e,f)
                        else:
                            self.map[c] = (f,e)

        pass

    def generate_split_code(self):
        """
        This function creates the split code of a given Gauss code
        """
        # Check that gauss code exists
        if self.gauss_code is None:
            self.generate_gauss_code()

        # Intialize split graph to check for connectivity
        # Check that graph exists
        if self.g is None:
            self.generate_network()

        # Intialize map
        if self.map is None:
            self.generate_gauss_code_map()

        split_graph = self.g.to_undirected().copy()
        split_graph = nx.MultiGraph(split_graph)

        # Intialize split code
        split_code = copy.deepcopy(self.gauss_code)

        # Pass through each crossing:
        crossings = list(self.map.keys())
        for c in crossings:
            # Get edges
            cur_edges = []

            for edge in self.edges:
                # Get split code
                split = split_code[tuple(edge)]
                for k in split[1:-1]:
                    if k[0] == c:
                        cur_edges.append(edge)
                # Check that we already have the edges
                if len(cur_edges) == 2:
                    break
            e, f = cur_edges

            # Check if edges are the same (RI)
            if np.allclose(e,f):
                # Get beta string
                for i, w in enumerate(split_code[tuple(e)][1:-1]):
                    if w[0] == c:
                        beta_0 = i+1
                        break
                for j, w in enumerate(split_code[tuple(e)][(beta_0+1):-1]):
                    if w[0] == c:
                        beta_1 = j+1
                        break
                beta = split_code[tuple(e)][(beta_0+1):beta_1]
                split_code[tuple(e)][(beta_0+1):beta_1] = beta[::-1]

            # If edges are different
            else:
                # Get true edges
                e0, e1 = (split_code[tuple(e)][0],split_code[tuple(e)][-1])
                f0, f1 = (split_code[tuple(f)][0],split_code[tuple(f)][-1])

                # Delete current edges
                split_graph.remove_edge(e0,e1)
                split_graph.remove_edge(f0,f1)

                # Rewire edges
                split_graph.add_edge(e0,f1)
                split_graph.add_edge(f0,e1)

                # Check for connectivity
                # Case 1 (w*=aPd and u* = gPb)
                if nx.is_connected(split_graph):
                    # Get delta and beta
                    for i, w in enumerate(split_code[tuple(e)][1:-1]):
                        if w[0] == c:
                            beta_0 = i+2
                            break

                    for i, w in enumerate(split_code[tuple(f)][1:-1]):
                        if w[0] == c:
                            gamma_0 = i+2

                    beta = split_code[tuple(e)][beta_0:]
                    gamma = split_code[tuple(f)][gamma_0:]

                    # Update split code
                    split_code[tuple(e)] = split_code[tuple(e)][:beta_0] + gamma
                    split_code[tuple(f)] = split_code[tuple(f)][:gamma_0] + beta
                # Case 2 (w* = aPg^{-1} and u* = d^{-1}Pb)
                else:

                    # Delete previous rewiring
                    split_graph.remove_edge(e0,f1)
                    split_graph.remove_edge(f0,e1)

                    # Rewire edges
                    split_graph.add_edge(e0,f0)
                    split_graph.add_edge(f1,e1)

                    # Get gamma and delta
                    for i, w in enumerate(split_code[tuple(e)][1:-1]):
                        if w[0] == c:
                            beta_0 = i+1
                            break

                    for i, w in enumerate(split_code[tuple(f)][1:-1]):
                        if w[0] == c:
                            gamma_0 = i+1

                    alpha = split_code[tuple(e)][:beta_0]
                    beta = split_code[tuple(e)][beta_0+1:]
                    gamma = split_code[tuple(f)][:gamma_0]
                    delta = split_code[tuple(f)][gamma_0+1:]

                    split_code[tuple(e)] = alpha + [split_code[tuple(e)][beta_0]] + gamma[::-1]
                    split_code[tuple(f)] = delta[::-1] + [split_code[tuple(f)][gamma_0]] + beta

        self.split_code = split_code

        pass

    def is_realizable(self):
        """
        This functions checks whether a given Gauss code is realizable.
        """
        # Check for Gauss code
        if self.gauss_code is None:
            raise ValueError("No Gauss code given.")

        # Check for split code
        if self.split_code is None:
            self.generate_split_code()

        # Create split graph
        split_edges = [[r[0],r[-1]] for r in list(self.split_code.values())]
        split_graph = nx.Graph()
        split_graph.add_edges_from(split_edges)

        # Check planarity of split graph
        return nx.algorithms.planarity.check_planarity(split_graph, counterexample=False)[0]

    def generate_embedding(self,plot=False,split_graph=False):
        """
        Generates an embedding by creating a node for each crossing and generating a layout with springs.
        The crossing nodes are then removed and the original nodes maintain their spring position.
        
        If split_graph:
            This function creates a two-dimensional embedding of a given Gauss code.
            It does this using the split code method described in 'Chord Diagrams and
            Gauss Code for Graphs' by Fleming and Mellor.

        Parameters:
            plot (bool): returns plot of network in 2d with no crossing embeddings
            split_graph (bool): determines whether to return split graph embedding
        """
        # Check that Gauss code exists
        if self.gauss_code is None:
            self.generate_gauss_code()
        
        # Perform embedding using spring-force
        if not split_graph:
            new_g = nx.Graph()
            n = len(self.g.nodes())
            for edge in self.gauss_code.keys():
                path = self.gauss_code[edge]
                prev = path[0]
                for p in path[1:]:
                    if type(p) is tuple:
                        crossing = int(p[0]+n)
                        new_g.add_edge(crossing,prev)
                        prev = crossing
                    else:
                        new_g.add_edge(prev,p)
                        prev = p
            # Get planar positions
            planar_pos = nx.planar_layout(new_g)
            node_pos = {}
            for i,u in enumerate(list(self.g.nodes())):
                node_pos[u] = planar_pos[u]
            self.node_pos = node_pos
            
            # Draw embedding
            if plot:
                nx.draw(self.g,pos=self.node_pos,with_labels=True)
                plt.show()
            
            
        # Perform split graph embedding
        if split_graph:
            # Check that split code exists
            if self.split_code is None:
                self.generate_split_code()

            # Get number of nodes
            n = len(self.g.nodes())

            # Create split graph positions
            split_edges = [[r[0],r[-1]] for r in list(self.split_code.values())]
            split_graph = nx.Graph()
            split_graph.add_edges_from(split_edges)
            split_node_pos = nx.planar_layout(split_graph)
            
            # Save embedding
            self.node_pos = split_node_pos
            # Draw embedding
            if plot:
                nx.draw(split_graph,pos=split_node_pos,with_labels=True)
                plt.show()

        pass

    def plot_3d(self,figsize=(13,13),labels=True,color=None,file=None):
        """
        This function plots the physical network in 3d.
        It rotates the network by theta at angle psi.

        Parameters:
            labels (bool) - label nodes
            color (dict) - give colors to certain edges
            file (str) - filename
        """

        # Check for node order
        if self.node_order is None:
            self.generate_node_order()

        # Intialize colors
        if color is None:
            initialize_colors = cm.rainbow(np.linspace(0,1,len(self.edges)))
            color = {tuple(e): c for e, c in zip(list(self.edges),list(initialize_colors))}

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')

        # Plot nodes
        ax.scatter3D(self.node_pos[:,0],self.node_pos[:,1],self.node_pos[:,2],s=100)

        # Label nodes
        if labels:
            nodes = list(self.node_order.keys())
            for n in nodes:
                ax.text(self.node_pos[self.node_order[n],0],
                       self.node_pos[self.node_order[n],1],
                       self.node_pos[self.node_order[n],2],
                       f"{n}",size=20)

        # Plot edges
        for e in self.edges:
            # Get node positions
            edge_pos = self.node_pos[[self.node_order[e[0]],self.node_order[e[1]]],:]
            ax.plot3D(edge_pos[:,0],edge_pos[:,1],edge_pos[:,2],color=color[tuple(e)])

        ax.set_axis_off()

        # Save plot
        if file:
            plt.savefig(file)

        plt.show()
        pass

    def plot_shadow(self,figsize=(13,13),labels=True,color=None,file=None):
        """
        This function plots the shadow of a given network onto the xplane.
        """

        # Check for node order
        if self.node_order is None:
            self.generate_node_order()

        # Intialize colors
        if color is None:
            initialize_colors = cm.rainbow(np.linspace(0,1,len(self.edges)))
            color = {tuple(e): c for e, c in zip(list(self.edges),list(initialize_colors))}

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()

        # Plot nodes
        ax.scatter(self.node_pos[:,0],self.node_pos[:,1],s=100)

        # Label nodes
        if labels:
            nodes = list(self.node_order.keys())
            for n in nodes:
                ax.text(self.node_pos[self.node_order[n],0],
                       self.node_pos[self.node_order[n],1],
                       f"{n}",size=20)

        # Plot edges
        for e in self.edges:
            # Get node positions
            edge_pos = self.node_pos[[self.node_order[e[0]],self.node_order[e[1]]],:2]
            ax.plot(edge_pos[:,0],edge_pos[:,1],color=color[tuple(e)])

        plt.tick_params(left=False,bottom=False)
        ax.axis('off')

        # Save plot
        if file:
            plt.savefig(file)

        plt.show()
        pass

    def plot_2d(self, figsize=(13,13), labels=True,color=None,file=None):
        """
        This function plots the network with crossings clear flatten to the x-axis.
        """
        # Check for node order
        if self.node_order is None:
            self.generate_node_order()

        # Intialize colors
        if color is None:
            initialize_colors = cm.rainbow(np.linspace(0,1,len(self.edges)))
            color = {tuple(e): c for e, c in zip(self.edges,initialize_colors)}

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()

        # Plot nodes
        ax.scatter(self.node_pos[:,0],self.node_pos[:,1],s=100)

        # Label nodes
        if labels:
            nodes = list(self.node_order.keys())
            for n in nodes:
                ax.text(self.node_pos[self.node_order[n],0],
                       self.node_pos[self.node_order[n],1],
                       f"{n}",size=20)

        # Plot edges
        for i, e in enumerate(self.edges):
            # Get node positions
            edge_pos = self.node_pos[[self.node_order[e[0]],self.node_order[e[1]]],:2]
            ax.plot(edge_pos[:,0],edge_pos[:,1],color=color[tuple(e)])

        # Draw crossing
        for k in self.crossings:
            # Erase crossing
            circle = plt.Circle(self.crossings[k][0], .05, color='w',zorder=i+1)
            ax.add_patch(circle)

            # Find function
            edge = self.crossings[k][1]
            f = lambda t: (1-t)*self.node_pos[self.node_order[edge[0]]]+t*self.node_pos[self.node_order[edge[1]]]
            # Redraw over edge
            over_crossing = np.array([f(self.crossings[k][2]-.03),f(self.crossings[k][2]+.03)])
            ax.plot(over_crossing[:,0],over_crossing[:,1],color=color[tuple(edge)],zorder=i+2)

        plt.tick_params(left=False,bottom=False)
        ax.axis('off')

        # Save plot
        if file:
            plt.savefig(file)

        else:
            plt.show()
        pass

    # <editor-fold>
    def plot_information(self, figsize=(13,13), xmin=-2.5,xmax=2.5,ymin=-2.5,
                         ymax=2.5,zmin=-1,zmax=1.4,labels=True,color=None,
                         file=None):
        """
        This function plots a 2d plot, 3d plot, the crossing matrix, and
        the relative Gauss code.
        """
        if color is None:
            initialize_colors = cm.rainbow(np.linspace(0,1,len(self.edges)))
            color = {tuple(e): c for e, c in zip(self.edges,initialize_colors)}


        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,2,1,projection='3d')


        # Plot nodes
        ax1.scatter3D(self.node_pos[:,0],self.node_pos[:,1],self.node_pos[:,2],s=100)

        # Label nodes
        if labels:
            nodes = list(self.node_order.keys())
            for n in nodes:
                ax1.text(self.node_pos[self.node_order[n],0],
                       self.node_pos[self.node_order[n],1],
                       self.node_pos[self.node_order[n],2],
                       f"{n}",size=20)

        # Plot edges
        for e in self.edges:
            # Get node positions
            edge_pos = self.node_pos[[self.node_order[e[0]],self.node_order[e[1]]],:]
            ax1.plot3D(edge_pos[:,0],edge_pos[:,1],edge_pos[:,2],color=color[tuple(e)])

        ax1.set_xlim(xmin,xmax)
        ax1.set_ylim(ymin,ymax)
        ax1.set_zlim(zmin,zmax)
#         ax1.set_axis_off()

        ax2 = fig.add_subplot(2,2,3)
        # Plot nodes
        ax2.scatter(self.node_pos[:,0],self.node_pos[:,1],s=100)

        # Label nodes
        if labels:
            nodes = list(self.node_order.keys())
            for n in nodes:
                ax2.text(self.node_pos[self.node_order[n],0],
                       self.node_pos[self.node_order[n],1],
                       f"{n}",size=20)

        # Plot edges
        for i, e in enumerate(self.edges):
            # Get node positions
            edge_pos = self.node_pos[[self.node_order[e[0]],self.node_order[e[1]]],:2]
            ax2.plot(edge_pos[:,0],edge_pos[:,1],color=color[tuple(e)])

        # Draw crossing
        for k in self.crossings:
            # Erase crossing
            circle = plt.Circle(self.crossings[k][0], .1, color='w',zorder=i+1)
            ax2.add_patch(circle)

            # Find function
            edge = self.crossings[k][1]
            f = lambda t: (1-t)*self.node_pos[self.node_order[edge[0]]]+t*self.node_pos[self.node_order[edge[1]]]
            # Redraw over edge
            over_crossing = np.array([f(self.crossings[k][2]-.03),f(self.crossings[k][2]+.03)])
            ax2.plot(over_crossing[:,0],over_crossing[:,1],color=color[tuple(edge)],zorder=i+2)

        plt.tick_params(left=False,bottom=False)
        ax2.axis('off')

        ax3 = fig.add_subplot(2,2,2)
        im = ax3.imshow(self.R,vmin=-1,vmax=1)
        xticks_loc = ax3.get_xticks().tolist()
        yticks_loc = ax3.get_yticks().tolist()
        ax3.set_xticks([i for i in range(len(self.edges))])
        ax3.set_yticks([i for i in range(len(self.edges))])
        ax3.set_xticklabels([str(e) for e in self.edges])
        ax3.set_yticklabels([str(e) for e in self.edges])
        fig.colorbar(im)

        ax4 = fig.add_subplot(2,2,4)
        text_kwargs = dict(ha='left', va='center', fontsize=16, color='k')
        for i,e in enumerate(self.edges):
            # Get gauss code
            gauss_code = self.gauss_code[tuple(e)]
            # Get small gauss code
            for j in range(1,len(gauss_code)-1):
                gauss_code[j] = gauss_code[j][:2]
            ax4.text(.1,1-float(f".{i}5"),s=f"{gauss_code}",**text_kwargs)
        ax4.axis('off')

        if file is None:
            plt.show()

        else:
            plt.savefig(file)
            plt.close()
    # </editor-fold>
    def generate_node_order(self):
        """
        Generate node order of network based on Gauss code or networkx
        """
        if self.g is None:
            if self.gauss_code is None:
                raise ValueError("Object requires network or gauss code")
            else:
                pass

        else:
            self.node_order = {n: i for i, n in enumerate(list(self.g.nodes()))}

    def rotate(self,theta,axis,radians = False,update_gauss=True):
        """
        Rotate network theta degrees around axis

        Parameters:
            theta (float) - angle
            axis (ndarray) - axis of rotation
        """
        # Convert to radians
        if radians is False:
            theta = np.radians(theta)

        # Get rotations
        rotation_vector = theta*axis
        rot = Rotation.from_rotvec(rotation_vector)

        # Rotate points
        for i,n in enumerate(self.node_pos):
            self.node_pos[i] = rot.apply(n)

        if update_gauss:
            self.generate_gauss_code()
            self.update_crossing_matrix()
            self.generate_gauss_code_map()
            self.generate_split_code()

        # Rotate crossings
        if self.crossings is not None and update_gauss is False:
            for i, k in enumerate(self.crossings):
                self.crossings[i] = rot.apply(k)

        pass
    
    def to_sc_directed(self, g, is_strong = False):
        """
        This function converts an undirected network to a strongly connected directed network.
        The network is assumed to have no bridges.

        Parameters:
            g (networkx) - undirected network

        Returns:
            balanced_g (networkx) - directed network
        """
        # Perform DFS traversal
        node_order = np.array(list(nx.dfs_preorder_nodes(g)))
        edges = list(nx.dfs_edges(g))
        not_tree_edges = []
        for e in g.edges():
            if e in edges or e[::-1] in edges:
                continue
            else:
                not_tree_edges.append(e)
            
        # Intialize directed graph
        di_g = nx.DiGraph()
        for e in edges:
            di_g.add_edge(e[0],e[1])
        for e in not_tree_edges:
            # Find indices
            idx_0 = np.where(node_order == e[0])[0][0]
            idx_1 = np.where(node_order == e[1])[0][0]
            if idx_0 < idx_1:
                di_g.add_edge(e[1],e[0])
            else:
                di_g.add_edge(e[0],e[1])
        if is_strong:
            return di_g, nx.is_strongly_connected(di_g)

        return di_g
    
    def orient_network(self,gauss=False):
        """
        This function takes an undirected network g
        and gives it an oriention such that each component
        connected by a bridge is strongly connected.

        Parameters:
            g (networkx) - undirected network

        Returns:
            di_g (networkx) - directed network
        """
        if gauss:
            # Reorient Gauss Code
            new_gauss = {}
            keys = list(self.gauss_code.keys())

            # Get new gauss code values
            for e in keys:
                if e in self.g.edges():
                    new_gauss[e] = self.gauss_code[e]
                else:
                    new_gauss[e[::-1]] = self.gauss_code[e][::-1]

            # Reverse edges within the paths
            new_keys = list(new_gauss.keys())
            for j, paths in enumerate(new_gauss.values()):
                for i, p in enumerate(paths):
                    if type(p) is tuple:
                        if tuple(p[2]) not in new_keys:
                            new_gauss[new_keys[j]][i] = (p[0],p[1],p[2][::-1])

            self.gauss_code = new_gauss
            pass
        else:
        
            # Find all bridges
            bridges = nx.bridges(self.g)

            # Create copy of network
            g_copy = copy.deepcopy(self.g)
            di_g = nx.DiGraph()
            di_g.add_nodes_from(self.g.nodes())

            # Remove bridges
            for e in bridges:
                g_copy.remove_edge(e[0],e[1])
                di_g.add_edge(e[0],e[1])

            # Create SCC of each component
            # Get components
            for c in nx.connected_components(g_copy):
                # Make component graph
                component_graph = nx.subgraph(g_copy, c)
                di_component = self.to_sc_directed(component_graph)

                # Add directed edges
                for e in di_component.edges():
                    di_g.add_edge(e[0],e[1])
            # Reorient Gauss Code
            new_gauss = {}
            keys = list(self.gauss_code.keys())

            # Get new gauss code values
            for e in keys:
                if e in di_g.edges():
                    new_gauss[e] = self.gauss_code[e]
                else:
                    new_gauss[e[::-1]] = self.gauss_code[e][::-1]

            # Reverse edges within the paths
            new_keys = list(new_gauss.keys())
            for j, paths in enumerate(new_gauss.values()):
                for i, p in enumerate(paths):
                    if type(p) is tuple:
                        if tuple(p[2]) not in new_keys:
                            new_gauss[new_keys[j]][i] = (p[0],p[1],p[2][::-1])

            self.gauss_code = new_gauss


            return di_g
    
    def imbalance_correction(self, g, tol = 10e-8, max_iter = int(10e4)):
        """
        This function takes a strongly connected directed network and
        gives each of the edges a weight such that the sum of the
        in-degree and out-degree of a node is 0.

        This is based on the paper ...
        """
        # Intialize edge weights
        nx.set_edge_attributes(g, values = 1, name = 'weight')

        # Update network
        for _ in range(max_iter):
            nodes_to_correct = []
            neg_nodes_to_correct = []
            # Check node total degree
            for v in list(g.nodes()):
                # Get edge weights
                in_neighbors = list(g.predecessors(v))
                out_neighbors = list(g.successors(v))
                # Compare in and out degree
                v_in = sum([g[n][v]['weight'] for n in in_neighbors])
                v_out = sum([g[v][n]['weight'] for n in out_neighbors])
                if v_in > v_out:
                    nodes_to_correct.append((v,v_in-v_out))
                if v_out > v_in:
                    neg_nodes_to_correct.append((v,v_out-v_in))
            if nodes_to_correct == []:
                return self.g
            # Choose random pos and neg pari
            pos_node = nodes_to_correct[np.random.choice(np.arange(len(nodes_to_correct)))]
            neg_node = neg_nodes_to_correct[np.random.choice(np.arange(len(neg_nodes_to_correct)))]

            # Find path
            path = nx.shortest_path(g,source=pos_node[0],target=neg_node[0])

            # Get path edges
            for i in range(len(path)-1):
                g[path[i]][path[i+1]]['weight'] += pos_node[1]

        raise ValueError("Does not converge")
    
    def balance_network(self, tol = 10e-8, max_iter = int(10e4)):
        """
        This function takes a strongly connected directed network and
        gives each of the edges a weight such that the sum of the in-
        and out-edge weights at each node is 0.

        This algorithm is taken from A. Makhdoumi and A. Ozdaglar, 
        "Graph balancing for distributed subgradient methods over directed graphs," 
        2015 54th IEEE Conference on Decision and Control (CDC), 2015, pp. 1364-1371, 
        doi: 10.1109/CDC.2015.7402401.

        Parameters:
            g (networkx) - graph

        Return:
            g (networkx) - graph with balanced weights
        """
        
        # Intialize edge weights
        nx.set_edge_attributes(self.g, values = 0, name = 'weight')

        # Get strongly connected components
        sc_components = nx.strongly_connected_components(self.g)
        for c in sc_components:
            if len(c) == 0:
                continue
            # Weight edges in component
            self.imbalance_correction(nx.subgraph(self.g,c))

        return None
    
    def alexander_matrix(self,sym=True):
        """
        This function defines the Alexander matrix of a network from its Gauss code.
        It assumes that the network has been balanced.

        Parameters:
            sym (bool) - computes alexander matrix symbolically. if False, computes Alexander matrix with t=-1.
        """
        # If no map update map
        if self.map is None:
            self.generate_gauss_code_map()
        # If no crossing labels
        if self.crossing_labels is None:
            self.label_crossings()

        # Get number of crossings
        c = len(self.map.keys())
        n = len(self.nodes)
        m = len(self.edges)

        # Compute matrix symbolically
        # Intialize variable
        t = sy.symbols('t')

        # Intialize matrix
        alex = sy.zeros(c+n,c+m)

        # Loop through all crossings
        cur_arc = 0
        edge_to_arc = {e:np.array([0,0]) for e in self.gauss_code.keys()}
        crossing_dict = {k:i for i,k in enumerate(self.map.keys())}

        # Add crossings to matrix
        for e in self.gauss_code.keys():
            path = self.gauss_code[e]

            # Initialize arc
            arc = []

            # Loop through path of edge
            for i, p in enumerate(path):
                # Check if we are at a crossing
                if type(p) is tuple:
                    # Check for under crossing
                    if p[1] == -1:
                        # Add crossing
                        arc.append(p)
                        # Pass through arc and update crossings
                        for j, crossing in enumerate(arc):
                            # Check that we are at a vertex
                            if type(crossing) is int:
                                if crossing is e[0]:
                                    edge_to_arc[e][0] = cur_arc
                                else:
                                    edge_to_arc[e][1] = cur_arc
                                continue
                            # Check for over crossings within arc
                            elif crossing[1] == 1:
                                # Add crossing info
                                alex[crossing_dict[crossing[0]],cur_arc] = 1-t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                            # Check for first undercrossing
                            elif crossing[1] == -1 and j == 0:
                                if self.crossing_labels[crossing[0]] == 1:
                                    alex[crossing_dict[crossing[0]],cur_arc] = -1
                                else:
                                    alex[crossing_dict[crossing[0]],cur_arc] = t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                            # Check for last undercrosssing
                            else:
                                if self.crossing_labels[crossing[0]] == 1:
                                    alex[crossing_dict[crossing[0]],cur_arc] = t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                                else:
                                    alex[crossing_dict[crossing[0]],cur_arc] = -1
                        # Initialize next crossing
                        arc = [p]
                        cur_arc += 1
                    # Add over crossings
                    else:
                        arc.append(p)
                # Check for beginning of arc
                elif i == 0:
                    arc.append(p)
                # Check for end of arc at vertex
                else:
                    arc.append(p)
                    # Pass through arc and update crossings
                    for j, crossing in enumerate(arc):
                        # Check for nodes
                        if type(crossing) is int:
                            if crossing is e[0]:
                                edge_to_arc[e][0] = cur_arc
                            else:
                                edge_to_arc[e][1] = cur_arc
                            continue
                        # Check for overcrossings within arc
                        elif crossing[1] == 1:
                            # Add crossing info
                            alex[crossing_dict[crossing[0]],cur_arc] = 1-t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                        # Check for first undercrossing
                        elif crossing[1] == -1 and j == 0:
                            if self.crossing_labels[crossing[0]] == 1:
                                alex[crossing_dict[crossing[0]],cur_arc] = -1
                            else:
                                alex[crossing_dict[crossing[0]],cur_arc] = t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                        # Check for last undercrosssing
                        else:
                            if self.crossing_labels[crossing[0]] == 1:
                                alex[crossing_dict[crossing[0]],cur_arc] = t**(self.g[crossing[2][0]][crossing[2][1]]['weight'])
                            else:
                                alex[crossing_dict[crossing[0]],cur_arc] = -1
                    # Initialize next crossing
                    arc = [p]
                    cur_arc += 1
        cur_vertex = c
        # Add vertices
        for v in self.nodes:
            # Get in and out edges
            in_edges = self.g.in_edges(v)
            out_edges = self.g.out_edges(v)
            # Calculate m for each edges
            edges = list(in_edges) + list(out_edges)
            weights = np.array([self.g[e[0]][e[1]]['weight'] for e in edges])
            weights[len(in_edges):] *= -1
            m = np.zeros(len(edges),dtype=int)
            for i in range(len(edges)):
                m[i] = sum(weights[:i]) + max([weights[i],0])
                for j in range(i):
                    # Get edges
                    cur_edge = edges[j]
                    idx =  edge_to_arc[cur_edge]
                    if j >= len(in_edges):
                        alex[cur_vertex,idx[0]] = -t**(m[j])
                    else:
                        alex[cur_vertex,idx[1]] = t**(m[j])
                idx = edge_to_arc[edges[i]]
                if i >= len(in_edges):
                    alex[cur_vertex,idx[0]] = -t**(m[i])
                else:
                    alex[cur_vertex,idx[1]] = t**(m[i])
            cur_vertex += 1
        if sym is False:
            return np.array(alex.subs({'t':-1}))
        
        self.alex_mat = alex
        return alex
    
    def alexander_polynomial(self):
        """
        Compute the alexander polynomial of a given Gauss code.
        """
        # Get size of cofactor matrices
        m, n = self.alex_mat.shape
        r = np.min([m,n]) - 1
        
        # Get cofactors
        cofactors = []
        for i in range(int(m-r+1)):
            for j in range(int(n-r+1)):
                cofactors.append(self.alex_mat[i:i+r,j:j+r])
        # Calculate determinants
        gcd = np.nan
        for i, co in enumerate(cofactors):
            if i == 0:
                prev = sy.det(co)
            if i == 1:
                cur = sy.det(co)
                gcd = sy.gcd(prev,cur)
            else:
                cur = sy.det(co)
                gcd = sy.gcd(gcd, cur)
                prev = cur
        
        # Get gcd
        return sy.simplify(gcd)
    
    def determinant(self):
        """
        Compute determinant of a given Gauss code.
        """
        # Get size of cofactor matrices
        m, n = self.alex_mat.shape
        r = np.min([m,n]) - 1
        
        # Get cofactors
        cofactors = []
        for i in range(int(m-r+1)):
            for j in range(int(n-r+1)):
                cofactors.append(self.alex_mat[i:i+r,j:j+r].subs({'t':-1}))
                
        # Calculate determinants
        dets = []
        for co in cofactors:
            dets.append(sy.det(co))
        
        # Get gcd
        return sy.gcd(dets)