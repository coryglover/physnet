import pandas as pd
import networkx as nx
import copy
import numpy as np
from itertools import combinations
import math

class PhysNet():
    """
    This is the base class for the physnet package.
    It contains general functions and calls submodules for specific functions.

    Attributes:
        df (DataFrame) - SWC file information
        g (networkx Graph) - SWC graph
        segmented_g (networkx Graph) - segmented graph
    """
    def __init__(self):
        """
        Intialize attributes.
        """
        self.df = None
        self.g = None
        self.contracted_g = None

    def read_swc(self,file):
        """
        Reads SWC and converts into a pandas DataFrame.

        Parameters:
            file (str) - swc file path
        """
        self.df = pd.read_csv(file,delimiter=' ',names=['node_id','type','x','y','z','radius','parents'])
        pass

    def create_g(self):
        """
        Generates graphs from SWC file.
        """
        self.g = nx.Graph()
        for i in range(len(self.df)):
            self.g.add_edge(self.df.loc[i,'node_id'],self.df.loc[i,'parents'])
        self.g.remove_node(-1)
        pass

    def degree_segment(self,d=2):
        """
        Segements a skeletonization based on the degree of the nodes.
        It contracts all edges into one for all nodes of degree less than or equal to d.

        Parameters:
            d (int) - maximum degree to contract
        """
        # Intialize contracted network
        self.contracted_g = copy.deepcopy(self.g)

        # Contract nodes of degree d or less and greater than 1
        for u in self.g.nodes():
            if self.g.degree(u) <= d and self.g.degree(u) != 1:
                # Update contracted graph
                neighbors = list(self.contracted_g.neighbors(u))
                self.contracted_g.add_edge(neighbors[0],neighbors[1])
                self.contracted_g.remove_node(u)

        # Update df
        for j, e in enumerate(self.contracted_g.edges()):
            # Get path between nodes in edge
            path = nx.shortest_path(self.g,e[0],e[1])
            # Update segments
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                df_idx = np.array(self.df[['node_id','parents']])
                try:
                    edge = np.array([u,v])
                    idx = np.where(np.all(edge==df_idx,axis=1))[0][0]
                except:
                    edge = np.array([v,u])
                    idx = np.where(np.all(edge==df_idx,axis=1))[0][0]
                self.df.iloc[idx,1] = j
        pass

    def find_angle(self,e,degrees=False):
        """
        Finds angle of a given edge segment

        Parameters:
            e (edge)
            degrees (bool) - return angle in degrees

        Returns:
            theta (float) - angle
        """
        # Get vectors
        e0_coord = np.array(self.df[self.df['node_id']==e[0]][['x','y','z']])
        e1_coord = np.array(self.df[self.df['node_id']==e[1]][['x','y','z']])
        e_vec = e1_coord - e0_coord

        # Get length of vector
        e_vec = e_vec / np.linalg.norm(e_vec)
        unit = np.array([1,0,0])

        # Get angle
        theta = np.arccos(np.dot(e_vec,unit))
        if degrees:
            return math.degrees(theta)
        return theta

    def physicality_segment(self,theta_eps,r_eps,degrees=False):
        """
        Segements a skeletonization based on absolute difference of angle or absolute difference
        in radius width between adjacent skeleton links.

        Parameters:
            theta_eps (float) - difference tolerance for angles
            r_eps (float) - difference tolerance for radius
        """
        # Intialize contracted network
        self.contracted_g = copy.deepcopy(self.g)

        # Loop through nodes
        for u in self.g.nodes():
            # Get edges associated with each node
            edges = self.contracted_g.edges(u)
            # Compare each pair of edges
            edge_pairs = list(combinations(edges,2))
            for e, f in edge_pairs:
                print(e,f)
                # Get angles
                e_theta = self.find_angle(e,degrees)
                f_theta = self.find_angle(f,degrees)

                # Get absolute difference between angles
                if np.abs(e_theta - f_theta) < theta_eps:
                    # Check radius width
                    df_idx = np.array(self.df[['node_id','parents']])
                    try:
                        eidx = np.where(np.all(e==df_idx,axis=1))[0][0]
                    except:
                        eidx = np.where(np.all(e[::-1]==df_idx,axis=1))[0][0]
                    try:
                        fidx = np.where(np.all(f==df_idx,axis=1))[0][0]
                    except:
                        fidx = np.where(np.all(f[::-1]==df_idx,axis=1))[0][0]
                    # Check radius
                    e_r = self.df.iloc[eidx,5]
                    f_r = self.df.iloc[fidx,5]
                    if np.abs(e_r-f_r) < r_eps:
                        # Contract edge
                        self.contracted_g.add_edge(e[1],f[1])
                        print(f"new_edge {(e[1],f[1])}")
                        self.contracted_g.remove_edge(e[0],e[1])
                        self.contracted_g.remove_edge(f[0],f[1])

            # Remove node if no longer connected
            if self.contracted_g.degree(u) == 0:
                self.contracted_g.remove_node(u)

        pass
