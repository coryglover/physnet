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

    def physicality_segment(self,theta_eps=1.5,r_eps=1,degrees=False,dual=False):
        """
        Segements a skeletonization based on absolute difference of angle or absolute difference
        in radius width between adjacent skeleton links as well as degree two nodes.

        Parameters:
            theta_eps (float) - difference tolerance for angles
            r_eps (float) - difference tolerance for radius
        """
        # Intialize contracted network
        self.contracted_g = copy.deepcopy(self.g)

        # Perform dfs
        dfs_edges = list(nx.dfs_edges(self.g))

        # Initialize variables
        segment = 0
        last_segment = 0
        visited_nodes = []
        visited_edges = {u: [] for u in self.g.nodes}
        df_idx = np.array(self.df[['node_id','parents']])

        # Traverse edges
        for i, e in enumerate(dfs_edges):
            # Find edge index
            try:
                eidx = np.where(np.all(e==df_idx,axis=1))[0][0]
            except:
                eidx = np.where(np.all(e[::-1]==df_idx,axis=1))[0][0]

            # Check that we are at a new node
            if e[0] in visited_nodes:
                # Get edges already visited
                edges = list(set(visited_edges[e[0]] + visited_edges[e[1]]))
                # Compare with each edge
                for k in range(len(edges)):
                    prev_eidx = edges[k][1]
                    # Get raidus of both edges
                    if np.abs(self.df.iloc[eidx,5] - self.df.iloc[prev_eidx,5]) > r_eps:
                        # Check if we need both conditions
                        if dual:
                            e_angle = self.find_angle(e,degrees)
                            f_angle = self.find_angle(dfs_edges[i-1],degrees)

                            # Compare angles
                            if degrees:
                                max_angle = 360
                            else:
                                max_angle = 2*np.pi

                            diff_angle = min(abs(e_angle-f_angle),max_angle-abs(e_angle-f_angle))
                            # Update segment
                            if diff_angle > theta_eps:
                                segment = self.df.iloc[prev_eidx,1]
                        else:
                            # Update segment
                            segment = self.df.iloc[prev_eidx,1]

                self.df.iloc[eidx,1] = segment
                prev_eidx = eidx
                continue

            # Update visited nodes
            visited_nodes.append(e[0])
            visited_edges[e[0]] += [(e,eidx)]
            visited_edges[e[1]] += [(e,eidx)]

            # Initialize first segment
            if i == 0:
                self.df.iloc[eidx,1] = segment
                prev_eidx = eidx

            # Update other edges
            else:
                # Get raidus of both edges
                if np.abs(self.df.iloc[eidx,5] - self.df.iloc[prev_eidx,5]) > r_eps:
                    # Check if we need both conditions
                    if dual:
                        e_angle = self.find_angle(e,degrees)
                        f_angle = self.find_angle(dfs_edges[i-1],degrees)

                        # Compare angles
                        if degrees:
                            max_angle = 360
                        else:
                            max_angle = 2*np.pi

                        diff_angle = min(abs(e_angle-f_angle),max_angle-abs(e_angle-f_angle))
                        # Update segment
                        if diff_angle > theta_eps:
                            last_segment += 1
                            segment = last_segment

                    else:
                        # Update segment
                        last_segment += 1
                        segment = last_segment
                # Check angle
                else:
                    e_angle = self.find_angle(e,degrees)
                    f_angle = self.find_angle(dfs_edges[i-1],degrees)

                    # Compare angles
                    if degrees:
                        max_angle = 360
                    else:
                        max_angle = np.pi

                    diff_angle = min(abs(e_angle-f_angle),max_angle-abs(e_angle-f_angle))
                    # Update segment
                    if diff_angle > theta_eps:
                        last_segment += 1
                        segment = last_segment

                # Update DataFrame
                self.df.iloc[eidx,1] = segment
                prev_eidx = eidx

        pass
