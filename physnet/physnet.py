### Physical Network Analysis Package
### Center for Complex Networks, Network Science Institute, Northeastern University

### Authors: Cory Glover, Benjamin Piazza, Csaba Both
### Contact: Cory Glover, glover.co@northeastern.edu

import pandas as pd
import networkx as nx
import copy
import numpy as np
from itertools import combinations
import math
import matplotlib.cm as cm
import matplotlib as matplotlib
from sympy import *
import os

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

    def read_swc(self,file,usecols=None):
        """
        Reads SWC and converts into a pandas DataFrame.

        Parameters:
            file (str) - swc file path
            usecols (list) - list of column names in swc file
        """
        self.df = pd.read_csv(file,delimiter=' ',names=['node_id','type','x','y','z','radius','parents'],usecols=usecols)
        pass

    def to_swc(self,file):
        """
        Converts dataframe to SWC
        """
        self.df.to_csv(file,sep=' ',header=False,index=False)
        pass

    def read_swc_bulk(self,foldername):
        """
        Reads all SWC in a folder and aggregates their data into one pandas DataFrame.

        Parameters:
            folder (str) - swc folder directory
        """
        base_directory = os.getcwd()
        os.chdir(base_directory + "\\" + foldername)
        files = listdir()
        filenames = []
        for i in range(len(files)):
            if len(files[i]) > 4:
                if files[i][-4:-1] == '.sw':
                    filenames.append(files[i][:-4])

        self.df = pd.read_csv(filenames[0]+".swc",delimiter=' ',names=['node_id','type','x','y','z','radius','parents'])
        for i in range(1,len(filenames)):
            df2 = pd.read_csv(filenames[i]+".swc",delimiter=' ',names=['node_id','type','x','y','z','radius','parents'])
            self.df = pd.concat([self.df,df2])
        pass

    def draw_networkx(self,cmap_name='Wistia',file=None):
        """
        Draw networkx object with edges colored by segments
        """
        # Get color values
        norm = matplotlib.colors.Normalize(vmin=np.min(self.df['type']),vmax=np.max(self.df['type']))
        cmap = cm.get_cmap(cmap_name)
        colors = norm(self.df['type'])

        # Get node positions
        pos = {self.df.iloc[i,0]: self.df.iloc[i,[2,3,4]] for i in range(len(self.df))}

        # Make plot
        nx.draw(self.g,pos=pos,edge_color=colors,node_size=.1)
        if file:
            plt.savefig(file)

        pass

    def create_g(self):
        """
        Generates graphs from SWC file.
        """
        self.g = nx.Graph()
        for i in range(len(self.df)):
            self.g.add_edge(self.df.loc[i,'node_id'],self.df.loc[i,'parents'],type=self.df.loc[i,'type'],weight=self.df.loc[i,'radius'])
        self.g.remove_node(-1)
        node_pos = {self.df.loc[i,'node_id']: [self.df.loc[i,'x'],self.df.loc[i,'y'],self.df.loc[i,'z']]}
        nx.set_node_attributes(self.g, node_pos, "pos")
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
            degrees (bool) - compare angles by degree
            dual (bool) - only change segment if both angle and radial condition are met
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

    def connected_edges_to_angles(self,connected_edges,radius=1):
        """
        Returns edge angles for all edge pairs in connected edges array.

        Parameters:
            connected_edges (list) - list of connected edges
            radius (int)

        Return:
            angle_list (list) - edge angles between pairs of edges in connected_edges
        """

        def asSpherical(xyz):
            x       = xyz[0]
            y       = xyz[1]
            z       = xyz[2]
            XsqPlusYsq = x**2 + y**2
            r = sqrt(XsqPlusYsq + z**2)
            theta = atan2(z,sqrt(XsqPlusYsq))
            phi = atan2(y,x)
            return [r,theta,phi]

        angle_list = []
        for i in range(len(connected_edges)):
            edge_l = connected_edges[i]
            sub_angles = []
            base_pos = self.g.nodes[edge_l[0][0]]['pos']
            paths = nx.single_source_shortest_path_length(self.g, edge_l[0][0], cutoff=radius)
            far_nodes = []
            a = list(paths.values())
            b = list(paths.keys())
            paths = [(b[i],a[i]) for i in range(len(a))]
            for sub_path in paths:
                if sub_path[1] == radius:
                    far_nodes.append(sub_path[0])
            for j in range(len(edge_l)):
                if len(edge_l) == len(far_nodes) and radius > 1:
                    for k in range(len(far_nodes)):
                        if nx.shortest_path_length(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k])) == radius-1:
                            pos2 = self.g.nodes[far_nodes[k]]['pos']
                else:
                    pos2 = self.g.nodes[edge_l[j][1]]['pos']
                xyz = [a_i - b_i for a_i, b_i in zip(pos2, base_pos)]
                spherical = asSpherical(xyz)
                sub_angles.append(list([spherical[1],spherical[2]]))
            angle_list.append(sub_angles)

        return angle_list

    def connected_edges_to_widths(self,connected_edges,radius=1):
        """
        Returns edge widths for all edge pairs in connected edges array.

        Parameters:
            connected_edges (list)
            radius (int)

        Returns:
            width_list (list) - list of edge width for pairs of connected edges
        """

        width_list = []
        for i in range(len(connected_edges)):
            edge_l = connected_edges[i]
            sub_widths = []
            base_pos = self.g.nodes[edge_l[0][0]]['pos']
            paths = nx.single_source_shortest_path_length(self.g, edge_l[0][0], cutoff=radius)
            far_nodes = []
            a = list(paths.values())
            b = list(paths.keys())
            paths = [(b[i],a[i]) for i in range(len(a))]
            for sub_path in paths:
                if sub_path[1] == radius:
                    far_nodes.append(sub_path[0])
            for j in range(len(edge_l)):
                if len(edge_l) == len(far_nodes) and radius > 1:
                    for k in range(len(far_nodes)):
                        if nx.shortest_path_length(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k])) == radius-1:
                            shortest_path = nx.shortest_path(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k]))
                            weight = self.g[shortest_path[-1]][shortest_path[-2]]['weight']
                            sub_widths.append(float(weight))
                else:
                    weight = self.g[edge_l[j][0]][edge_l[j][1]]['weight']
                    sub_widths.append(float(weight))
            width_list.append(sub_widths)

        return width_list

    def reject_outliers(self, data, m=1):
        """
        Filters out the outliers in a set of data based on the mean and the standard
        deviation.

        Parameters:
            data (ndarray)
            m (float) - multiple of standard deviations

        Return:
            filtered_data (ndarray) - data without outliers
        """
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def tangent(self,v1, v2):
        """
        Calculates the tangent vector as the unit vector pointing from
        vector v1 to vector v2.

        Parameters:
            v1 (ndarray)
            v2 (ndarray)

        Returns:
            t (ndarray) - tangent vector
        """
        if np.allclose(v2,v1):
            return 0
        return (v2 - v1) / np.sqrt(np.sum((v2-v1)**2))


    #calculate the curvature in edge center v2
    def curvature(self,e,prev=None,next=None):
        """
        Calculate curvature of a segment.

        Parameters:
            e (ndarray) - segment
            prev (ndarray) - previous segment
            next (ndarray) - next segment

        Returns:
            curvature (float)
        """
        df_idx = np.array(self.df[['node_id','parents']])

        # Get vectors
        e0_coord = np.array(self.df[self.df['node_id']==e[0]][['x','y','z']])
        e1_coord = np.array(self.df[self.df['node_id']==e[1]][['x','y','z']])
        e_vec = e1_coord - e0_coord

        if prev is None:
            # Find prev edges
            edge1 = np.where(df_idx[:,0]==e[0])[0][0]
            edge2 = np.where(df_idx[:,1]==e[0])[0][0]
            if np.allclose(df_idx[edge1],e):
                prev = df_idx[edge2][0]
            else:
                prev = df_idx[edge1][1]

            # Get coordinates
            prev_coord0 = np.array(self.df[self.df['node_id']==prev][['x','y','z']])
            prev_coord1 = np.array(self.df[self.df['node_id']==e[0]][['x','y','z']])
            prev_vec = prev_coord1 - prev_coord0

        else:
            prev_coord0 = np.array(self.df[self.df['node_id']==prev[0]][['x','y','z']])
            prev_coord1 = np.array(self.df[self.df['node_id']==prev[1]][['x','y','z']])
            prev_vec = prev_coord0 - prev_coord1

        if next is None:
            # Find next edge
            edge1 = np.where(df_idx[:,0]==e[1])[0][0]
            edge2 = np.where(df_idx[:,1]==e[1])[0][0]

            if np.allclose(df_idx[edge1],e):
                next = df_idx[edge2][0]
            else:
                next = df_idx[edge1][1]

            # Get coordinates
            next_coord0 = np.array(self.df[self.df['node_id']==next][['x','y','z']])
            next_coord1 = np.array(self.df[self.df['node_id']==e[1]][['x','y','z']])
            next_vec = next_coord1 - next_coord0

        else:
            next_coord0 = np.array(self.df[self.df['node_id']==next[0]][['x','y','z']])
            next_coord1 = np.array(self.df[self.df['node_id']==next[1]][['x','y','z']])
            next_vec = next_coord0 - next_coord1

        e_vec = e_vec[0]
        prev_vec = prev_vec[0]
        next_vec = next_vec[0]

        n = sqrt(sum((self.tangent(e_vec,next_vec)- self.tangent(e_vec,prev_vec))**2))

        d = sqrt(sum(((e_vec+next_vec)/2 - (prev_vec+e_vec)/2)**2))

        return n/d


    #the avarage curvature of a link
    def curvature_link(self,type):
        """
        Calculate the curvature of a segmented link.

        Parameters:
            type (int)

        Returns:
            total_cur (float)
        """
        # Get link
        type_edges = self.df[self.df['type']==type][['node_id','parents']]

        # Get ends of link
        end1 = list(set(type_edges['node_id']).difference(set(type_edges['parents'])))[0]
        end2 = list(set(type_edges['parents']).difference(set(type_edges['node_id'])))[0]
        type_edges = np.array(type_edges)
        end1_idx = np.where(type_edges[:,0]==end1)[0][0]
        end2_idx = np.where(type_edges[:,1]==end2)[0][0]

        # Remove ends of link
        l = np.array(type_edges)
        l = np.delete(l,end1_idx,0)
        l = np.delete(l,end2_idx,0)

        # Check that link is long enough
        if len(l) < 3:
            return -1

        # Get curvature of every segment
        s = np.zeros(len(l))
        for i in range(len(s)):
            s[i] = self.curvature(l[i])

        return np.sum(self.reject_outliers(s))

    def find_connected_edges(self,d=3):
        """
        Returns connected edges at nodes of degree d.

        Parameters:
            d (int) - degree

        Returns:
            connected_edges (list) - connected edges at nodes of degree d
        """

        degree_list = list([(node, val) for (node, val) in self.g.degree()])
        d_nodes = [degree_pair[0] for degree_pair in degree_list if degree_pair[1] == d]
        connected_edges = [list(self.g.edges(element)) for element in d_nodes]

        return connected_edges

    def connected_edges_to_angles(self,connected_edges,radius=1):
        """
        Returns edge angles for all edge pairs in connected edges array.

        Parameters:
            connected_edges (list) - list of connected edges
            radius (int)

        Return:
            angle_list (list) - edge angles between pairs of edges in connected_edges
        """

        def asSpherical(xyz):
            x       = xyz[0]
            y       = xyz[1]
            z       = xyz[2]
            XsqPlusYsq = x**2 + y**2
            r = sqrt(XsqPlusYsq + z**2)
            theta = math.atan2(z,sqrt(XsqPlusYsq))
            phi = math.atan2(y,x)
            return [r,theta,phi]

        angle_list = []
        for i in range(len(connected_edges)):
            edge_l = connected_edges[i]
            sub_angles = []
            base_pos = self.g.nodes[edge_l[0][0]]['pos']
            paths = nx.single_source_shortest_path_length(self.g, edge_l[0][0], cutoff=radius)
            far_nodes = []
            a = list(paths.values())
            b = list(paths.keys())
            paths = [(b[i],a[i]) for i in range(len(a))]
            for sub_path in paths:
                if sub_path[1] == radius:
                    far_nodes.append(sub_path[0])
            counter = 0
            for j in range(len(edge_l)):
                if len(edge_l) == len(far_nodes) and radius > 1:
                    for k in range(len(far_nodes)):
                        if nx.shortest_path_length(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k])) == radius-1:
                            pos2 = self.g.nodes[far_nodes[k]]['pos']
                elif edge_l[j][1] != -1:
                    pos2 = self.g.nodes[edge_l[j][1]]['pos']
                else:
                    counter = 1
                    break
                xyz = [a_i - b_i for a_i, b_i in zip(pos2, base_pos)]
                spherical = asSpherical(xyz)
                sub_angles.append(list([spherical[1],spherical[2]]))
            if counter == 0:
                angle_list.append(sub_angles)
            else:
                angle_list.append([])

        return angle_list

    def connected_edges_to_widths(self,connected_edges,radius=1):
        """
        Returns edge widths for all edge pairs in connected edges array.

        Parameters:
            connected_edges (list)
            radius (int)

        Returns:
            width_list (list) - list of edge width for pairs of connected edges
        """

        width_list = []
        for i in range(len(connected_edges)):
            edge_l = connected_edges[i]
            sub_widths = []
            base_pos = self.g.nodes[edge_l[0][0]]['pos']
            paths = nx.single_source_shortest_path_length(self.g, edge_l[0][0], cutoff=radius)
            far_nodes = []
            a = list(paths.values())
            b = list(paths.keys())
            paths = [(b[i],a[i]) for i in range(len(a))]
            for sub_path in paths:
                if sub_path[1] == radius:
                    far_nodes.append(sub_path[0])
            for j in range(len(edge_l)):
                if len(edge_l) == len(far_nodes) and radius > 1:
                    for k in range(len(far_nodes)):
                        if nx.shortest_path_length(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k])) == radius-1:
                            shortest_path = nx.shortest_path(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k]))
                            weight = self.g[shortest_path[-1]][shortest_path[-2]]['weight']
                            sub_widths.append(float(weight))
                else:
                    weight = self.g[edge_l[j][0]][edge_l[j][1]]['weight']
                    sub_widths.append(float(weight))
            width_list.append(sub_widths)

        return width_list

    def connected_edges_to_widths_average(self,connected_edges,radius=5):
        """
        Returns average edge widths for all edge pairs in connected edges array up to radius.

        Parameters:
            connected_edges (list)
            radius (int)

        Returns:
            width_list (list) - list of average edge width for pairs of connected edges
        """

        width_list = []
        for i in range(len(connected_edges)):
            edge_l = connected_edges[i]
            sub_widths = []
            base_pos = self.g.nodes[edge_l[0][0]]['pos']
            paths = nx.single_source_shortest_path_length(self.g, edge_l[0][0], cutoff=radius)
            far_nodes = []
            a = list(paths.values())
            b = list(paths.keys())
            paths = [(b[i],a[i]) for i in range(len(a))]
            for sub_path in paths:
                if sub_path[1] == radius:
                    far_nodes.append(sub_path[0])
            counter = 0
            for j in range(len(edge_l)):
                if len(edge_l) == len(far_nodes) and radius > 1:
                    for k in range(len(far_nodes)):
                        if nx.shortest_path_length(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k])) == radius-1:
                            shortest_path = nx.shortest_path(self.g, source=int(edge_l[j][1]), target=int(far_nodes[k]))
                            weight_array = []
                            for i in range(len(shortest_path)-1):
                                weight_array.append(float(self.g[shortest_path[i]][shortest_path[i+1]]['weight']))
                            weight = np.mean(weight_array)
                            sub_widths.append(float(weight))
                            counter += 1
                else:
                    weight = self.g[edge_l[j][0]][edge_l[j][1]]['weight']
                    sub_widths.append(float(weight))
            if counter == 3:
                width_list.append(sub_widths)
            else:
                width_list.append([])

        return width_list
