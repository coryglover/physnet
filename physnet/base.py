import pandas as pd

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
        self.segemented_g = None

    def read_swc(self,file):
        """
        Reads SWC and converts into a pandas DataFrame.

        Parameters:
            file (str) - swc file path
        """
        self.df = pd.read_csv(file,delimiter=' ',names=['node_id','type','x','y','z','radius','parents'])
        pass

    def degree_segment(self,d=2):
        """
        Segements a graph by contracting nodes of degree 2 into the same link.
        Segements are labeled in df.
        Segemented_g stores the contracted graph.

        Parameters:
            d (int) - node degree to contract
        """
        return d
