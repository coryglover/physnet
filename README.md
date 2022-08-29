# physnet
A package containing functions relating to the storing and analysis of physical network models.
This package works with physical network data stored as SWC files where the type of a link is assumed to be a label indicating which edge the subedge belongs to.
It's functionality includes predicting underlying graph structure from a given segmentation and analyzing said structure.
Specifically this analyzes focuses on branching analysis and curvature analysis.
We hope to expand to more analysis capabilities as they arrive.
Segmentation and analysis tools are primarily built on Pandas, Networkx, and Numpy.

## SWC Files
A SWC file contains a row for each link in the segmentation of a physical network.
Each row has 6 columns representing the links attributes:
-  `node_id`: ending node label,
- `type`: link label,
- `x`: ending node x-coordinate,
- `y`: ending node y-coordinate,
- `z`: ending node z-coordinate,
- `radius`: link radius,
- `parents`: starting node.
The link label indicates which edge in the underlying physical graph structure the segmented link belongs to.

`physnet` has the ability to read SWC into a Pandas DataFrame (`read_swc()`).
This initiates a `PhysNet` object with attribute `df` which contains the raw information of SWC file.
The function `to_swc` writes the attribute `df` to an SWC file.

## Updating Segments

