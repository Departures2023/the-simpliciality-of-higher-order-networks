import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import xgi

hyperedges = [[1, 2, 3], [3, 4, 5, 6], [2, 4, 6], [2, 6]]
H = xgi.Hypergraph(hyperedges)

pos = xgi.barycenter_spring_layout(H, seed=1)

node_list = list(H.nodes)
S = "#418FDF"
I = "#DA291C"
node_colors = {1:S, 2:I, 3:I, 4:S, 5:S, 6:I} 

node_fc = [node_colors[n] for n in H.nodes]

edge_colors = [to_rgba(c) for c in ["#FF8C00", "#7D19FF", "#FF1290"]]
edge_colors = [(r, g, b, 0.6) for r, g, b, _ in edge_colors]

xgi.draw(H, pos=pos, edge_fc=edge_colors, node_fc=node_fc)

plt.show()