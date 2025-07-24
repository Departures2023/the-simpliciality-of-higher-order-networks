import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import xgi

hyperedges = [[1, 2, 3], [3, 4, 5, 6], [2, 4, 6]]
H = xgi.Hypergraph(hyperedges)

pos = xgi.barycenter_spring_layout(H, seed=1)

node_list = list(H.nodes)
S = "#418FDF"
I = "#DA291C"
node_colors = [S, I, S, S, S, S] 

edge_colors = ["rgba(229, 230, 228, 0.6)", 
               "rgba(186, 186, 186, 0.6)",  # green
               "rgba(158, 158, 158, 0.6)"]  # blue

edge_colors = [to_rgba(c) for c in ['#E5E6E4', '#BABABA', '#9E9E9E']]
edge_colors = [(r, g, b, 0.6) for r, g, b, _ in edge_colors]

xgi.draw(H, pos=pos, edge_fc=edge_colors, node_fc=node_colors)

plt.show()