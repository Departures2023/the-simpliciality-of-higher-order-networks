#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'edge_rewiring'))

try:
    from edge_rewiring.model_generation import connected_num,  model_generation_es
    import xgi
    from sod.simpliciality import edit_simpliciality
    
    print("Testing custom connectivity functions...")
    
    # Create a simple test hypergraph
    H_test = xgi.Hypergraph()
    H_test.add_edge([1, 2, 3])      # Edge 1
    H_test.add_edge([3, 4, 5])      # Edge 2 (shares node 3 with Edge 1)
    H_test.add_edge([6, 7, 8])      # Edge 3 (isolated)
    H_test.add_edge([4, 9, 10])     # Edge 4 (shares node 4 with Edge 2)
    
    print(f"\nTest hypergraph:")
    print(f"  Edges: {H_test.edges.members()}")
    print(f"  XGI connected components: {xgi.number_connected_components(H_test)}")
    print(f"  connected_num (edge pairs sharing nodes): {connected_num(H_test)}")
    
    # Test with model generation
    print(f"\n" + "="*50)
    print("Testing with model generation...")
    
    H_model = xgi.load_xgi_data("diseasome")
    H_model.cleanup(singletons=True, multiedges=True, connected=False)
    
    print(f"\nGenerated hypergraph:")
    print(f"  Nodes: {H_model.num_nodes}")
    print(f"  Edges: {H_model.num_edges}")
    print(f"  XGI connected components: {xgi.number_connected_components(H_model)}")
    print(f"  connected_num (edge pairs sharing nodes): {connected_num(H_model)}")
    print(f"  Edit simpliciality: {edit_simpliciality(H_model, min_size=2):.4f}")
    
    # Test which function correlates better with connectivity goal
    print(f"\n💡 Analysis:")
    print(f"  - XGI connected components: measures true graph connectivity")
    print(f"  - connected_num: counts hyperedge pair connections")
    print(f"  - connected_nodes_count: counts bridging nodes")
    print(f"  - Higher connected_num/connected_nodes_count usually means better connectivity")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc() 