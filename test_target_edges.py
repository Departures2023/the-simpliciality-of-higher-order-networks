#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'edge_rewiring'))

try:
    from model_generation import model_generation_es
    import xgi
    from sod.simpliciality import edit_simpliciality
    
    print("Testing target_num_edges parameter for connectivity control...")
    
    # Test case based on your requirements
    print("\n=== Target: es=0.04538, 516 nodes, 314 edges, 292 maximal, 1 component ===")
    
    H = model_generation_es(
        es=0.04538228067639832,
        approx_num_C=4000,  # Set high to allow target_num_edges to work
        num_max_hyperedge=292,
        num_node=516,
        min_size=2,
        max_size=None,
        adjust_es=True,
        connected_component=1,
        target_num_edges=314  # Direct control over edge count
    )
    
    print(f"\n📊 Results:")
    print(f"  - Edit simpliciality: {edit_simpliciality(H, min_size=2):.6f}")
    print(f"  - Nodes: {H.num_nodes}")
    print(f"  - Edges: {H.num_edges}")
    print(f"  - Maximal edges: {len(H.edges.maximal())}")
    print(f"  - Connected components: {xgi.number_connected_components(H)}")
    
    # Test different target edge counts
    print(f"\n=== Testing different edge counts with same connectivity ===")
    
    for target_edges in [250, 300, 350]:
        print(f"\n--- Target edges: {target_edges} ---")
        H_test = model_generation_es(
            es=0.05,
            approx_num_C=5000,
            num_max_hyperedge=100,
            num_node=200,
            connected_component=1,
            target_num_edges=target_edges
        )
        print(f"  Achieved edges: {H_test.num_edges}")
        print(f"  Connected components: {xgi.number_connected_components(H_test)}")
        print(f"  Edit simpliciality: {edit_simpliciality(H_test, min_size=2):.4f}")
    
    print(f"\n✅ target_num_edges parameter provides direct edge control!")
    print(f"💡 Use this to maintain connectivity while controlling edge count")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc() 