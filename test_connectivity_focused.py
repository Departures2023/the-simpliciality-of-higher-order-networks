#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'edge_rewiring'))

try:
    from model_generation import model_generation_es
    import xgi
    from sod.simpliciality import edit_simpliciality
    
    print("Testing connectivity-focused approach for your specific case...")
    
    # Your exact target parameters
    target_es = 0.04538228067639832
    target_nodes = 516
    target_edges = 314
    target_maximal = 292
    target_components = 1
    
    print(f"\n🎯 Target: ES={target_es:.6f}, {target_nodes} nodes, {target_edges} edges, {target_maximal} maximal, {target_components} component")
    
    # Strategy: Use higher approx_num_C to give more flexibility, 
    # rely on target_num_edges to control final edge count
    print("\n🧪 Attempt 1: Standard approach")
    H1 = model_generation_es(
        es=target_es,
        approx_num_C=1000,  # Lower to reduce initial complexity
        num_max_hyperedge=target_maximal,
        num_node=target_nodes,
        adjust_es=True,
        connected_component=target_components,
        target_num_edges=target_edges
    )
    
    print(f"\n📊 Results Attempt 1:")
    print(f"  - Edit simpliciality: {edit_simpliciality(H1, min_size=2):.6f} (target: {target_es:.6f})")
    print(f"  - Nodes: {H1.num_nodes} (target: {target_nodes})")
    print(f"  - Edges: {H1.num_edges} (target: {target_edges})")
    print(f"  - Maximal edges: {len(H1.edges.maximal())} (target: {target_maximal})")
    print(f"  - Connected components: {xgi.number_connected_components(H1)} (target: {target_components})")
    
    # Try with different parameters
    print(f"\n🧪 Attempt 2: Higher edge budget")
    H2 = model_generation_es(
        es=target_es,
        approx_num_C=2000,  # Give more edge budget
        num_max_hyperedge=target_maximal,
        num_node=target_nodes,
        adjust_es=False,  # Don't adjust ES, use raw formula
        connected_component=target_components,
        target_num_edges=target_edges + 50  # Slightly higher to allow connectivity
    )
    
    print(f"\n📊 Results Attempt 2:")
    print(f"  - Edit simpliciality: {edit_simpliciality(H2, min_size=2):.6f} (target: {target_es:.6f})")
    print(f"  - Nodes: {H2.num_nodes} (target: {target_nodes})")
    print(f"  - Edges: {H2.num_edges} (target: {target_edges})")
    print(f"  - Maximal edges: {len(H2.edges.maximal())} (target: {target_maximal})")
    print(f"  - Connected components: {xgi.number_connected_components(H2)} (target: {target_components})")
    
    # Analysis
    best_connectivity = min(xgi.number_connected_components(H1), xgi.number_connected_components(H2))
    print(f"\n🔍 Analysis:")
    print(f"  - Best connectivity achieved: {best_connectivity} components")
    print(f"  - The very low target ES ({target_es:.6f}) makes connectivity challenging")
    print(f"  - Consider accepting ES ~0.05-0.10 for better connectivity")
    
    # Suggest optimized parameters
    if best_connectivity > 1:
        print(f"\n💡 Suggestions for better connectivity:")
        print(f"  1. Increase target_num_edges to 350-400")
        print(f"  2. Accept slightly higher ES (~0.06-0.08)")
        print(f"  3. Reduce num_max_hyperedge to 200-250")
        print(f"  4. Use adjust_es=False for more direct control")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc() 