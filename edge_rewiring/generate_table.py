from collections import defaultdict
import re

raw_data = """
dataset: congress-bills
round: 0
num_maximal_hyperedge: 31985
max_to_rewire: 21
success_update: 1
num_same_size: 0
total_time: 200.171
edges_searching_time: 2.436
rewiring_time: 0.071
num_missing_subface: 991
delta_SF: 0.000022
delta_ES: 0.000000
delta_FES: 0.000019
--------------------------------------------------
dataset: congress-bills
round: 1
num_maximal_hyperedge: 31985
max_to_rewire: 0
success_update: 0
num_same_size: 0
total_time: 92.472
edges_searching_time: 2.199
rewiring_time: 0.000
num_missing_subface: 10
delta_SF: 0.000000
delta_ES: 0.000000
delta_FES: 0.000000
--------------------------------------------------
dataset: congress-bills
round: 2
num_maximal_hyperedge: 31985
max_to_rewire: 2
success_update: 1
num_same_size: 0
total_time: 201.487
edges_searching_time: 2.321
rewiring_time: 0.068
num_missing_subface: 499
delta_SF: 0.000000
delta_ES: 0.000000
delta_FES: 0.000000
--------------------------------------------------
dataset: diseasome
round: 8
num_maximal_hyperedge: 69
max_to_rewire: 0
success_update: 0
num_same_size: 0
total_time: 0.237
edges_searching_time: 0.159
rewiring_time: 0.000
num_missing_subface: 56
delta_SF: 0.000000
delta_ES: 0.000000
delta_FES: 0.000000
--------------------------------------------------
dataset: hospital-lyon
round: 8
num_maximal_hyperedge: 60
max_to_rewire: 1
success_update: 1
num_same_size: 0
total_time: 0.625
edges_searching_time: 0.020
rewiring_time: 0.000
num_missing_subface: 1
delta_SF: 0.009535
delta_ES: 0.000000
delta_FES: 0.002105
--------------------------------------------------
dataset: email-enron
round: 8
num_maximal_hyperedge: 162
max_to_rewire: 19
success_update: 1
num_same_size: 0
total_time: 2.118
edges_searching_time: 0.033
rewiring_time: 0.001
num_missing_subface: 993
delta_SF: 0.001546
delta_ES: 0.000000
delta_FES: 0.000038
--------------------------------------------------
dataset: contact-high-school
round: 8
num_maximal_hyperedge: 210
max_to_rewire: 1
success_update: 1
num_same_size: 0
total_time: 2.556
edges_searching_time: 0.146
rewiring_time: 0.002
num_missing_subface: 1
delta_SF: 0.002059
delta_ES: 0.000000
delta_FES: 0.000729
--------------------------------------------------
dataset: disgenenet
round: 8
num_maximal_hyperedge: 463
max_to_rewire: 1
success_update: 1
num_same_size: 0
total_time: 3.823
edges_searching_time: 0.005
rewiring_time: 0.050
num_missing_subface: 24
delta_SF: 0.000000
delta_ES: 0.000000
delta_FES: 0.001150
--------------------------------------------------
dataset: contact-primary-school
round: 8
num_maximal_hyperedge: 345
max_to_rewire: 1
success_update: 0
num_same_size: 1
total_time: 2.948
edges_searching_time: 0.362
rewiring_time: 0.000
num_missing_subface: 1
delta_SF: 0.000000
delta_ES: 0.000000
delta_FES: 0.000000
--------------------------------------------------
dataset: congress-bills
round: 1
num_maximal_hyperedge: 31985
max_to_rewire: 5
success_update: 1
num_same_size: 0
total_time: 183.657
edges_searching_time: 2.232
rewiring_time: 0.065
num_missing_subface: 5
delta_SF: 0.000022
delta_ES: 0.000000
delta_FES: 0.000013
--------------------------------------------------
dataset: congress-bills
round: 2
num_maximal_hyperedge: 31985
max_to_rewire: 18
success_update: 1
num_same_size: 0
total_time: 178.857
edges_searching_time: 2.574
rewiring_time: 0.066
num_missing_subface: 228
delta_SF: 0.000043
delta_ES: 0.000000
delta_FES: 0.000033
--------------------------------------------------
"""

def parse_blocks(text):
    blocks = text.strip().split('--------------------------------------------------')
    records = []
    for block in blocks:
        record = {}
        for line in block.strip().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                record[key] = value
        if record:
            records.append(record)
    return records

def compute_averages(records, fields):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec['dataset']].append(rec)
    
    table_rows = []
    for dataset in sorted(grouped):
        data = grouped[dataset]
        row = [dataset]
        for field in fields:
            vals = [r.get(field, 0) for r in data]
            avg = sum(vals) / len(vals) if vals else 0
            row.append(f"{avg:.6f}" if isinstance(avg, float) else str(avg))
        table_rows.append(" & ".join(row) + r" \\")
    return table_rows

fields = [
    "total_time", "rewiring_time", "edges_searching_time",
    "delta_SF", "delta_ES", "delta_FES"
]

records = parse_blocks(raw_data)
rows = compute_averages(records, fields)

# Output LaTeX table
print("Dataset & " + " & ".join(f.replace('_', r'\_') for f in fields) + r" \\ \hline")
for row in rows:
    print(row)
