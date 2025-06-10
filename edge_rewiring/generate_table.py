import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
experiment_dir = os.path.join(base_dir, "experiment_result")
output_dir = os.path.join(base_dir, "exptable_result")

os.makedirs(output_dir, exist_ok=True)

# List of files to process
text_files = [
    "congress-bills.txt",
    "contact-high-school.txt",
    "contact-primary-school.txt",
    "diseasome.txt",
    "disgenenet.txt",
    "email-enron.txt",
    "email-eu.txt",
    "hospital-lyon.txt",
    "ndc-substances.txt",
    "tags-ask-ubuntu.txt"
]

#
def parse_blocks(text):
    blocks = text.strip().split('--------------------------------------------------')
    data = []
    for block in blocks:
        if not block.strip():
            continue
        entry = {}
        for line in block.strip().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                entry[key.strip()] = value.strip()
        data.append(entry)
    return data

def generate_latex_table(data):
    if not data:
        return "No data available."

    dataset_name = data[0].get("dataset", "unknown").replace('_', '-')
    rounds = [int(row["round"]) for row in data if "round" in row]

    latex = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Edge Rewiring Results for \\texttt{{{dataset_name}}} Dataset (Rounds {min(rounds)}--{max(rounds)})}}",
        "\\begin{tabular}{l" + "c" * len(data) + "}",
        "\\hline",
        "\\textbf{Metric} & " + " & ".join(f"\\textbf{{Round {row['round']}}}" for row in data) + " \\\\",
        "\\hline"
    ]

    # extract keys excluding 'dataset' and 'round'
    keys = [k for k in data[0] if k not in ("dataset", "round")]
    for key in keys:
        label = key.replace('_', ' ').capitalize()
        if "delta" in key.lower():
            label = f"$\\Delta_{{\\mathrm{{{key.split('_')[1].upper()}}}}}$"
        values = [row.get(key, "") for row in data]
        latex.append(f"{label} & " + " & ".join(values) + " \\\\")

    latex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(latex)

# Process each file in the provided list
for fname in text_files:
    input_path = os.path.join(experiment_dir, fname)

    try:
        with open(input_path, "r") as file:
            content = file.read()
        parsed_data = parse_blocks(content)
        latex_table = generate_latex_table(parsed_data)

        # Write .tex output
        output_filename = fname.replace(".txt", ".tex")
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w") as out_file:
            out_file.write(latex_table)

        print(f"LaTeX table written: {output_filename}")
    except FileNotFoundError:
        print(f"File not found: {input_path}")
    except Exception as e:
        print(f"Error processing {fname}: {e}")
