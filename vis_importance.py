import json
import matplotlib.pyplot as plt
import argparse
import os
import re


def extract_layer_id(name):
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def classify_layer(name):
    if ".mlp." in name:
        return "mlp"
    elif ".attn." in name or ".self_attn." in name or ".attention." in name:
        return "attn"
    return "other"


def extract_proj_type(name):
    match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.", name)
    return match.group(1) if match else "other"


def sort_and_group(data):
    grouped = {"mlp": {}, "attn": {}}
    for k, v in data.items():
        cat = classify_layer(k)
        if cat not in grouped:
            continue
        proj = extract_proj_type(k)
        grouped[cat].setdefault(proj, []).append((k, v))
    for cat in grouped:
        for proj in grouped[cat]:
            grouped[cat][proj].sort(key=lambda x: (extract_layer_id(x[0]), x[0]))
    return grouped


def plot_raw(data, category, proj_type, base_name, output_dir):
    layer_names = [k for k, _ in data]
    raw = [v["raw_importance"] for _, v in data]
    x = range(len(layer_names))

    plt.figure(figsize=(14, 4))
    plt.bar(x, raw)
    plt.yscale("log")
    plt.xticks(x, layer_names, rotation=90, fontsize=6)
    plt.ylabel("Raw Importance (log)")
    plt.title(f"Raw Importances - {category.upper()} - {proj_type}")
    plt.tight_layout()

    fname = f"{base_name}_{category}_{proj_type}_raw"
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
    print(f"📁 Saved raw plot: {fname}.(png|pdf)")
    plt.close()


def plot_combined(data, category, proj_type, base_name, output_dir):
    layer_names = [k for k, _ in data]
    norm = [v["normalized_importance"] for _, v in data]
    rho = [v["compression_retention_rho"] for _, v in data]
    x = range(len(layer_names))

    plt.figure(figsize=(14, 4))
    plt.plot(x, norm, label="Normalized Importance")
    plt.plot(x, rho, label="Compression Retention (ρ)")
    plt.xticks(x, layer_names, rotation=90, fontsize=6)
    plt.ylabel("Value")
    plt.title(f"Combined Metrics - {category.upper()} - {proj_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = f"{base_name}_{category}_{proj_type}_combined"
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
    print(f"📁 Saved combined plot: {fname}.(png|pdf)")
    plt.close()


from collections import defaultdict


def plot_layerwise_aggregates(grouped, base_name, output_dir):
    for category in ["mlp", "attn"]:
        layer_data = defaultdict(list)
        for proj in grouped[category].values():
            for name, scores in proj:
                layer_idx = extract_layer_id(name)
                layer_data[layer_idx].append(scores)

        if not layer_data:
            continue

        sorted_layers = sorted(layer_data.keys())
        mean_raw = [sum(s["raw_importance"] for s in layer_data[i]) / len(layer_data[i]) for i in sorted_layers]
        mean_norm = [sum(s["normalized_importance"] for s in layer_data[i]) / len(layer_data[i]) for i in sorted_layers]
        mean_rho = [sum(s["compression_retention_rho"] for s in layer_data[i]) / len(layer_data[i]) for i in sorted_layers]

        # Plot raw importance
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_layers, mean_raw)
        plt.yscale("log")
        plt.xlabel("Layer Index")
        plt.ylabel("Raw Importance (log)")
        plt.title(f"Mean Raw Importance per Layer - {category.upper()}")
        plt.tight_layout()
        fname = f"{base_name}_{category}_layerwise_raw"
        plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        print(f"📊 Saved {category} layerwise raw plot: {fname}.(png|pdf)")
        plt.close()

        # Plot combined normalized + rho
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_layers, mean_norm, label="Normalized Importance")
        plt.plot(sorted_layers, mean_rho, label="Compression Retention (ρ)")
        plt.xlabel("Layer Index")
        plt.ylabel("Value")
        plt.title(f"Mean Metrics per Layer - {category.upper()}")
        plt.legend()
        plt.tight_layout()
        fname = f"{base_name}_{category}_layerwise_combined"
        plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        print(f"📊 Saved {category} layerwise combined plot: {fname}.(png|pdf)")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON file with importance scores")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.dirname(args.input) or "."

    grouped = sort_and_group(data)

    for category in ["mlp", "attn"]:
        for proj_type, entries in grouped[category].items():
            if not entries:
                continue
            plot_raw(entries, category, proj_type, base_name, output_dir)
            plot_combined(entries, category, proj_type, base_name, output_dir)
    plot_layerwise_aggregates(grouped, base_name, output_dir)


if __name__ == "__main__":
    main()
