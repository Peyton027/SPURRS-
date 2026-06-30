from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist

INPUT_CSV = "centered_matrix.csv"
OUTPUT_PDF = "Normal_vs_Obese_individual_heatmap_fixed_sample_order.pdf"

COLOR_LIMIT = 2.0
HEATMAP_COLORS = ["#2166AC", "#F7F7F7", "#B2182B"]

CLUSTER_METRIC = "euclidean"
LINKAGE_METHOD = "ward"

NORMAL_SAMPLES = [
    "N1", "N3", "N4", "N5", "N6",
    "N7", "N8", "N9", "N10"
]

OBESE_SAMPLES = [
    "O1", "O2", "O3", "O5", "O6",
    "O7", "O8", "O9", "O10"
]

SAMPLE_ORDER = NORMAL_SAMPLES + OBESE_SAMPLES


def orient_linkage_by_score(tree, scores):
    tree = tree.copy()
    n_leaves = len(scores)

    node_scores = {
        i: (float(scores[i]), 1)
        for i in range(n_leaves)
    }

    for row_index in range(tree.shape[0]):
        node_id = n_leaves + row_index

        left_id = int(tree[row_index, 0])
        right_id = int(tree[row_index, 1])

        left_score, left_size = node_scores[left_id]
        right_score, right_size = node_scores[right_id]

        if left_score < right_score:
            tree[row_index, 0], tree[row_index, 1] = (
                tree[row_index, 1],
                tree[row_index, 0]
            )

            left_score, right_score = right_score, left_score
            left_size, right_size = right_size, left_size

        combined_score = (
            left_score * left_size + right_score * right_size
        ) / (left_size + right_size)

        node_scores[node_id] = (
            combined_score,
            left_size + right_size
        )

    return tree


def cluster_rows(matrix):
    if matrix.shape[0] < 2:
        return np.arange(matrix.shape[0]), None

    distances = pdist(matrix, metric=CLUSTER_METRIC)

    tree = linkage(
        distances,
        method=LINKAGE_METHOD
    )

    row_scores = matrix.mean(axis=1)

    tree = orient_linkage_by_score(
        tree,
        row_scores
    )

    return leaves_list(tree), tree


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {input_path}")

    df = pd.read_csv(input_path)

    for col in ["gene_symbol", "major_class"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    missing_samples = [
        sample for sample in SAMPLE_ORDER
        if sample not in df.columns
    ]

    if missing_samples:
        raise ValueError(
            "Missing sample columns: " + ", ".join(missing_samples)
        )

    matrix = df[SAMPLE_ORDER].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "All sample columns must contain finite numeric values."
        )

    row_labels = [
        f"{symbol}  [{str(category).replace(' channels', '').replace('; ', ' + ')}]"
        for symbol, category in zip(
            df["gene_symbol"],
            df["major_class"]
        )
    ]

    row_order, row_tree = cluster_rows(matrix)

    matrix = matrix[row_order, :]
    row_labels = np.array(row_labels)[row_order].tolist()

    n_genes = len(row_labels)
    n_samples = len(SAMPLE_ORDER)

    cohort_codes = np.array(
        [0] * len(NORMAL_SAMPLES) +
        [1] * len(OBESE_SAMPLES)
    )

    fig_height = max(20, 0.22 * n_genes + 6)

    fig = plt.figure(figsize=(18, fig_height))

    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[0.55, 0.12, 0.18, 14],
        width_ratios=[0.18, 0.28, 1.0, 0.035],
        hspace=0.06,
        wspace=0.025
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.975,
        bottom=0.05
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")

    title_ax.text(
        0.5,
        0.70,
        "Person-by-Person Ion-Channel / Transporter Expression Heatmap: Normal vs Obese",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold"
    )

    title_ax.text(
        0.5,
        0.20,
        "Genes hierarchically clustered by Normal + Obese expression and oriented from higher to lower mean centered expression; samples retained in original order",
        ha="center",
        va="center",
        fontsize=9
    )

    top_ax = fig.add_subplot(gs[1, 2])
    top_ax.axis("off")

    row_dend_ax = fig.add_subplot(gs[3, 0])

    if row_tree is not None:
        dendrogram(
            row_tree,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="black",
            ax=row_dend_ax
        )

        row_dend_ax.set_ylim(10 * n_genes, 0)

    row_dend_ax.set_xticks([])
    row_dend_ax.set_yticks([])

    for spine in row_dend_ax.spines.values():
        spine.set_visible(False)

    labels_ax = fig.add_subplot(gs[3, 1])

    labels_ax.set_xlim(0, 1)
    labels_ax.set_ylim(n_genes - 0.5, -0.5)
    labels_ax.axis("off")

    font_size = max(5.2, min(8.0, 190 / max(n_genes, 1)))

    for row_index, label in enumerate(row_labels):
        labels_ax.text(
            0.99,
            row_index,
            label,
            ha="right",
            va="center",
            fontsize=font_size
        )

    strip_ax = fig.add_subplot(gs[2, 2])

    strip_ax.imshow(
        cohort_codes.reshape(1, -1),
        aspect="auto",
        cmap=plt.get_cmap("tab10", 2),
        vmin=0,
        vmax=1
    )

    strip_ax.set_xlim(-0.5, n_samples - 0.5)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])

    strip_ax.axvline(
        len(NORMAL_SAMPLES) - 0.5,
        color="black",
        linewidth=1.0
    )

    strip_ax.text(
        (len(NORMAL_SAMPLES) - 1) / 2,
        1.55,
        "Normal",
        transform=strip_ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

    strip_ax.text(
        len(NORMAL_SAMPLES) + (len(OBESE_SAMPLES) - 1) / 2,
        1.55,
        "Obese",
        transform=strip_ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

    for spine in strip_ax.spines.values():
        spine.set_visible(False)

    heat_ax = fig.add_subplot(gs[3, 2])

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        HEATMAP_COLORS,
        N=256
    )

    norm = TwoSlopeNorm(
        vmin=-COLOR_LIMIT,
        vcenter=0,
        vmax=COLOR_LIMIT
    )

    image = heat_ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm
    )

    heat_ax.set_xticks(np.arange(n_samples))

    heat_ax.set_xticklabels(
        SAMPLE_ORDER,
        rotation=90,
        fontsize=8
    )

    heat_ax.tick_params(
        axis="x",
        length=0,
        pad=3
    )

    heat_ax.set_yticks([])
    heat_ax.set_xlim(-0.5, n_samples - 0.5)
    heat_ax.set_ylim(n_genes - 0.5, -0.5)

    heat_ax.axvline(
        len(NORMAL_SAMPLES) - 0.5,
        color="black",
        linewidth=1.0
    )

    cbar_ax = fig.add_subplot(gs[3, 3])

    cbar = fig.colorbar(
        image,
        cax=cbar_ax
    )

    cbar.set_label(
        "Centered expression\n(log2-CPM deviation from the Normal mean)",
        fontsize=9
    )

    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.61,
        0.012,
        "Blue = lower than the Normal-group mean; white = near the Normal-group mean; "
        "red = higher. "
        f"Color range: −{COLOR_LIMIT:g} to +{COLOR_LIMIT:g}; values beyond this range are color-saturated.",
        ha="center",
        va="bottom",
        fontsize=8
    )

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(
            fig,
            bbox_inches="tight",
            pad_inches=0.15
        )

    plt.close(fig)


if __name__ == "__main__":
    main()