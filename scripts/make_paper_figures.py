"""Draw the paper's figures and arc-level statistics from the frontier files.

Run from the repository root; requires matplotlib but no solver::

    python scripts/make_paper_figures.py OUTPUT_DIR

Every figure is written as a PDF. Every arc-level panel plots one point per
connection, because the two directions of a connection carry equal capacity.
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from tdindp_capacities import DATA_DIR, load_benchmark, load_frontier, load_networks
from tdindp_capacities.analysis import undirected_connections

TAG = "2026-10-31"
LAYERS = ("Gas", "Power", "Water")
FULL_WIDTH = 6.69
#: Connections whose capacity moved by less than this are treated as unchanged.
UNCHANGED_TOLERANCE = 1e-3

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
DIVERGING_LOW = "#d03b3b"
DIVERGING_MID = "#f0efec"
SEQUENTIAL = ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def layer_values(capacities, connections):
    """``{layer: array}`` for one capacity vector, in connection order."""
    return {
        k: np.array([capacities[(i, j, k, 0)] for i, j in connections[k]])
        for k in LAYERS
    }


def plain_number(value, _position=None):
    return f"{value:,.0f}"


def compact_thousands(value, _position=None):
    return f"{value / 1000:.0f}k"


def log_axes(ax, low, high):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ticks = [t for t in (1, 3, 10, 30, 100, 300) if low <= t <= high]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(plain_number))
        axis.set_minor_formatter(NullFormatter())
    ax.plot([low, high], [low, high], color=MUTED, linewidth=0.7, zorder=1)


def split_change(before, after):
    """Masks of connections raised, lowered, and unchanged going before -> after."""
    change = after - before
    raised = change > UNCHANGED_TOLERANCE
    lowered = change < -UNCHANGED_TOLERANCE
    return raised, lowered, ~(raised | lowered)


def corner_note(ax, text):
    ax.text(
        0.97, 0.05, text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=SECONDARY, linespacing=1.25,
    )


def scatter(ax, x, y, color, marker="o", size=13, zorder=3):
    ax.scatter(
        x, y, s=size, color=color, marker=marker, edgecolors="white",
        linewidths=0.35, zorder=zorder, clip_on=False,
    )


def frontier_overview(figures, out):
    frontier = figures["fixed"]
    summary = frontier.summary_table().sort_values("sum_u")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.75), constrained_layout=True)
    ax.plot(summary["sum_u"], summary["sum_log_u"], color=BLUE, linewidth=1.2, zorder=3)
    sweep = summary[~summary["row"].isin([0, 18, 1])]
    ax.scatter(sweep["sum_u"], sweep["sum_log_u"], s=22, color=BLUE,
               edgecolors="white", linewidths=0.6, zorder=4, label="Sweep solutions")
    points = [(0, "s", SECONDARY, r"$u^{C}$", (8, 1), "left", "Minimum-capacity reference"),
              (18, "*", AQUA, r"$u^{B}$", (9, -4), "left", "Designated solution"),
              (1, "D", BLUE, r"$u^{NF}$", (-1, -13), "right", "Fairness-anchor solution")]
    for row, marker, color, text, offset, align, label in points:
        match = summary[summary["row"] == row].iloc[0]
        ax.scatter(match["sum_u"], match["sum_log_u"], s=70 if marker == "*" else 30,
                   marker=marker, color=color, edgecolors="white", linewidths=0.6,
                   zorder=5, label=label)
        ax.annotate(text, (match["sum_u"], match["sum_log_u"]), xytext=offset,
                    textcoords="offset points", ha=align, color=INK, fontsize=9.5)
    ax.set_xlabel(r"Total capacity over unique directed pairs, $C(u)$")
    ax.set_ylabel(r"Sum of log capacities, $F(u)$")
    ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.3)
    save(fig, out, "calibration_frontier")


def reference_to_selection(figures, out, statistics):
    frontier, connections = figures["fixed"], figures["connections"]
    before = layer_values(frontier.capacities(0), connections)
    after = layer_values(frontier.capacities(18), connections)
    selected_minimum = min(after[k].min() for k in LAYERS)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.55), constrained_layout=True,
                             sharex=True, sharey=True)
    statistics["reference_to_selection"] = {}
    for ax, k in zip(axes, LAYERS):
        raised, lowered, same = split_change(before[k], after[k])
        log_axes(ax, 0.8, 600)
        ax.axhline(selected_minimum, color=AXIS, linewidth=0.8, zorder=1)
        scatter(ax, before[k][same], after[k][same], MUTED, zorder=3)
        scatter(ax, before[k][raised], after[k][raised], BLUE, zorder=4)
        scatter(ax, before[k][lowered], after[k][lowered], ORANGE, marker="^", zorder=5)
        ax.set_title(f"{k} ({len(before[k])} connections)")
        ax.set_xlabel(r"Reference capacity, $u^{C}$")
        corner_note(ax, f"{raised.sum()} raised\n{lowered.sum()} lowered")
        statistics["reference_to_selection"][k] = {
            "connections": int(len(before[k])),
            "raised": int(raised.sum()),
            "lowered": int(lowered.sum()),
            "unchanged": int(same.sum()),
            "at_selection_minimum": int((after[k] < selected_minimum + 1e-3).sum()),
            "at_reference_minimum": int((before[k] < 1 + 1e-3).sum()),
            "largest_decrease": float((before[k] - after[k]).max()),
            "largest_increase": float((after[k] - before[k]).max()),
        }
    axes[0].set_ylabel(r"Designated capacity, $u^{B}$")
    axes[0].annotate(
        f"$u^{{B}}$ minimum {selected_minimum:.2f}", (0.95, selected_minimum), xytext=(0, -10),
        textcoords="offset points", fontsize=8, color=SECONDARY,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=4.5,
                   label=r"Raised from $u^{C}$"),
            Line2D([], [], color=ORANGE, marker="^", linestyle="", markersize=4.5,
                   label=r"Lowered from $u^{C}$"),
            Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=4.5,
                   label="Unchanged"),
            Line2D([], [], color=MUTED, linewidth=0.8, label="Equal capacity"),
        ],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.125),
        handletextpad=0.3, columnspacing=1.4,
    )
    save(fig, out, "arc_reference_vs_selection")


def capacity_heatmap(figures, out):
    frontier, connections = figures["fixed"], figures["connections"]
    order = frontier.summary_table().sort_values("sum_u")["row"].tolist()
    columns = [layer_values(frontier.capacities(r), connections) for r in order]
    cmap = LinearSegmentedColormap.from_list("capacity", SEQUENTIAL)
    norm = LogNorm(vmin=0.01, vmax=1)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.0), constrained_layout=True)
    x_positions = [0, 1, len(order) - 1]
    x_labels = [r"$u^{C}$", r"$u^{B}$", r"$u^{NF}$"]
    x_aligns = ("right", "left", "center")
    for ax, k in zip(axes, LAYERS):
        grid = np.column_stack([column[k] for column in columns])
        grid = grid[np.argsort(grid[:, 0])]
        ceiling = grid[:, 0].max()
        ax.grid(False)
        mesh = ax.pcolormesh(
            np.arange(grid.shape[1] + 1) - 0.5, np.arange(grid.shape[0] + 1) - 0.5,
            grid / ceiling, cmap=cmap, norm=norm, rasterized=True, linewidth=0,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        for label, align in zip(ax.get_xticklabels(), x_aligns):
            label.set_ha(align)
        ax.set_title(f"{k} ({grid.shape[0]} connections)")
        ax.set_xlabel("Solutions, by increasing $C(u)$")
        ax.set_yticks([0, grid.shape[0] - 1])
        ax.set_yticklabels(["1", str(grid.shape[0])])
    axes[0].set_ylabel(r"Connection, sorted by $u^{C}$")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, aspect=26, extend="min")
    colorbar.set_label("Share of the layer ceiling (log scale)")
    colorbar.set_ticks([0.01, 0.03, 0.1, 0.3, 1])
    colorbar.ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    colorbar.ax.minorticks_off()
    colorbar.outline.set_edgecolor(AXIS)
    colorbar.outline.set_linewidth(0.5)
    save(fig, out, "arc_capacity_heatmap")


def merged_series(frontier, matched, layers):
    """Sweep rows plus matched-budget solves, by decreasing total capacity."""
    rows = []
    for row in range(1, frontier.n_points):
        rows.append((frontier.capacities(row), False))
    for row in range(matched.n_points):
        rows.append((matched.capacities(row), True))
    records = []
    for capacities, supplementary in rows:
        values = list(capacities.values())
        record = {"total": sum(values), "log_sum": sum(math.log(v) for v in values),
                  "minimum": min(values), "supplementary": supplementary}
        for k in layers:
            record[k] = max(v for key, v in capacities.items() if key[2] == k)
        records.append(record)
    return sorted(records, key=lambda r: -r["total"])


def rule_handles():
    return [
        Line2D([], [], color=BLUE, marker="o", markersize=4, linewidth=1.3,
               label="Fixed ceilings"),
        Line2D([], [], color=ORANGE, marker="^", markersize=4, linewidth=1.3,
               label="Moving ceilings"),
        Line2D([], [], color=SECONDARY, marker="o", markerfacecolor="white", markersize=4,
               linestyle="", label="Supplementary solve"),
    ]


def plot_rule(ax, records, field, color, marker):
    totals = [r["total"] for r in records]
    values = [r[field] for r in records]
    ax.plot(totals, values, color=color, linewidth=1.3, zorder=3)
    for record in records:
        ax.plot(
            record["total"], record[field], marker=marker, color=color, markersize=3.6,
            markerfacecolor="white" if record["supplementary"] else color,
            markeredgewidth=0.7, linestyle="", zorder=4, clip_on=False,
        )


def layer_maximum_paths(figures, out):
    fixed = merged_series(figures["fixed"], figures["matched_fixed"], LAYERS)
    moving = merged_series(figures["moving"], figures["matched_moving"], LAYERS)
    bounds = figures["analysis"]["throughput_bounds"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5), constrained_layout=True)
    ticks = [10000, 15000, 20000, 30000, 50000]
    for ax, k in zip(axes, LAYERS):
        plot_rule(ax, fixed, k, BLUE, "o")
        plot_rule(ax, moving, k, ORANGE, "^")
        bound = bounds[k]["bound"]
        ax.axhline(bound, color=SECONDARY, linewidth=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(53000, 9800)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(compact_thousands))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(k)
        span = max(r[k] for r in fixed) - bound
        ax.set_ylim(bound - 0.16 * span, bound + 1.08 * span)
        ax.annotate(
            f"Bound {bound:,.2f}", (0.03, bound), xycoords=("axes fraction", "data"),
            xytext=(0, -9), textcoords="offset points", fontsize=8, color=SECONDARY,
        )
    axes[0].set_ylabel("Largest capacity in the layer")
    fig.supxlabel("Total capacity (the budget tightens to the right)", fontsize=9)
    fig.legend(handles=rule_handles() + [
        Line2D([], [], color=SECONDARY, linewidth=0.7, label="Throughput bound"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), handletextpad=0.4,
        columnspacing=1.4)
    save(fig, out, "ceiling_layer_maxima")


def tradeoff_comparison(figures, out, statistics):
    fixed = merged_series(figures["fixed"], figures["matched_fixed"], LAYERS)
    moving = merged_series(figures["moving"], figures["matched_moving"], LAYERS)
    floor = figures["analysis"]["floor_total"]
    selected = figures["fixed"].point_summary(18)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7), constrained_layout=True)
    panels = (
        (axes[0], "log_sum", r"Sum of log capacities, $F(u)$", selected["sum_log_u"]),
        (axes[1], "minimum", "Smallest arc capacity", selected["min_capacity"]),
    )
    for ax, field, label, selected_value in panels:
        plot_rule(ax, [r for r in fixed if r["total"] < 24500], field, BLUE, "o")
        plot_rule(ax, [r for r in moving if r["total"] < 24500], field, ORANGE, "^")
        ax.scatter([selected["sum_u"]], [selected_value], marker="*", s=70, color=AQUA,
                   edgecolors="white", linewidths=0.5, zorder=6, clip_on=False)
        ax.axvline(floor, color=SECONDARY, linewidth=0.8, zorder=2)
        ax.set_xlim(10000, 24000)
        ax.set_xlabel(r"Total capacity, $C(u)$")
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    for ax in axes:
        ax.annotate(
            f"Moving-rule floor {floor:,.0f}", (floor, ax.get_ylim()[1]), xytext=(4, -9),
            textcoords="offset points", fontsize=8, color=SECONDARY,
        )
    fig.legend(handles=rule_handles() + [
        Line2D([], [], color=AQUA, marker="*", markersize=8, linestyle="",
               label=r"Designated $u^{B}$"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.085), handletextpad=0.4,
        columnspacing=1.4)
    save(fig, out, "ceiling_tradeoff")
    statistics["floor_premium_percent"] = 100 * (floor / figures["minimum_total"] - 1)


def ratio_statistics(before, after):
    ratio = np.abs(np.log(after / before))
    return {
        "median_ratio_percent": float(100 * (np.exp(np.median(ratio)) - 1)),
        "largest_ratio_percent": float(100 * (np.exp(ratio.max()) - 1)),
        "largest_percent_of_before": float(100 * (np.abs(after - before) / before).max()),
        "within_5_percent": int((ratio <= math.log(1.05)).sum()),
        "lowered": int((after < before - UNCHANGED_TOLERANCE).sum()),
        "raised": int((after > before + UNCHANGED_TOLERANCE).sum()),
        "connections": int(len(ratio)),
    }


def rules_by_arc(figures, out, statistics):
    """Fixed against moving, connection by connection, at two common budgets."""
    connections = figures["connections"]
    analysis = figures["analysis"]
    budgets = [s["budget"] for s in analysis["matched_budget"] if s["rule"] == "fixed"]
    index = budgets.index(analysis["prefix_selection_budget"])
    last_row = figures["moving"].n_points - 1
    pairs = (
        (figures["fixed"].capacities(last_row), figures["moving"].capacities(last_row)),
        (figures["matched_fixed"].capacities(index), figures["matched_moving"].capacities(index)),
    )
    bounds = analysis["throughput_bounds"]
    labels = []
    for fixed_capacities, _ in pairs:
        labels.append(sum(fixed_capacities.values()))
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.75), constrained_layout=True,
                             sharex=True, sharey=True)
    statistics["rules_by_arc"] = []
    for line, (fixed_capacities, moving_capacities), total in zip(axes, pairs, labels):
        fixed_values = layer_values(fixed_capacities, connections)
        moving_values = layer_values(moving_capacities, connections)
        entry = {"total": total, "layers": {}}
        for ax, k in zip(line, LAYERS):
            log_axes(ax, 5, 600)
            ax.axhline(bounds[k]["bound"], color=AXIS, linewidth=0.8, zorder=1)
            raised, lowered, same = split_change(fixed_values[k], moving_values[k])
            scatter(ax, fixed_values[k][same], moving_values[k][same], MUTED, zorder=3)
            scatter(ax, fixed_values[k][raised], moving_values[k][raised], BLUE, zorder=4)
            scatter(ax, fixed_values[k][lowered], moving_values[k][lowered], ORANGE,
                    marker="^", zorder=5)
            ax.annotate(
                f"Ceiling {bounds[k]['bound']:,.2f}", (5.6, bounds[k]["bound"]),
                xytext=(0, 3), textcoords="offset points", fontsize=8, color=SECONDARY,
            )
            corner_note(ax, f"{raised.sum()} raised\n{lowered.sum()} lowered")
            entry["layers"][k] = ratio_statistics(fixed_values[k], moving_values[k])
        line[0].set_ylabel(f"Moving ceilings, $C={total:,.0f}$")
        statistics["rules_by_arc"].append(entry)
    for ax, k in zip(axes[0], LAYERS):
        ax.set_title(k)
    for ax in axes[1]:
        ax.set_xlabel("Fixed ceilings, same budget")
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=4.5,
                   label="Higher under moving ceilings"),
            Line2D([], [], color=ORANGE, marker="^", linestyle="", markersize=4.5,
                   label="Lower under moving ceilings"),
            Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=4.5,
                   label="Unchanged"),
            Line2D([], [], color=MUTED, linewidth=0.8, label="Equal capacity"),
        ],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05),
        handletextpad=0.3, columnspacing=1.2,
    )
    save(fig, out, "arc_fixed_vs_moving")


def prefix_vs_corrected(figures, out, statistics):
    """The earlier selection against the corrected solve at the same budget."""
    connections = figures["connections"]
    analysis = figures["analysis"]
    budgets = [s["budget"] for s in analysis["matched_budget"] if s["rule"] == "fixed"]
    index = budgets.index(analysis["prefix_selection_budget"])
    earlier = layer_values(figures["benchmark"].capacities(18), connections)
    corrected = layer_values(figures["matched_fixed"].capacities(index), connections)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.45), constrained_layout=True,
                             sharex=True, sharey=True)
    statistics["prefix_vs_corrected"] = {}
    for ax, k in zip(axes, LAYERS):
        log_axes(ax, 5, 600)
        scatter(ax, earlier[k], corrected[k], BLUE)
        entry = ratio_statistics(earlier[k], corrected[k])
        statistics["prefix_vs_corrected"][k] = entry
        corner_note(ax, f"{entry['within_5_percent']} of {entry['connections']}\nwithin 5%")
        ax.set_title(k)
        ax.set_xlabel(r"Earlier designated solution, $u^{P}$")
    axes[0].set_ylabel("Corrected, same budget")
    save(fig, out, "arc_prefix_vs_corrected")


def procedure_overview(out):
    """The sweep as a sequence of solves, with the one step where the rules differ."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, width, height, lines, face="#f4f3ef", edge=AXIS):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0,rounding_size=1.0",
            facecolor=face, edgecolor=edge, linewidth=0.8,
        ))
        ax.text(x + width / 2, y + height / 2, lines, ha="center", va="center",
                fontsize=7.8, color=INK, linespacing=1.35)

    def arrow(start, end, color=SECONDARY, connection="arc3"):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color,
            connectionstyle=connection, shrinkA=0, shrinkB=0,
        ))

    top, height = 16, 14
    box(0, top, 21, height, "Minimum-capacity\nsolve: $\\min\\ C(u)$\ngives $u^{C}$")
    box(25, top, 21, height, "Set layer ceilings:\n$\\overline{u}_k=$ largest\n$u^{C}$ in layer $k$")
    box(50, top, 21, height, "Fairness anchor:\n$\\max\\ F(u)$ sets\nthe first budget")
    box(75, top, 25, height, "Budgeted solve:\n$\\max\\ F(u)$ subject to\n$C(u)\\leq$ budget, $u\\leq\\overline{u}_k$",
        face="#e6f0fc", edge=BLUE)
    for left, right in ((21, 25), (46, 50), (71, 75)):
        arrow((left, top + height / 2), (right, top + height / 2))
    arrow((94, top + height), (81, top + height), color=BLUE, connection="arc3,rad=0.75")
    ax.text(87.5, 37.8, "Tighten the budget and repeat", ha="center", va="top", fontsize=7.8,
            color=SECONDARY)
    ax.text(0, 12.6, "After each feasible budgeted solve, the two rules differ:", fontsize=7.8,
            color=SECONDARY, va="center")
    box(0, 0.5, 48, 9.5, "Fixed rule: keep the ceilings set from $u^{C}$", face="#ffffff", edge=BLUE)
    box(52, 0.5, 48, 9.5, "Moving rule: replace each ceiling with the\nlayer's largest capacity in this solve",
        face="#ffffff", edge=ORANGE)
    save(fig, out, "procedure_overview")


def node_positions(k):
    frame = pd.read_csv(os.path.join(DATA_DIR, f"{k}Nodes.csv"))
    return {f"{k}_node_{row.ID}": (row.X, row.Y) for row in frame.itertuples()}


def capacity_width(value):
    return 0.3 + 0.17 * math.sqrt(value)


def draw_map(ax, k, connections, networks, values, colors):
    positions = node_positions(k)
    order = np.argsort(values)
    segments = [(positions[connections[k][m][0]], positions[connections[k][m][1]]) for m in order]
    ax.add_collection(LineCollection(
        segments, linewidths=[capacity_width(values[m]) for m in order],
        colors=[colors[m] for m in order], capstyle="round", zorder=2,
    ))
    supply = set(networks.supply_nodes(k))
    plain = [p for node, p in positions.items() if node not in supply]
    source = [p for node, p in positions.items() if node in supply]
    ax.scatter(*zip(*plain), s=2.2, color=SECONDARY, zorder=3, linewidths=0)
    ax.scatter(*zip(*source), s=11, color=INK, marker="s", zorder=4, edgecolors="white",
               linewidths=0.3)
    points = np.array(list(positions.values()))
    margin = 0.03
    ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax.set_aspect(1 / math.cos(math.radians(35.15)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def width_handles():
    return [Line2D([], [], color=SECONDARY, linewidth=capacity_width(value), label=f"Capacity {value}")
            for value in (1, 10, 100, 400)]


def capacity_maps(figures, out):
    connections = figures["connections"]
    frontier = figures["fixed"]
    rows = ((0, r"Reference, $u^{C}$"), (18, r"Designated, $u^{B}$"))
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.75), constrained_layout=True)
    for line, (row, label) in zip(axes, rows):
        values = layer_values(frontier.capacities(row), connections)
        smallest = min(values[k].min() for k in LAYERS)
        for ax, k in zip(line, LAYERS):
            colors = [ORANGE if v < smallest + 1e-3 else BLUE for v in values[k]]
            draw_map(ax, k, connections, figures["networks"], values[k], colors)
        line[0].set_ylabel(label)
    for ax, k in zip(axes[0], LAYERS):
        ax.set_title(k)
    fig.legend(
        handles=width_handles() + [
            Line2D([], [], color=ORANGE, linewidth=1.2, label="At the solution's minimum"),
            Line2D([], [], color=INK, marker="s", linestyle="", markersize=3.2, label="Supply node"),
        ],
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.055), handletextpad=0.4, columnspacing=1.0,
        fontsize=8,
    )
    save(fig, out, "capacity_maps")


def capacity_profiles(figures, out):
    connections = figures["connections"]
    frontier = figures["fixed"]
    series = ((0, r"$u^{C}$", MUTED, "-"), (18, r"$u^{B}$", BLUE, "-"), (1, r"$u^{NF}$", AQUA, "-"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.35), constrained_layout=True, sharey=True)
    for ax, k in zip(axes, LAYERS):
        for row, label, color, style in series:
            values = np.sort(layer_values(frontier.capacities(row), connections)[k])
            share = (np.arange(len(values)) + 0.5) / len(values) * 100
            ax.plot(share, values, color=color, linewidth=1.25, linestyle=style, label=label,
                    drawstyle="steps-mid")
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.8, 600)
        ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0, 50, 100])
        ax.set_title(k)
    axes[0].set_ylabel("Connection capacity (log scale)")
    fig.supxlabel("Connections, from smallest to largest capacity (%)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    names = {r"$u^{C}$": r"Reference $u^{C}$", r"$u^{B}$": r"Designated $u^{B}$",
             r"$u^{NF}$": r"Fairness-anchor solution $u^{NF}$"}
    fig.legend(handles, [names[label] for label in labels], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1), handletextpad=0.5, columnspacing=1.6)
    save(fig, out, "capacity_profiles")


def gas_node_example(figures, out, statistics):
    """The two arcs at Gas node 1, which set that layer's throughput bound."""
    first = ("Gas_node_5", "Gas_node_1", "Gas", 0)
    second = ("Gas_node_1", "Gas_node_13", "Gas", 0)
    analysis = figures["analysis"]
    budgets = [s["budget"] for s in analysis["matched_budget"] if s["rule"] == "fixed"]
    index = budgets.index(analysis["prefix_selection_budget"])
    fixed_solve = figures["matched_fixed"].capacities(index)
    moving_solve = figures["matched_moving"].capacities(index)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.45), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    rows = (("Moving ceilings", moving_solve, ORANGE, "#f7c3ad"), ("Fixed ceilings", fixed_solve, BLUE, "#a9cbf2"))
    for position, (label, capacities, strong, light) in enumerate(rows):
        a, b = capacities[first], capacities[second]
        ax.barh(position, a, color=strong, height=0.56, edgecolor="white", linewidth=1)
        ax.barh(position, b, left=a, color=light, height=0.56, edgecolor="white", linewidth=1)
        ax.text(a / 2, position, f"Arc to node 5\n{a:.2f}", ha="center", va="center", color="white",
                fontsize=8, linespacing=1.2)
        ax.text(a + b / 2, position, f"Arc to node 13\n{b:.2f}", ha="center", va="center", color=INK,
                fontsize=8, linespacing=1.2)
        ax.text(0, position + 0.32, label, ha="left", va="bottom", color=INK, fontsize=8.5)
    ax.axvline(351.35, color=SECONDARY, linewidth=0.8, zorder=3, ymax=0.93)
    ax.text(357, 1.62, "Even split 351.35", ha="left", va="center", fontsize=8, color=SECONDARY)
    ax.set_xlim(0, 703)
    ax.set_ylim(-0.42, 1.74)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel(f"Capacity at node 1, at total capacity {budgets[index]:,.0f}")

    ax = axes[1]
    fixed = merged_arc_series(figures["fixed"], figures["matched_fixed"], first, second)
    moving = merged_arc_series(figures["moving"], figures["matched_moving"], first, second)
    ax.plot(fixed[0], fixed[1], color=BLUE, linewidth=1.4, label="Fixed: arc to node 5")
    ax.plot(fixed[0], fixed[2], color="#8cb9ee", linewidth=1.4, label="Fixed: arc to node 13")
    ax.plot(moving[0], moving[1], color=ORANGE, linewidth=1.4, label="Moving: both arcs")
    ax.set_xscale("log")
    ax.set_xlim(53000, 9800)
    ax.xaxis.set_major_locator(FixedLocator([10000, 15000, 20000, 30000, 50000]))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_thousands))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Total capacity (the budget tightens to the right)")
    ax.set_ylabel("Capacity of the arc")
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2, fontsize=8)
    save(fig, out, "gas_node_example")
    statistics["gas_node_example"] = {
        "fixed": [fixed_solve[first], fixed_solve[second]],
        "moving": [moving_solve[first], moving_solve[second]],
    }


def merged_arc_series(frontier, matched, first, second):
    rows = [frontier.capacities(r) for r in range(1, frontier.n_points)]
    rows += [matched.capacities(r) for r in range(matched.n_points)]
    rows.sort(key=lambda capacities: -sum(capacities.values()))
    return ([sum(c.values()) for c in rows], [c[first] for c in rows], [c[second] for c in rows])


def rule_change_map(figures, out, statistics):
    """Where capacity goes when the ceilings move, at one common budget."""
    connections = figures["connections"]
    analysis = figures["analysis"]
    budgets = [s["budget"] for s in analysis["matched_budget"] if s["rule"] == "fixed"]
    index = budgets.index(analysis["prefix_selection_budget"])
    fixed = layer_values(figures["matched_fixed"].capacities(index), connections)
    moving = layer_values(figures["matched_moving"].capacities(index), connections)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.95), constrained_layout=True)
    counts = {}
    for ax, k in zip(axes, LAYERS):
        ratio = moving[k] / fixed[k]
        colors = [BLUE if r > 1.05 else ORANGE if r < 1 / 1.05 else AXIS for r in ratio]
        draw_map(ax, k, connections, figures["networks"], moving[k], colors)
        ax.set_title(k)
        counts[k] = {"higher": int((ratio > 1.05).sum()), "lower": int((ratio < 1 / 1.05).sum()),
                     "within_5_percent": int(((ratio <= 1.05) & (ratio >= 1 / 1.05)).sum())}
    statistics["rule_change_map"] = counts
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, linewidth=1.4, label="More than 5% higher under moving ceilings"),
            Line2D([], [], color=ORANGE, linewidth=1.4, label="More than 5% lower"),
            Line2D([], [], color=AXIS, linewidth=1.4, label="Within 5%"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11), handletextpad=0.4, columnspacing=1.2,
        fontsize=8,
    )
    save(fig, out, "rule_change_map")


def rule_difference_heatmap(figures, out, statistics):
    connections = figures["connections"]
    fixed, moving = figures["fixed"], figures["moving"]
    matched_fixed, matched_moving = figures["matched_fixed"], figures["matched_moving"]
    pairs = [(fixed.capacities(r), moving.capacities(r), sum(fixed.capacities(r).values()))
             for r in range(9, moving.n_points)]
    pairs += [
        (matched_fixed.capacities(r), matched_moving.capacities(r),
         sum(matched_fixed.capacities(r).values()))
        for r in range(matched_fixed.n_points)
    ]
    pairs.sort(key=lambda pair: -pair[2])
    fixed_columns = [layer_values(p[0], connections) for p in pairs]
    moving_columns = [layer_values(p[1], connections) for p in pairs]
    cmap = LinearSegmentedColormap.from_list("rule", [DIVERGING_LOW, DIVERGING_MID, BLUE])
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.0), constrained_layout=True)
    totals = [p[2] for p in pairs]
    tick_positions = [0, len(totals) // 2, len(totals) - 1]
    for ax, k in zip(axes, LAYERS):
        fixed_grid = np.column_stack([c[k] for c in fixed_columns])
        moving_grid = np.column_stack([c[k] for c in moving_columns])
        percent = 100 * (moving_grid / fixed_grid - 1)
        order = np.argsort(percent[:, -1])
        ax.grid(False)
        mesh = ax.pcolormesh(
            np.arange(percent.shape[1] + 1) - 0.5, np.arange(percent.shape[0] + 1) - 0.5,
            percent[order], cmap=cmap, norm=norm, rasterized=True, linewidth=0,
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{totals[i] / 1000:.1f}k" for i in tick_positions])
        ax.set_title(f"{k} ({percent.shape[0]} connections)")
        ax.set_yticks([0, percent.shape[0] - 1])
        ax.set_yticklabels(["1", str(percent.shape[0])])
        statistics.setdefault("rule_difference", {})[k] = {
            "tightest_budget_lowest_percent": float(percent[:, -1].min()),
            "tightest_budget_highest_percent": float(percent[:, -1].max()),
        }
    axes[0].set_ylabel("Connection, sorted by final change")
    fig.supxlabel("Common budget (columns ordered from loosest to tightest)", fontsize=9)
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, aspect=26, extend="both")
    colorbar.set_label("Moving vs. fixed capacity (%)")
    colorbar.outline.set_edgecolor(AXIS)
    colorbar.outline.set_linewidth(0.5)
    save(fig, out, "arc_rule_difference")


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.03,
                metadata={"CreationDate": None}, dpi=400)
    plt.close(fig)


def load_inputs():
    networks = load_networks()
    fixed = load_frontier()
    with open(os.path.join(DATA_DIR, f"paper_analysis_{TAG}.json"), encoding="utf-8") as stream:
        analysis = json.load(stream)
    return {
        "networks": networks,
        "connections": undirected_connections(networks),
        "benchmark": load_benchmark(),
        "fixed": fixed,
        "minimum_total": sum(fixed.capacities(0).values()),
        "analysis": analysis,
        "moving": load_frontier(filename=f"pareto_moving_ceiling_{TAG}.csv"),
        "matched_fixed": load_frontier(filename=f"matched_budget_fixed_{TAG}.csv"),
        "matched_moving": load_frontier(filename=f"matched_budget_moving_{TAG}.csv"),
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    figures = load_inputs()
    statistics = {}
    procedure_overview(out)
    frontier_overview(figures, out)
    capacity_maps(figures, out)
    capacity_profiles(figures, out)
    reference_to_selection(figures, out, statistics)
    capacity_heatmap(figures, out)
    layer_maximum_paths(figures, out)
    gas_node_example(figures, out, statistics)
    tradeoff_comparison(figures, out, statistics)
    rule_change_map(figures, out, statistics)
    rules_by_arc(figures, out, statistics)
    rule_difference_heatmap(figures, out, statistics)
    prefix_vs_corrected(figures, out, statistics)
    with open(os.path.join(out, "figure_statistics.json"), "w", encoding="utf-8") as stream:
        json.dump(statistics, stream, indent=2)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
