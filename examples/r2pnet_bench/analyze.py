#!/usr/bin/env python3
"""Binned breakdown of r2pnet_bench CSV output (standard library only).

    python3 analyze.py FILE.csv [--by thickness|radius|ntrue|class] [--cat a,b] [--methods m1,m2]

For each category and bin, prints the mean symmetric difference (fraction of
the cell) per method, and the share of samples above 5% in parentheses.
"""
import argparse
import csv
import math
from collections import defaultdict

THICK_EDGES = [0.005, 0.02, 0.1, 0.3, 0.6, 1.0, 1.5, 3.0]
RADIUS_EDGES = [2, 4, 8, 16, 40]


def bin_of(value, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})"
    return f">={edges[-1]:g}" if value >= edges[-1] else f"<{edges[0]:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--by", default="thickness", choices=["thickness", "radius", "ntrue", "class"])
    ap.add_argument("--cat", default="")
    ap.add_argument("--methods", default="")
    args = ap.parse_args()

    cats_wanted = set(filter(None, args.cat.split(",")))
    methods_wanted = [m for m in args.methods.split(",") if m]

    data = defaultdict(lambda: defaultdict(list))   # (cat, bin) -> method -> [symdiff]
    methods_seen, order = [], []
    for path in args.csv:
        with open(path) as f:
            for r in csv.DictReader(f):
                cat = r["category"]
                if cats_wanted and cat not in cats_wanted:
                    continue
                m = r["method"]
                if methods_wanted and m not in methods_wanted:
                    continue
                if m not in methods_seen:
                    methods_seen.append(m)
                if args.by == "thickness":
                    b = bin_of(float(r["thickness"]), THICK_EDGES)
                elif args.by == "radius":
                    rad = abs(float(r["radius1"]))
                    b = "flat" if rad == 0 else bin_of(rad, RADIUS_EDGES)
                elif args.by == "ntrue":
                    b = f"faces={r['n_true']}"
                else:
                    b = f"class={r['class']}"
                key = (cat, b)
                if key not in data:
                    order.append(key)
                data[key][m].append(float(r["symdiff"]))

    methods = methods_wanted or methods_seen
    w = 16
    print(f"{'category':<20}{'bin':<14}{'n':>6} " + "".join(f"{m:>{w}}" for m in methods))

    def sort_key(k):
        cat, b = k
        num = b.strip("[>=<faces=class=flat").split(",")[0]
        try:
            v = float(num)
        except ValueError:
            v = -1.0 if b == "flat" else math.inf
        return (cat, v)

    for key in sorted(order, key=sort_key):
        row = data[key]
        n = max(len(v) for v in row.values())
        cells = []
        for m in methods:
            v = row.get(m, [])
            if not v:
                cells.append(f"{'-':>{w}}")
                continue
            mean = sum(v) / len(v)
            bad = 100.0 * sum(1 for x in v if x > 0.05) / len(v)
            cells.append(f"{mean:>9.2e} ({bad:>3.0f}%)")
        print(f"{key[0]:<20}{key[1]:<14}{n:>6} " + "".join(cells))


if __name__ == "__main__":
    main()
