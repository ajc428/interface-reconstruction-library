#!/usr/bin/env python3
"""Audit the one-plane cell dump (r2p_plic_cells_*.txt from R2P_PLIC_DUMP / r2p_plic_dump).

For every dumped cell, decide from its 5^3 volume-fraction block whether a
single plane is geometrically possible, or whether both faces of a thin film
must cross the cell (so one plane is wrong).

Per cell, in cell units (centre cell = [-1/2, 1/2]^3), film phase = liquid,
or gas when the dump says the film is gas:
  1. Film normal n and thickness t: along the axis closest to the film's
     normal (first guess: principal axes of the film cloud), take the 5x5
     columns of 5 cells. In each column that holds the film whole (empty end
     cells, film cells contiguous), the film's mid-position follows from its
     VFs: inner cells full, end cells' film against their inner faces. A plane
     fitted through those positions gives n; t = the centre column's sum
     * |n_axis| (exact for a flat slab), or the 3x3 central columns' median
     if the centre column does not hold the film whole.
  2. Thin sheet: film in all 9 central columns, at least MIN_COLUMNS columns
     fitted, a flat film cloud (eigenvalue ratio below MAX_FLATNESS) and
     t < MAX_THICKNESS. Other cells (resolved interfaces, drops, ligaments,
     film ends) are only counted.
  3. One-face depth D: how deep into the cell, along n, the single plane with
     normal n that holds the cell's film VF reaches. D <= t: the far face is
     outside the cell, one plane is right (a corner clip). D > t: the far face
     is inside too, the cell needs two planes.
On synthetic flat films (t = 0.005-0.3 cells, random normals and offsets),
D/t > RATIO_WRONG flags 96% of the cells both faces cross and 1 of 276 corner
clips. Axis-aligned films through the cell read D/t ~ 1 and are missed.
Most wrong cells are slivers at the edge of a tilted film's footprint, where
one plane misplaces almost nothing. The share film VF / t (about 1 where the
film crosses the whole cell) separates those from cells that hold a real
piece of the film (share > MIN_SHARE), which are listed.

Usage:
  python3 r2p_plic_audit.py FILE [FILE ...] [--csv flagged.csv] [--top N]
The CSV holds the flagged cells (i, j, k as coordinates) for ParaView.
"""
import math
import statistics
import sys
from collections import Counter, defaultdict

EMPTY = 1e-12          # film VF at or below this is empty (VFlo)
MIN_COLUMNS = 9        # fitted columns, of the 25
MAX_FLATNESS = 0.1     # smallest / largest eigenvalue of the film cloud
MAX_THICKNESS = 1.0    # cells; thicker is not a sub-grid film
MAX_SPREAD = 3.0       # largest / smallest central column sum of a uniform film
RATIO_WRONG = 1.2      # D / t above this: one plane is clearly wrong
MIN_SHARE = 0.3        # film VF / t above this: the cell holds a real piece of the film

REASONS = {1: "PLIC routing", 2: "domain boundary", 3: "network one plane", 4: "Newton clean-up",
           5: "pinch prevention", 6: "pass-2 selection", 7: "pass-2 clean-up"}


def eigh3(a):
    """Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi)."""
    a = [row[:] for row in a]
    v = [[1.0 if r == c else 0.0 for c in range(3)] for r in range(3)]
    for _ in range(50):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-15 * (abs(a[0][0]) + abs(a[1][1]) + abs(a[2][2]) + 1e-300):
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-300:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(3)], [[v[r][c] for r in range(3)] for c in range(3)]


def below_volume(m, s):
    """Volume of {m . x <= s} in the unit cube [0,1]^3, m >= 0 componentwise."""
    m = [max(x, 1e-4) for x in m]
    tot = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                r = s - (a * m[0] + b * m[1] + c * m[2])
                if r > 0.0:
                    tot += (-1) ** (a + b + c) * r ** 3
    return min(max(tot / (6.0 * m[0] * m[1] * m[2]), 0.0), 1.0)


def one_face_depth(n, f):
    """Depth along n of the one plane, normal n, that cuts film VF f off a cell corner."""
    m = [abs(x) for x in n]
    lo, hi = 0.0, sum(m)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if below_volume(m, mid) < f:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cube_slab_vf(n, lo, hi):
    """Volume of {lo <= n . x <= hi} in the cell [-1/2, 1/2]^3 (n unit)."""
    m = [abs(x) for x in n]
    shift = 0.5 * sum(m)   # n . x ranges over [-shift, shift]
    return below_volume(m, hi + shift) - below_volume(m, lo + shift)


def column_mid(col, lo):
    """Mid-position along a column of the film it holds, and the fit weight.
    Cells between the first and last film cell count as full; the end cells'
    film sits against their inner faces. One film cell: its centre, low weight
    (the film's place inside the cell is unknown)."""
    idx = [s for s, x in enumerate(col) if x > EMPTY]
    first, last = idx[0], idx[-1]
    if first == last:
        return lo + first, 0.05
    a = (lo + first + 0.5) - col[first]
    b = (lo + last - 0.5) + col[last]
    return 0.5 * (a + b), 1.0


def fit_film(f, n0):
    """Fit the film's mid-surface over the 5x5 columns along the axis closest
    to n0. Returns (n, off, t, columns, nonempty, uniform): n . x = off at the
    mid-surface, t the thickness, all in cell units."""
    film = lambda p: f[(p[0] + 2) + 5 * (p[1] + 2) + 25 * (p[2] + 2)]
    ax = max(range(3), key=lambda k: abs(n0[k]))
    l1, l2 = [k for k in range(3) if k != ax]
    rows, sums, nonempty, centre = [], [], 0, None
    for o1 in range(-2, 3):
        for o2 in range(-2, 3):
            col = []
            for s in range(-2, 3):
                p = [0, 0, 0]
                p[ax], p[l1], p[l2] = s, o1, o2
                col.append(film(p))
            tot = sum(col)
            if abs(o1) <= 1 and abs(o2) <= 1 and tot > EMPTY:
                nonempty += 1
            if tot <= EMPTY or col[0] > EMPTY or col[-1] > EMPTY:
                continue
            idx = [s for s, x in enumerate(col) if x > EMPTY]
            if idx[-1] - idx[0] + 1 != len(idx):
                continue   # film not contiguous along the column
            z, w = column_mid(col, -2)
            rows.append((o1, o2, z, w))
            if abs(o1) <= 1 and abs(o2) <= 1:
                sums.append(tot)
            if o1 == 0 and o2 == 0:
                centre = tot
    if len(rows) < 6 or not sums:
        return None
    # weighted least squares z = a + b o1 + c o2
    A = [[0.0] * 3 for _ in range(3)]
    r = [0.0] * 3
    for o1, o2, z, w in rows:
        phi = (1.0, o1, o2)
        for p in range(3):
            r[p] += w * phi[p] * z
            for q in range(3):
                A[p][q] += w * phi[p] * phi[q]
    M = [A[p][:] + [r[p]] for p in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda k: abs(M[k][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for k in range(3):
            if k != c:
                g = M[k][c] / M[c][c]
                M[k] = [x - g * y for x, y in zip(M[k], M[c])]
    a, b, c = (M[p][3] / M[p][p] for p in range(3))
    norm = math.sqrt(1.0 + b * b + c * c)
    n = [0.0, 0.0, 0.0]
    n[ax], n[l1], n[l2] = 1.0 / norm, -b / norm, -c / norm
    # thickness through the cell itself where its column holds the film whole
    # (a film whose thickness varies, e.g. near its end), else the 3x3 median
    t = (centre if centre is not None else statistics.median(sums)) / norm
    # uniform film: the 9 central columns all hold it whole, at similar amounts
    uniform = len(sums) == 9 and max(sums) <= MAX_SPREAD * min(sums)
    return n, a / norm, t, len(rows), nonempty, uniform


def analyse(block, film_is_gas):
    """Return (n, off, t, columns, nonempty, uniform, flatness) or None if no film."""
    f = [1.0 - x if film_is_gas else x for x in block]
    w = sx = sy = sz = 0.0
    pts = []
    for c in range(-2, 3):
        for b in range(-2, 3):
            for a in range(-2, 3):
                x = f[(a + 2) + 5 * (b + 2) + 25 * (c + 2)]
                if x > EMPTY:
                    pts.append((x, a, b, c))
                    w += x; sx += x * a; sy += x * b; sz += x * c
    if w <= 0.0 or len(pts) < 4:
        return None
    cx, cy, cz = sx / w, sy / w, sz / w
    cov = [[0.0] * 3 for _ in range(3)]
    for x, a, b, c in pts:
        d = (a - cx, b - cy, c - cz)
        for r in range(3):
            for s in range(3):
                cov[r][s] += x * d[r] * d[s]
    vals, vecs = eigh3(cov)
    order = sorted(range(3), key=lambda k: vals[k])
    flat = vals[order[0]] / vals[order[2]] if vals[order[2]] > 0 else 1.0
    fit = fit_film(f, vecs[order[0]])
    if fit is None:
        return None
    n = fit[0]
    if max(range(3), key=lambda k: abs(n[k])) != max(range(3), key=lambda k: abs(vecs[order[0]][k])):
        fit = fit_film(f, n) or fit   # refit along the better axis
    return fit + (flat,)


def read_cells(paths):
    """Yield (step, i, j, k, reason, class, guard, film_is_gas, VF, 5^3 VFs) per
    dumped cell. step is the time step; dumps written before the step column
    was added give the reconstruction call number there instead."""
    for path in paths:
        with open(path) as fh:
            while True:
                head = fh.readline()
                if not head:
                    break
                if not head.startswith("cell"):
                    continue
                vals = fh.readline().split()
                h = head.split()
                if len(vals) != 125 or len(h) not in (10, 11):
                    continue   # a record cut off (file still being written)
                if len(h) == 11:   # cell <call> <step> ...: report by time step
                    h = h[:1] + h[2:]
                yield (int(h[1]), int(h[2]), int(h[3]), int(h[4]), int(h[5]), int(h[6]), int(h[7]),
                       h[8] == "T", float(h[9]), [float(x) for x in vals])


def main(argv):
    files, csv_path, top = [], None, 25
    it = iter(argv)
    for arg in it:
        if arg == "--csv":
            csv_path = next(it)
        elif arg == "--top":
            top = int(next(it))
        else:
            files.append(arg)
    if not files:
        sys.exit(__doc__)

    total, sheet, wrong, border, neck, neck_wrong, big, neck_big = (Counter() for _ in range(8))
    wrong_guard, wrong_class = defaultdict(Counter), defaultdict(Counter)
    per_call = defaultdict(Counter)
    flagged = []
    for call, i, j, k, reason, cls, guard, gas, vf, block in read_cells(files):
        total[reason] += 1
        r = analyse(block, gas)
        if r is None:
            continue
        n, off, t, columns, nonempty, uniform, flat = r
        if columns < MIN_COLUMNS or nonempty < 9 or flat > MAX_FLATNESS or not t < MAX_THICKNESS:
            continue
        fvf = 1.0 - vf if gas else vf
        ratio = one_face_depth(n, fvf) / t
        if not uniform:
            # thickness changes across the stencil (a neck, a rim, a film end):
            # the slab test is only indicative there
            neck[reason] += 1
            if ratio > RATIO_WRONG:
                neck_wrong[reason] += 1
                neck_big[reason] += fvf / t > MIN_SHARE
                flagged.append((ratio, call, i, j, k, reason, cls, guard, gas, fvf, t, n, False))
            continue
        sheet[reason] += 1
        if ratio > RATIO_WRONG:
            wrong[reason] += 1
            big[reason] += fvf / t > MIN_SHARE
            wrong_guard[reason][guard] += 1
            wrong_class[reason][cls] += 1
            per_call[call][reason] += 1
            flagged.append((ratio, call, i, j, k, reason, cls, guard, gas, fvf, t, n, True))
        elif ratio > 1.0:
            border[reason] += 1

    print("Uniform thin films: the slab test is reliable. Non-uniform films (necks, rims, film")
    print("ends): only indicative, since one plane may follow the thicker side's surface.\n")
    print(f"{'':<31}{'---- uniform films ----':>31}{'-- non-uniform films --':>30}")
    print(f"{'reason':<22}{'dumped':>9}{'cells':>8}{'D/t 1-1.2':>10}{'wrong':>7}{'big':>6}"
          f"{'cells':>10}{'D/t>1.2':>9}{'big':>6}")
    rows = [(str(r) + " " + REASONS.get(r, "?"), r) for r in sorted(total)] + [("all", None)]
    for label, r in rows:
        get = (lambda c: c[r]) if r is not None else (lambda c: sum(c.values()))
        print(f"{label:<22}{get(total):>9}{get(sheet):>8}{get(border):>10}{get(wrong):>7}{get(big):>6}"
              f"{get(neck):>10}{get(neck_wrong):>9}{get(neck_big):>6}")
    print(f"big: the wrong cell's film VF exceeds {MIN_SHARE} x the film thickness (a real piece of film, "
          "not a sliver)")

    if wrong:
        print("\nWrong one-plane cells in uniform films, by reason: guard values / classifier classes")
        for reason in sorted(wrong):
            g = ", ".join(f"{k}: {v}" for k, v in sorted(wrong_guard[reason].items()))
            c = ", ".join(f"{k}: {v}" for k, v in sorted(wrong_class[reason].items()))
            print(f"  {reason} {REASONS.get(reason, '?'):<20} guard {{{g}}}  class {{{c}}}")
        print("\nWrong one-plane cells in uniform films per time step (first and last 10 steps with any)")
        calls = sorted(per_call)
        for call in calls[:10] + (["..."] if len(calls) > 20 else []) + calls[max(10, len(calls) - 10):]:
            if call == "...":
                print("  ...")
                continue
            print(f"  step {call:>5}: " + ", ".join(f"reason {k}: {v}" for k, v in sorted(per_call[call].items())))
    big_cells = sorted((x for x in flagged if x[9] / x[10] > MIN_SHARE), key=lambda x: x[9] / x[10], reverse=True)
    if big_cells:
        print(f"\nWrong one-plane cells holding a real piece of film ({len(big_cells)}; largest share first, "
              f"top {min(top, len(big_cells))})")
        print(f"  {'share':>6} {'D/t':>6} {'unif':>4} {'step':>5} {'i':>5} {'j':>5} {'k':>5} {'rsn':>3} {'cls':>3} "
              f"{'grd':>3} {'film VF':>10} {'t cells':>8}  normal")
        for ratio, call, i, j, k, reason, cls, guard, gas, fvf, t, n, uni in big_cells[:top]:
            print(f"  {fvf / t:6.2f} {ratio:6.2f} {int(uni):4d} {call:5d} {i:5d} {j:5d} {k:5d} {reason:3d} {cls:3d} "
                  f"{guard:3d} {fvf:10.3e} {t:8.4f}  ({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})")
    if csv_path:
        with open(csv_path, "w") as out:
            out.write("i,j,k,step,reason,class,guard,film_is_gas,film_vf,thickness,ratio,uniform,share\n")
            for ratio, call, i, j, k, reason, cls, guard, gas, fvf, t, n, uni in flagged:
                out.write(f"{i},{j},{k},{call},{reason},{cls},{guard},{int(gas)},{fvf:.6e},{t:.6e},{ratio:.4f},"
                          f"{int(uni)},{fvf / t:.4f}\n")
        print(f"\n{len(flagged)} flagged cells written to {csv_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
