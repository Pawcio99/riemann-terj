"""Convert an Odlyzko zero table (plain text) to the project .npz format.

Tables: https://www-users.cse.umn.edu/~odlyzko/zeta_tables/
Absolute tables (first 100 000 zeros, zeros6 = first 2 001 052 zeros):
    python -m src.odlyzko zeros6 --start-index 1 --out data/odl_1_2M.npz
High tables (10^12, 10^21, 10^22): first run `head -n 5 <file>`; if the file
states that a constant was subtracted, pass it as --base (an exact integer):
    python -m src.odlyzko zeros_1e21 --start-index 1000000000000000000001 --base <INT> --out data/odl_1e21.npz
Non-numeric lines are skipped. Never print the whole file into the context.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--start-index", required=True, help="index of the first zero in the file")
    ap.add_argument("--base", default="0", help="integer subtracted from the zeros in the file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vals = []
    with open(a.path) as f:
        for line in f:
            toks = line.split()
            if len(toks) != 1:  # data lines hold exactly one number; header prose is skipped
                continue
            try:
                vals.append(float(toks[0]))
            except ValueError:
                pass
    x = np.array(vals)
    base = int(a.base)
    if not np.all(np.diff(x) > 0):
        raise SystemExit("values are not strictly increasing: check the file format (head -n 20)")
    np.savez(a.out, base=str(base), x=x, start_index=str(int(a.start_index)))
    print(f"saved {len(x)} zeros, base={base}, offsets {x[0]:.6f} .. {x[-1]:.6f} -> {a.out}")


if __name__ == "__main__":
    main()
