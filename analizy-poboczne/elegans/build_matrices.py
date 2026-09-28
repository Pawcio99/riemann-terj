"""Macierze neuron x neuron (272) z Cook et al. 2019, SI 5 (corrected July 2020).

Uruchom: python build_matrices.py  (w katalogu dane-elegans/; potrzebne numpy, openpyxl)
"""
import json

import numpy as np
import openpyxl

SRC = "SI5_connectome_July2020.xlsx"
DROP = {"PHARYNX", "SEX SPECIFIC"}


def clean(v):
    return v.strip() if isinstance(v, str) else v


def read_sheet(ws):
    """Zwraca (etykiety wierszy, grupy wierszy, etykiety kolumn, macierz) od wiersza 4 / kolumny D."""
    rows = list(ws.iter_rows(values_only=True))
    col_labels = [clean(c) for c in rows[2][3:]]
    while col_labels and col_labels[-1] is None:
        col_labels.pop()
    row_labels, groups, data, g = [], [], [], None
    for r in rows[3:]:
        if r[2] is None:
            continue
        if r[0] is not None:
            g = clean(r[0])
        row_labels.append(clean(r[2]))
        groups.append(g)
        data.append([0 if v is None else v for v in r[3:3 + len(col_labels)]])
    M = np.array(data, dtype=float)
    assert None not in col_labels, "luka w etykietach kolumn"
    return row_labels, groups, col_labels, M


def select(row_labels, col_labels, M, keep):
    for lab_list, what in ((row_labels, "wierszach"), (col_labels, "kolumnach")):
        assert len(set(lab_list)) == len(lab_list), f"duplikaty etykiet w {what}"
        missing = [k for k in keep if k not in lab_list]
        assert not missing, f"brak w {what}: {missing}"
    ri = [row_labels.index(k) for k in keep]
    ci = [col_labels.index(k) for k in keep]
    assert [row_labels[i] for i in ri] == [col_labels[j] for j in ci] == keep, "kolejnosc sie rozjezdza"
    return M[np.ix_(ri, ci)]


def summary(name, W):
    asym = np.argwhere(np.triu(W != W.T, 1))
    s = dict(shape=list(W.shape), nonzero=int(np.count_nonzero(W)),
             asym_pairs_unordered=int(len(asym)), asym_entries_ordered=int(np.count_nonzero(W != W.T)),
             symmetric=bool(np.array_equal(W, W.T)), zero_rows=int((~W.any(axis=1)).sum()),
             zero_cols=int((~W.any(axis=0)).sum()), total_weight=float(W.sum()),
             integer_valued=bool(np.all(W == np.round(W))), negative=int((W < 0).sum()))
    print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in s.items()))
    return s


def main():
    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)

    rl, grp, cl, M = read_sheet(wb["hermaphrodite chemical"])
    counts = {g: grp.count(g) for g in dict.fromkeys(grp)}
    print(f"chemical: wiersze {len(rl)}, kolumny {len(cl)}, macierz {M.shape}; grupy wierszy {counts}")
    assert M.shape == (300, 454), M.shape
    assert counts.get("PHARYNX") == 20 and counts.get("SEX SPECIFIC") == 8, counts
    keep = [lab for lab, g in zip(rl, grp) if g not in DROP]
    print(f"300 - {counts['PHARYNX']} - {counts['SEX SPECIFIC']} = {len(keep)}")
    assert len(keep) == 272
    assert "CANL" not in keep and "CANR" not in keep
    Wc = select(rl, cl, M, keep)

    rl2, grp2, cl2, M2 = read_sheet(wb["hermaphrodite gap jn symmetric"])
    print(f"gap symmetric: wiersze {len(rl2)}, kolumny {len(cl2)}, macierz {M2.shape}")
    print(f"  pelna macierz gap symetryczna: {np.array_equal(M2[:, :len(rl2)], M2[:, :len(rl2)].T) if rl2 == cl2[:len(rl2)] else 'n/d (inna kolejnosc)'}")
    Wg = select(rl2, cl2, M2, keep)

    out = {}
    for name, W, fn in (("chemical", Wc, "herm_chem_272.npy"), ("gap_symmetric", Wg, "herm_gap_sym_272.npy")):
        np.save(fn, W)
        out[name] = dict(file=fn, **summary(name, W))
    with open("labels_272.json", "w") as f:
        json.dump(keep, f)
    with open("summary_272.json", "w") as f:
        json.dump(dict(source=SRC, dropped_groups=sorted(DROP), n=len(keep), matrices=out), f, indent=1)
    print("zapisano: herm_chem_272.npy, herm_gap_sym_272.npy, labels_272.json, summary_272.json")


if __name__ == "__main__":
    main()
