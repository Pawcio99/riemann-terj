# Analizy poboczne

Analizy poza rejestrem przewidywań TERJ. Dane nie są redystrybuowane; źródła: Cook i in. 2019 (WormWiring) oraz zbiór H01 (Shapson-Coe i in.); licencje danych wg wydawców.

## Jak odtworzyć

### C. elegans (`elegans/`)

Wymaga pliku `SI5_connectome_July2020.xlsx` (Cook i in. 2019, WormWiring) w katalogu `elegans/`. Notatki: `NOTATKA.md`, `raport2.md`, `NOTATKA3.md`, `NOTATKA4.md`; prerejestracje `prereg*.md` z sumami SHA-256 w `prereg*.sha256`.

```
cd elegans
python build_matrices.py
python prep_spectral.py
python spectral_gap.py controls
python spectral_gap.py data
python round2_mirror.py
python chem_spectral.py controls
python csr_diag.py
python chem_spectral.py data
python round4_reciprocity.py
```

### H01 (`h01/`)

Skrypty strumieniują pliki synaps z publicznego bucketu H01 i wymagają pliku `somas.csv` z tego samego wydania. Notatka z wynikami: `NOTATKA_H01.md`. Małe pliki wynikowe leżą w `parts/`, `parts2/` i `typecheck.json`. Pierwszy przebieg (`parts/`) powstał z `h01_pairs.py`, a drugi (`parts2/`, z podziałem na klasę miejsca presynaptycznego) z `h01_pairs2.py`. Oba dają te same sumy: 73 745 par i 112 344 rekordy. `h01_pairs.py` jest tu w wersji z łatką. Łatka (`h01_pairs.patch`, wynik `diff` względem wersji pierwotnej) dodaje rzutowanie identyfikatorów i współrzędnych, które w JSON-ach są tekstami, na liczby.

```
cd h01
# przebieg 1: wszystkie klasy -> parts/
python3 h01_pairs.py --selftest --somas somas.csv --head json_head.bin
python3 h01_pairs.py --run   --somas somas.csv --out parts --workers 4
python3 h01_pairs.py --merge --somas somas.csv --out parts --expect 166
python3 h01_components.py --pairs parts/pairs.csv --neurons parts/neurons.csv --empirical --out parts/components_emp.json
python3 h01_structure.py --pairs parts/pairs.csv --neurons parts/neurons.csv --out parts/structure.json
# przebieg 2: klasa miejsca, graf AXON -> parts2/
python3 h01_pairs2.py --selftest --somas somas.csv --head json_head.bin
python3 h01_pairs2.py --run   --somas somas.csv --out parts2 --workers 4
python3 h01_pairs2.py --merge --somas somas.csv --out parts2 --expect 166
python3 h01_components.py --pairs parts2/pairs_axon.csv --neurons parts2/neurons.csv --empirical --out parts2/components_emp.json
python3 h01_structure.py --pairs parts2/pairs_axon.csv --neurons parts2/neurons.csv --logbins 14 --out parts2/structure.json
# kontrola, sprawdzenie klasy miejsca i kierunku, rekordy a miejsca w pliku 0
python3 h01_control.py --pairs-all parts2/pairs_all.csv --n-neurons 15487 --reps 200 --out parts2/control.json
python3 h01_typecheck.py --somas somas.csv --nfiles 4 --out typecheck.json
python3 h01_dircheck.py parts2/pairs_all.csv 2 3 4 6 > parts2/dircheck.txt
python3 h01_recordcount.py --index 0
```
