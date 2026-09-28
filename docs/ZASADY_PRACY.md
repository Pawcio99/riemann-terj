# Zasady pracy

## Zasady naukowe
- Każde zdanie merytoryczne w raportach oznaczaj statusem: [TWIERDZENIE] (udowodnione w literaturze), [FAKT NUMERYCZNY] (nasz reprodukowalny wynik), [HIPOTEZA] (testowalna), [ANALOGIA TERJ] (interpretacja, nie wynik).
- Nie ogłaszaj dowodu ani przełomu. Wyniki negatywne dokumentuj równie starannie jak pozytywne.
- Każda liczba w raporcie pochodzi ze skryptu w repozytorium, ze stałym ziarnem losowym; raport podaje komendę, która ją odtwarza.
- Nie wymyślaj cytowań, wzorów ani stałych. To, co pochodzi z pamięci, oznacz „DO WERYFIKACJI" i sprawdź w źródle, zanim użyjesz tego w wynikach.
- Dane porównuj z modelem zerowym przy tej samej liczności próby i tym samym estymatorze (np. zera kontra symulowane CUE).
- Raporty `REPORT.md` pisz po polsku, ciągłą prozą w rejestrze akademickim, bez list punktowanych.

## Odtwarzalność
Python działa w `.venv`, a wersje bibliotek są w `requirements.txt`. Kluczowe biblioteki to python-flint (Arb: `flint.acb.zeta_zeros(n, ile)` liczy zera z kontrolą błędu), mpmath, numpy, scipy i matplotlib. Katalog `data/` nie jest wersjonowany; bloki zer odtwarzają `src/zeros.py` i `src/odlyzko.py`. Pełne wyniki trafiają do `results/`.

## Format danych
Blok zer to plik `.npz` z polami `base` (liczba całkowita zapisana jako tekst), `x` (float64, przesunięcia; wysokość = base + x) oraz `start_index` (numer pierwszego zera jako tekst). Odstępy liczymy z różnic `x`, więc precyzja zostaje zachowana także na wysokości 10^22.

## Podstawowe komendy
```bash
python tests/baseline_check.py
python -m src.zeros --start 1 --count 20000 --workers 6 --out data/z_1_20k.npz
python -m src.stats data/z_1_20k.npz --out results/stats_1_20k.json
python -m src.rp_model --out results/rp_scan.json
python -m src.explicit data/z_1_20k.npz --nzeros 500
```
