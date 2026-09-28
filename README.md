# riemann-terj

Numeryczna eksploracja hipotezy Riemanna przez geometrię widmową i teorię macierzy losowych, prowadzona jako niezależny test przewidywań ramy teoretycznej TERJ. Zobacz PRZEDMOWA.md po znaczenie znaczników wiarygodności i aktualny stan przewidywań.

Bilans projektu (co osiągnięto, czego nie wykazano, dokąd dalej): [docs/BILANS.md](docs/BILANS.md).

Autor: Paweł Majsterek, niezależny badacz (Independent researcher), Kopenhaga, Dania. Kontakt: majsterek_pawel@proton.me

## Szybki start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python tests/baseline_check.py

## Dane i odtworzenie wyników

Stan sprawdzony 2026-09-28 w świeżym klonie tego repozytorium (Python 3.14.4, wersje bibliotek jak w `requirements-lock.txt`). Czasy podano tylko tam, gdzie je zmierzono.

**Poziom 1: środowisko.** `python tests/baseline_check.py` przechodzi 6 z 6 kontroli w około 5 s.

**Poziom 2: ponowna analiza z wyników pośrednich w `results/`, bez `data/`.** Poniższe komendy nadpisują pliki w `results/`; sprawdzono, że po ich uruchomieniu pliki są bajt w bajt takie same jak śledzone w repozytorium (`git status` pozostaje pusty). Czasów nie mierzono w czystym środowisku.

    python -m src.test1_summary
    python -m src.test3_summary --out results/test3/summary_A.json
    python -m src.test3_fluctdiss --out results/test3/fluctdiss.json
    python -m src.test3_neff_check --out results/test3/neff_check.json

**Poziom 3: z surowych bloków zer; wymaga katalogu `data/`, którego nie ma w repozytorium.** Sprawdzono tylko blok od zera numer 10^3, tymi wywołaniami:

    python -m src.zeros --start 1000 --count 10000 --workers 6 --out data/z_1e3_10k.npz
    python -m src.test1 --out /tmp/t1_single.json --nrep 200 --nm 1000 --files data/z_1e3_10k.npz

(w teście użyto innej ścieżki tymczasowej; ścieżka wyjściowa jest dowolna, byle poza results/)

Pierwsze trwało 31 s, drugie około 2 min (czas ścienny). Plik wynikowy zapisano poza `results/`, w katalogu tymczasowym; wszystkie 17 pól wiersza `z_1e3_10k.npz` w `results/test1/summary.json` okazały się identyczne co do bitu.

Ograniczenia:

- Bloki od 10^4 do 10^8 nie były odtwarzane w czystym środowisku; komenda dla bloku 10^8 jest w `docs/TESTY.md`. Czasów nie mierzono w czystym środowisku.
- Tabel Odlyzki (n = 10^12, 10^21, 10^22) nie da się odtworzyć według obecnej dokumentacji, bo wartość `--base` dla `src.odlyzko` nie jest zapisana. Adres tabel jest w `docs/TESTY.md`.
- `python -m src.test2_summary` wymaga `data/`; bez niego kończy się błędem `FileNotFoundError` (brak `data/z_1e3_10k.npz`).
- `src.test1` pomija brakujące pliki bez błędu, więc przy niepełnych danych wynik ma mniej bloków. Sprawdź liczbę wierszy w pliku podanym w `--out`, a po `src.test1_summary` pole `n_blocks` w `results/test1/summary.json` (w repozytorium: 9).

## Struktura

- docs/TESTY.md — specyfikacja wykonanych testów
- docs/LITERATURA.md — bibliografia, z rozróżnieniem źródeł sprawdzonych i podanych z pamięci
- docs/POSTEP.md — dziennik postępu
- results/ — pełne wyniki, raporty i wykresy dla każdego testu
- src/ — kod źródłowy

## Licencja

Kod na licencji MIT (LICENSE). Wyniki i raporty w results/ i docs/ na tych samych warunkach.

## Zgłaszanie błędów

Jeśli metoda, kod albo interpretacja gdzieś nie trzyma się kupy, otwórz issue — po to jest to repozytorium publiczne.
