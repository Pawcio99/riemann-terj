# riemann-terj

Numeryczna eksploracja hipotezy Riemanna przez geometrię widmową i teorię macierzy losowych, prowadzona jako niezależny test przewidywań ramy teoretycznej TERJ. Zobacz PRZEDMOWA.md po znaczenie znaczników wiarygodności i aktualny stan przewidywań.

Bilans projektu (co osiągnięto, czego nie wykazano, dokąd dalej): [docs/BILANS.md](docs/BILANS.md).

Autor: Paweł Majsterek, niezależny badacz (Independent researcher), Kopenhaga, Dania. Kontakt: majsterek_pawel@proton.me

## Szybki start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python tests/baseline_check.py

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
