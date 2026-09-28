# Bilans projektu riemann-terj

Autor: Paweł Majsterek, niezależny badacz (Independent researcher), Kopenhaga, Dania. Kontakt: majsterek_pawel@proton.me

Co zbudowano, co osiągnięto, czego nie udało się wykazać i dokąd dalej.

Obowiązują te same znaczniki co w [PRZEDMOWA.md](../PRZEDMOWA.md): `[TWIERDZENIE]` (wynik z literatury), `[FAKT NUMERYCZNY]` (nasz odtwarzalny wynik, ze wskazanym plikiem), `[HIPOTEZA]` (zdanie testowalne, także obalone), `[ANALOGIA TERJ]` (interpretacja, nie dowód) oraz `[NIEZNANE]` (czego nie wiemy). Ten dokument nie zmienia tabeli przewidywań w PRZEDMOWA.md.

**W jednym akapicie.** Nie udowodniliśmy ani nie obaliliśmy hipotezy Riemanna. Trzy testy statystyczne na zerach funkcji dzeta nie wykazały żadnego odstępstwa od tego, co przewiduje literatura fizyki matematycznej, a przewidywanie, że w statystyce zer pozostaje domieszka symetrii odwrócenia czasu (GOE), nie znalazło potwierdzenia w zbadanym zakresie. Modele, z którymi porównywaliśmy dane, pochodzą z literatury (Bogomolny i in., Csordas–Smith–Varga); model Ĥ + λĜ, inspirowany językiem TERJ, jest zaimplementowany, ale jego skan (Test 2b) nie został wykonany. Wynik jest więc negatywny lub ograniczony rozdzielczością danych tam, gdzie stawialiśmy testowalne pytania, i nie daje potwierdzenia TERJ.

## 1. Możliwe kierunki dalszych badań

| # | Kierunek | Co by rozstrzygnęło | Koszt | Główne ryzyko | Ocena |
|---|---|---|---|---|---|
| 1 | **Test 2b**: skan (N, λ) modelu Ĥ + λĜ z kryterium zapisanym przed liczeniem | Jedyne miejsce, w którym aparat TERJ dałby własne przewidywanie ilościowe: czy istnieje λ zgodne z efektywnym rozmiarem N̂ z Testu 2 | średni | Wolne parametry: bez prerejestracji łatwo o dopasowanie a posteriori | Najważniejszy dla samego TERJ |
| 2 | **Lepszy model zerowy dla niskich wysokości** | Czy odrzucenie CUE(N_eff) przy N_eff < 4 wynika ze znanych poprawek arytmetycznych (poprawki od liczb pierwszych do korelacji zer; wzory do sprawdzenia w źródle, nie podaję ich z pamięci) | średni | Wynik może okazać się odtworzeniem znanej fizyki | Domyka największą niewiadomą Testów 1 i 2 |
| 3 | **Mechanistyczny predyktor czasu zderzenia**: układ równań ruchu zer z K najbliższymi sąsiadami zamiast dopasowania β = 0,68 | Czy β < 1 wynika z ruchu sąsiadów; jak błąd maleje z K | niski do średniego | Równanie ma nieskończoną sumę; zbieżność w K może być wolna | Tani i falsyfikowalny |
| 4 | Większa próba par w teście śladu arytmetycznego | Obecne n = 36 daje niską moc; nie wiemy, czy brak sygnału to brak efektu | wysoki (koszt rośnie z wysokością pary) | Bez punktu 3 rośnie tylko koszt | Dopiero po punkcie 3 |
| 5 | **Wizualizacje z prawdziwych danych** na stronie (histogram odstępów na tle GUE, trend N̂/N_eff, oś czasu zderzeń) | Komunikacja, nie nauka | niski | — | Zrobić najpierw |
| 6 | **Powtarzalność i cytowalność**: przypięte wersje bibliotek, `CITATION.cff`, archiwizacja z DOI | Ułatwia weryfikację przez innych | niski | Wymaga kilku działań w GitHubie | Zrobić |
| 7 | Uniwersalność widma sieci nerwowych (replikacja na samcu C. elegans; większe konektomy) | Czy odstępstwa opisowe z analiz pobocznych się powtarzają | wysoki | Prawdopodobnie brak różnicy względem modelu zerowego; nie dotyczy Riemanna | Poza głównym torem |
| 8 | Model zerowy z osadzeniem przestrzennym dla H01 | Czy widmo ludzkiego konektomu różni się od takiego modelu | bardzo wysoki | Obcięte aksony i niejasna wiarygodność klas miejsc | Nie zalecamy |
| 9 | Gałąź „metryka generowana przez H” (freestyle) | Wymagałaby dwuwymiarowego pola H(t, x) na siatce | wysoki | Brak danych; poprzednie próby nierozstrzygnięte | Zamknięta do nowego pomysłu |

Czego nie robić: nie wpisywać wyników analiz pobocznych do tabeli przewidywań TERJ, bo nie były przewidywaniami TERJ; nie interpretować zgodności ze statystyką GUE jako czegoś specyficznego dla TERJ ani dla funkcji dzeta, bo uniwersalność tej statystyki jest własnością bardzo wielu układów.

## 2. Co zbudowano: program i modele

Repozytorium zawiera wznawialny, kontrolowany potok obliczeniowy w Pythonie (biblioteki: python-flint z arytmetyką interwałową Arb, mpmath, numpy, scipy, matplotlib). Skrypty w `src/`: obliczanie zer (`zeros.py`, `odlyzko.py`), statystyka odstępów z bootstrapem (`stats.py`), zespoły macierzy losowych CUE, GUE, GOE i model przejścia (`rmt.py`, `rp_model.py`), formuła jawna i „spektroskopia liczb pierwszych” (`explicit.py`), Testy 1–3 z podsumowaniami (`test1.py`, `test2.py`, `test3*.py`) oraz silnik przepływu ciepła de Bruijna–Newmana z kontrolą precyzji (`heatflow.py`). Kontrola podstawowa: `python tests/baseline_check.py`.

Trzy modele, z jasnym statusem:

- **CUE(N_eff).** `[TWIERDZENIE]` z literatury (Bogomolny i in. 2006): lokalna statystyka zer na wysokości T odpowiada macierzom CUE o efektywnym rozmiarze N_eff = ln(T/2π)/√(12Λ). To przewidywanie nie pochodzi z TERJ.
- **Predyktor czasu zderzenia z lokalnego pola pływowego.** `[FAKT NUMERYCZNY]` (Test 3): dopasowanie ze stałym polem pływowym T daje zależność z β = 0,68 zamiast oczekiwanego 1; model nie przewiduje 6 z 42 zderzeń. Jest to model empiryczny, nie wyprowadzony.
- **Ĥ + λĜ** (`src/rmt.py`, `src/rp_model.py`). `[ANALOGIA TERJ]`: zaimplementowany, nieużyty w Testach 1–3; skan (N, λ) odłożono do Testu 2b (patrz REPORT Testu 2).

## 3. Co osiągnięto

Wszystkie liczby pochodzą z raportów w `results/` (pliki wskazane w kolumnie).

| Test | Wynik | Status | Plik |
|---|---|---|---|
| 1 | Statystyka odstępów w dziewięciu blokach (10³ do 10²²) zgodna z CUE(N_eff): odchylenie max \|z\| = 2,27, brak domieszki GOE (iloraz wiarygodności p od 0,19 do 1,0). Od bloku 10¹² p_KS w przedziale 0,22 do 0,88. Dla bloków 10³ do 10⁸ (N_eff < 4) test KS odrzuca CUE(N_eff) | `[FAKT NUMERYCZNY]`; wynik negatywny: brak domieszki GOE w zbadanym zakresie | `results/test1/REPORT.md` |
| 2 | Iloraz N̂/N_eff spada od 2,26 do 1,00, w przybliżeniu monotonicznie (w środku zakresu z niewielkim wzrostem, np. 1,59, 1,46, 1,63, 1,66 dla bloków 10⁶, 10⁷, 10⁸ i 10¹²); nie rozstrzygnięto, czy efektywny rozmiar rośnie logarytmicznie (wyraz wolny −0,008 ± 0,019; nachylenie 0,43 ± 0,04 wobec przewidywanych 0,230, nieinterpretowalne) | `[FAKT NUMERYCZNY]`, ograniczony rozdzielczością danych | `results/test2/REPORT.md` |
| 3A | 166 śledzonych par zer w przepływie ciepła, 53 zderzenia; pole pływowe T < 0 dla 42 z 42 par; brak naruszeń niezmiennika t_c/t_c⁰ ≥ 1 w zbadanym zakresie | `[FAKT NUMERYCZNY]` | `results/test3/REPORT_A.md`, `REPORT_B.md` |
| 3B | Brak sygnału arytmetycznego (ζ′, faza Grama, znak prawa Grama) ponad geometrię lokalną przy n = 36 | `[FAKT NUMERYCZNY]`, niska moc | `results/test3/REPORT_B.md` |

## 4. Czego nie udało się wykazać

- **Domieszka symetrii odwrócenia czasu w statystyce zer:** wynik negatywny w zbadanym zakresie wysokości (Test 1), bez śladu ponad model zerowy; nie wykluczamy jej przy innych wysokościach ani innych statystykach.
- **Hipoteza fluktuacyjno-dyssypacyjna** (silniejsze tłumienie = mniejszy rozrzut reszt): odrzucona; surowa korelacja ρ = 0,578 (p = 0,0002, n = 36) ma znak przeciwny do przewidywanego, a po kontroli na δ spada do 0,044. `[FAKT NUMERYCZNY]`, `results/test3/REPORT_B.md`.
- **Zamknięcie zaobserwowanej niezgodności modelu zerowego przy niskich wysokościach:** hipoteza, że to artefakt małego N, nie została potwierdzona.
- **Model stałego pola pływowego** nie tłumaczy ilościowo czasów zderzeń (β = 0,68, 6 z 42 par nieprzewidzianych).
- **Gałąź „metryka generowana przez H”:** wyprowadzono wzór na naruszenie warunku energetycznego, zweryfikowany trzema niezależnymi kontrolami, ale wynik pozostał nierozstrzygnięty (resztkowy trend z wysokością, Spearman ρ = 0,334, p = 0,023 na 46 parach, którego nie umiano rozdzielić od prawdziwego sygnału). Liczby nie są odtwarzalne z tego repozytorium.
- **Hipoteza Riemanna:** nie wnosimy nic do jej rozstrzygnięcia. Znane ograniczenia stałej de Bruijna–Newmana (0 ≤ Λ ≤ 0,2, Rodgers–Tao i Platt–Trudgian) pozostają bez zmian.

## 5. Analizy poboczne (poza rejestrem przewidywań TERJ)

Te analizy nie były przewidywaniami TERJ. Kod i notatki: katalog `analizy-poboczne/`; dane nie są redystrybuowane (licencje wg wydawców). Skrypty H01 pobierają pliki synaps strumieniowo z publicznego zasobu `h01-release` (Google Cloud Storage), ale plik `somas.csv` z tego samego wydania trzeba pobrać osobno (<https://storage.googleapis.com/h01-release/data/20210601/c3/tables/somas.csv>); dla C. elegans plik źródłowy `SI5_connectome_July2020.xlsx` (Cook i in. 2019, WormWiring) trzeba pobrać ręcznie (<https://wormwiring.org/si/SI%205%20Connectome%20adjacency%20matrices%2C%20corrected%20July%202020.xlsx>). Oba adresy sprawdzono 2026-09-28 (HTTP 200, rozmiary zgodne z plikami użytymi w analizach).

**Konektom C. elegans (Cook i in. 2019).** Dla rdzenia złączy szczelinowych (266 neuronów) średni stosunek kolejnych odstępów r̃ = 0,5079 wobec 0,5150 ± 0,0190 w grafach o tych samych stopniach (z = −0,37, p = 0,70; spectral_gap.json): brak wyróżnienia; klasa to GOE, nie GUE, jak u zer dzety. Dla macierzy synaps chemicznych statystyka zespolonego stosunku odstępów ⟨cos θ⟩ = −0,1886 wobec −0,1220 ± 0,0640 (z = −1,04; chem_spectral.json): brak wyróżnienia. Liczba wartości własnych rzeczywistych (40 wobec 26,4 ± 3,5; chem_spectral.json) po dodaniu wzajemności połączeń do modelu zerowego spadła do z = +2,22 (40 wobec 31,3 ± 3,9; round4_reciprocity.json); hipoteza z prerejestracji nie została potwierdzona, a resztę nadmiaru zostawiamy jako `[HIPOTEZA]` do replikacji. Symetria lustrzana L/R nie była testowana w modelu zerowym.

**Fragment kory człowieka H01.** Graf 15 487 neuronów, 73 745 par (112 344 rekordy; summary.json z przebiegu parts). W porównaniu z niezależnie opublikowaną analizą tego samego wydania (A. Salova, I. A. Kovács, „Combined topological and spatial constraints are required to capture the structure of neural connectomes”, arXiv:2405.06110, 2024: 15 730 neuronów z jedną duszą, 115 165 synaps) liczby zbliżone (15 487 wobec 15 730, o 1,5% mniej; 112 344 wobec 115 165, o 2,4% mniej); definicje zbiorów mogą się różnić. Największa składowa silnie spójna: 8 930 dla wszystkich klas, 2 739 dla samych krawędzi z presynapta klasy AXON (components_emp.json; losowe podzbiory tej samej wielkości: 4 448 [4 303; 4 582], control.json). Ponad połowa rekordów w parach (54,5%, summary.json z przebiegu parts2) ma presynaptę klasy DENDRITE. `[FAKT NUMERYCZNY]`: w żadnej z 10 345 par z co najmniej dwoma takimi rekordami nie ma rekordów w obu kierunkach (dircheck.txt); hipoteza o dowolnym kierunku tych krawędzi została obalona. `[HIPOTEZA]`: w rekordach klasy AXON typ rekordu koduje pobudzenie lub hamowanie (79% rekordów piramidalnych z typem „2”, 85% rekordów interneuronów z typem „1”; structure.json z przebiegu parts2); dokumentacja zbioru potwierdza istnienie klasyfikacji E/I, ale nie definiuje pola. `[NIEZNANE]`: znaczenie klasy DENDRITE w miejscu presynaptycznym, jednostki współrzędnych, rozbieżność liczby rekordów (166 216 068, summary.json) z liczbą synaps w dokumentacji (130 milionów, explore.html). Widma nie liczyliśmy i nie zalecamy tego bez modelu zerowego z osadzeniem przestrzennym.

Wniosek z obu: przy tych rozmiarach i przy tych modelach zerowych widma sieci nerwowych nie różnią się istotnie od grafów o tych samych stopniach; nie ma podstaw do łączenia ich z zerami dzety.

## 6. Co może się przydać innym

- **Potok zer z kontrolą indeksu i wznawianiem** (Arb), wraz z konwersją tabel Odlyzki do jednego formatu.
- **Silnik przepływu ciepła de Bruijna–Newmana** z kontrolą stabilności (podwojona precyzja, połowa kroku) i niezależną kontrolą liczby zer przez zliczanie zmian znaku.
- **Tabela CUE(N)** interpolowana w 1/N² oraz narzędzia do estymacji efektywnego rozmiaru.
- **Narzędzie zespolonego stosunku odstępów** z kalibracją na symulacjach Ginibre i Poissona.
- **Potok strumieniowego przetwarzania dużych plików JSON** (126 GB, bez zapisu na dysk) z wznawianiem.
- **Zasady pracy:** prerejestracja przed liczeniem, model zerowy, kontrola tożsamości, znaczniki wiarygodności.

Pułapki, na które natrafiliśmy: rzutowanie bardzo małych wielkości na `float64` gubi informację bez błędu; identyfikatory w danych bywają tekstami; efekty brzegowe zawyżają statystyki zespolone przy małej liczbie punktów; korelacja surowa może być cieniem zmiennej ukrytej (tu δ); model zerowy zachowujący tylko stopnie nie zachowuje wzajemności połączeń; wynik zgodny z przewidywaniem bywa zbyt słaby, żeby cokolwiek dowieść, dlatego potrzebna jest kontrola.

## 7. Zaproszenie

Zaprasza się do sprawdzenia metod, kodu i interpretacji, zwłaszcza osoby zajmujące się teorią macierzy losowych, analityczną teorią liczb, chaosem kwantowym i analizą sieci. Najbardziej wartościowe uwagi to: wskazanie błędu w modelu zerowym, lepszy model dla niskich wysokości (punkt 2 w sekcji 1), pomysł na zapisane z góry przewidywanie dla Testu 2b oraz replikacje. Zgłoszenia jako *issue* w repozytorium. Kod na licencji MIT, dokumenty na tych samych warunkach.
