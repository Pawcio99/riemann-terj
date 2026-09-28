Analiza poza rejestrem przewidywań TERJ; opis danych, nie test hipotezy.

# H01: graf połączeń między neuronami z duszą (c3), wersja ostateczna

**1. Dane i potok.** Źródłem jest wydanie H01 z 20210601, segmentacja c3. Zbiór roboczy to 15 487 neuronów, czyli segmenty c3 z dokładnie jedną duszą z somas.csv. Przeczytaliśmy strumieniowo 166 plików JSON: 126 066 253 429 bajtów i 166 216 068 rekordów (par miejsc pre- i postsynaptycznych), bez rekordów błędnych. Presynaptę w zbiorze ma 642 274 rekordów, postsynaptę 26 773 572, a obie strony 112 344. Pierwszy przebieg (h01_pairs.py, wyniki w parts/) objął wszystkie rekordy z obiema stronami w zbiorze i dał graf wszystkich klas. Pierwotna wersja tego skryptu zakładała identyfikatory liczbowe, a w JSON-ach są one tekstami; łatka to poprawia i jest zapisana w h01_pairs.patch. Drugi przebieg (h01_pairs2.py, wyniki w parts2/) zapisuje dodatkowo klasę miejsca presynaptycznego. Z jego wyników zbudowaliśmy graf AXON (pairs_axon.csv), złożony tylko z rekordów, w których ta klasa to AXON. Oba grafy podajemy równorzędnie. Składowe liczy h01_components.py z empirycznymi zasięgami obwiedni synaps, strukturę i p(d) liczy h01_structure.py, kontrolę h01_control.py, a spójność kierunku h01_dircheck.py.
Dokumentacja: strona explore zbioru (dane-h01/explore.html) podaje, że 130 milionów synaps wykryto automatycznie. Podaje też, że synapsy można podzielić na pobudzające i hamujące według tego, czy rozmiary miejsc pre- i postsynaptycznych są symetryczne, czy asymetryczne. W trzech pobranych stronach (explore, landing, proofreading) nie znaleźliśmy opisu pola type w JSON.

**2. Klasa miejsca presynaptycznego.** Spośród 112 344 rekordów w parach neuronów, z których każdy odpowiada parze miejsc pre- i postsynaptycznych (w pliku 0 praktycznie jeden do jednego), klasę AXON ma 39 753, DENDRITE 61 282, SOMA 7 000, UNKNOWN 4 034, a inne klasy 275. W czterech równomiernie rozłożonych plikach JSON (typecheck.json, 4 002 076 rekordów) klasa DENDRITE to 2,6% miejsc presynaptycznych w całej populacji i 61,6% w rekordach z presynaptą w zbiorze neuronów.

**3. Dwa grafy.** Oba grafy mają 15 487 węzłów. Kontrola polegała na 200 losowych podzbiorach po 29 645 par wylosowanych z grafu wszystkich klas. W ostatniej kolumnie podajemy medianę i przedział 2,5–97,5 percentyla.

| | graf wszystkich klas | graf AXON | kontrola: losowe podzbiory |
|---|---|---|---|
| pary | 73 745 | 29 645 | 29 645 |
| rekordy, z których każdy odpowiada parze miejsc pre- i postsynaptycznych (w pliku 0 praktycznie jeden do jednego) | 112 344 | 39 753 | — |
| pary neuronów z 1 / 2 / 3+ takimi rekordami (wagi par) | 56 012 / 10 277 / 7 456 | 24 278 / 3 387 / 1 980 | — |
| największa SCC | 8 930 | 2 739 | 4 451 (4 303–4 582) |
| największa WCC | 13 162 | 9 482 | 11 232 (11 169–11 323) |
| pary wzajemne | 1 030 | 206 | 166 (144–189) |
| neurony z krawędzią wychodzącą | 73,1% | 37,0% | 56,9% (56,4–57,3%) |
| neurony z krawędzią wchodzącą | 77,1% | 53,8% | 60,6% (60,2–61,0%) |

Pary spoza grafu AXON (44 100) mają największą SCC 3 780, największą WCC 12 561 i 290 par wzajemnych. Średni stopień wynosi 4,76 w grafie wszystkich klas i 1,91 w grafie AXON. Wśród niezerowych stopni wychodzących mediana wynosi 4 i 3, a maksimum 123 i 58.

**4. Zależność p(d).** Odległość liczymy w płaszczyźnie xy, w jednostkach siatki. W grafie AXON wyniki są w 15 przedziałach logarytmicznych (15 252 neurony, 29 365 par). p maleje monotonicznie: 8,1·10⁻³ dla d < 1 255 (188 par), 3,95·10⁻³ w przedziale 6 575–9 947, 6,9·10⁻⁴ w 22 770–34 451, 4,05·10⁻⁵ w 52 123–78 861, 1,7·10⁻⁶ w 119 314–180 519 i 2,5·10⁻⁷ w 180 519–273 120 (10 par). Powyżej nie ma par. W grafie wszystkich klas parts/structure.json podaje p w 10 przedziałach kwantylowych, więc przedziały obu grafów się nie pokrywają. Wartości wynoszą 2,87·10⁻³ dla d < 44 086 (66 688 par), 1,7·10⁻⁴ w 44 086–66 635, 7,3·10⁻⁶ w 118 002–135 155 i 8,6·10⁻⁸ w 207 476–412 237 (2 pary).

**5. Etykiety.**
[FAKT NUMERYCZNY] Wszystkie liczby w punktach 1–4 i 6 pochodzą z wymienionych plików w parts2/, parts/ i z typecheck.json.
[HIPOTEZA] Udział neuronów z krawędzią wychodzącą zależy od typu komórki odwrotnie, niż można by oczekiwać. W grafie AXON wynosi on 48,0% dla piramidalnych i 24,9% dla interneuronów. Wyjaśnienie jest nieznane.
[HIPOTEZA] Wśród rekordów z presynaptą AXON typ rekordu „2” stanowi 79% rekordów komórek piramidalnych (22 965 z 29 042), 79% spiny atypical (1 477 z 1 865) i 70% spiny stellate (158 z 226). Typ „1” stanowi 85% rekordów interneuronów (6 960 z 8 199). Jest to zgodne z odwzorowaniem „2 = pobudzające, 1 = hamujące”. Dokumentacja potwierdza, że istnieje klasyfikacja pobudzające/hamujące na poziomie synapsy, ale nie mówi, że zapisuje ją pole type, ani w którą stronę idzie odwzorowanie. Przy tym odwzorowaniu 21% rekordów piramidalnych i 15% rekordów interneuronów miałoby typ przeciwny.
[FAKT NUMERYCZNY] W rekordach spoza AXON udział typu „1” wynosi 72–95% niezależnie od typu komórki presynaptycznej. Liczyliśmy to przez odjęcie parts2/structure.json od parts/structure.json.
[FAKT NUMERYCZNY] Na grafie wszystkich klas (parts/structure.json) podział typów był niewidoczny: typ „1” miało 66% rekordów piramidalnych i 84% rekordów interneuronów. Ponad połowę rekordów stanowiły tam krawędzie DENDRITE, w większości typu „1”.
[FAKT NUMERYCZNY] Wynik h01_recordcount.py. Plik export000000000000.json ma 998 235 rekordów, 998 180 różnych identyfikatorów miejsc presynaptycznych (55 występuje w dwóch rekordach), 998 150 różnych identyfikatorów postsynaptycznych (85 w dwóch) i 998 235 różnych par (pre id, post id). Jeden rekord odpowiada więc prawie dokładnie jednej parze miejsc.
[NIEZNANE] Strona dokumentacji podaje 130 milionów synaps, a w pobranych plikach jest 166 216 068 rekordów. Nie tłumaczy tego to, że jedno miejsce może mieć kilku partnerów (sprawdziliśmy to na jednym pliku ze 166). Przyczyna rozbieżności jest nieznana.
[NIEZNANE] Nie znamy wiarygodności klasy miejsca presynaptycznego ani tego, co znaczy DENDRITE w miejscu presynaptycznym. Możliwe odczytania to nieprawidłowa etykieta klasy, fałszywe detekcje albo prawdziwe kontakty dendryt–dendryt.
[NIEZNANE] Nie znamy potwierdzonego znaczenia pola type na poziomie rekordu ani znaczenia klas UNKNOWN i SOMA w miejscu presynaptycznym. Nie znamy też jednostek współrzędnych ani rozmiaru woksela ani jakości rekonstrukcji aksonów.

**6. Co się zmieniło po sprawdzeniu klasy miejsca.** Pierwsza wersja grafu nie rozróżniała klasy miejsca presynaptycznego. Przewidywaliśmy, że po odfiltrowaniu największa SCC będzie mniejsza niż połowa z 8 930. Potwierdziło się to (2 739), ale przewidywanie było zbyt słabe, bo samo usunięcie 60% krawędzi mogło dać taki wynik. Dlatego dodaliśmy kontrolę z losowymi podzbiorami o tej samej liczbie par (parts2/control.json, 200 powtórzeń). Losowe podzbiory 29 645 par dają największą SCC 4 448 [4 303; 4 582], a graf AXON ma 2 739. Udział neuronów z krawędzią wychodzącą wynosi 37,0% wobec 56,9%, a pary wzajemne 206 wobec 166 [144; 189]. Kontrola nie rozstrzyga różnicy między grafami; nie wiemy, dlaczego graf AXON jest mniej spójny niż losowy podzbiór.
[FAKT NUMERYCZNY] Spójność kierunku sprawdziliśmy w parts2/dircheck.txt na parach nieuporządkowanych o co najmniej n rekordach danej klasy. Wartość oczekiwana przy dowolnym kierunku to suma 1 − 2^(1−n) po parach.

| klasa | n ≥ | pary nieuporządkowane | rekordy w obu kierunkach | oczekiwane przy dowolnym kierunku |
|---|---|---|---|---|
| dendrytyczna | 2 | 10 345 | 0 (0,0%) | 64,2% |
| dendrytyczna | 3 | 4 323 | 0 (0,0%) | 84,0% |
| dendrytyczna | 4 | 2 199 | 0 (0,0%) | 92,7% |
| dendrytyczna | 6 | 773 | 0 (0,0%) | 98,5% |
| AXON | 2 | 5 461 | 206 (3,8%) | 62,6% |
| AXON | 3 | 2 025 | 97 (4,8%) | 84,1% |
| AXON | 4 | 1 027 | 50 (4,9%) | 92,9% |
| AXON | 6 | 375 | 22 (5,9%) | 98,4% |

Wniosek: kierunek jest spójny na poziomie pary neuronów w obu klasach; wcześniejsza hipoteza o dowolnym kierunku została obalona. 206 par dwukierunkowych klasy AXON zgadza się z liczbą par wzajemnych w grafie AXON (components_emp.json).

**7. Ograniczenia.** Graf jest dolną granicą połączeń, bo aksony są obcięte albo odłączone od dusz: presynaptę w zbiorze ma 642 274 rekordów, a postsynaptę 26 773 572. Obwiednia synaps neuronu tylko w przybliżeniu pokazuje, gdzie neuryt został obcięty. Nie liczyliśmy widma grafu i nie zalecamy tego bez modelu zerowego z osadzeniem przestrzennym.

**8. Odtworzenie** (w ~/projekty/h01):
```
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
Opcja --logbins N daje N+1 przedziałów (N krawędzi i pierwszy przedział od zera); 14 daje 15 przedziałów.
