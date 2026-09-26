# Test 1: domieszka symetrii odwrócenia czasu w statystyce zer ζ

Odtworzenie liczb: `python -m src.test1 --out results/test1.json --nrep 200 --nm 1000` (log w `logs/test1.log`), następnie `python -m src.test1_summary`, które tworzy `results/test1/summary.json` i `results/test1/fig_a_vs_T.png` bez ponownego liczenia. Ziarna są stałe w kodzie `src/test1.py`.

## Pytanie i dane

[HIPOTEZA] Statystyka zer funkcji ζ na rosnących wysokościach zawiera składnik klasy GOE (symetria odwrócenia czasu) ponad znane poprawki skończonej wysokości; oczekiwanie standardowe brzmi: brak takiej domieszki. Analizowano dziewięć bloków po 9 999 odstępów, zaczynających się od zera o numerze 10^3, 10^4, 10^5, 10^6, 10^7 i 10^8 (obliczone przez `src.zeros`) oraz od 10^12, 10^21 i 10^22 (tabele Odlyzki przekonwertowane przez `src.odlyzko`). Wysokości T obejmują zakres od około 6,3·10^3 do 1,4·10^21.

## Model zerowy

[TWIERDZENIE] Bogomolny, Bohigas, Leboeuf i Monastra (2006, arXiv math/0602270) wykazali, że korelacje zer na skończonej wysokości opisuje z dobrym przybliżeniem CUE o efektywnym rozmiarze N_eff = ln(T/2π)/√(12Λ), gdzie Λ = 1,57314. Wzór i stałą zweryfikowano w artykule (wzór 19); przykład z artykułu daje 11,30, zgodnie z wzorem. [FAKT NUMERYCZNY] Dla naszych bloków N_eff rośnie od 1,59 (n = 10^3) przez 3,62 (n = 10^8) do 5,63 (10^12) i 10,78 (10^22). Dla każdego bloku wykładnik odpychania a estymowano tym samym estymatorem (MLE dla s < 0,25) na danych i na 200 replikach CUE(N_eff) o tej samej liczności, a odchylenie wyrażono jako z = (a − ⟨a⟩_null)/σ_null. Model alternatywny stanowi przejście GOE→GUE z parametrem α dopasowywanym na siatce; istotność ilorazu wiarygodności (p_LR) liczono z symulacji.

## Wykładnik odpychania

[FAKT NUMERYCZNY] We wszystkich dziewięciu blokach i dla obu rozwinięć widma (gęstościowego i lokalnego) odchylenie |z| nie przekracza 2,27, a nie ma ani jednego przypadku powyżej 3σ. Największe odchylenia dotyczą najniższego bloku (a = 3,55 wobec 2,98 ± 0,25, z = +2,27 dla rozwinięcia gęstościowego oraz z = +2,04 dla lokalnego) i bloku od 10^7 (a = 3,35, z = +1,76 i +1,60). Wszystkie z są dodatnie lub bliskie zera, czyli wykładnik zera jest co najwyżej nieco większy, a nie mniejszy od wartości modelu zerowego. Domieszka GOE obniżałaby a w stronę 2, więc kierunek odchyleń jest przeciwny do sygnału poszukiwanego. Na wykresie `fig_a_vs_T.png` (wykładnik a dla obu rozwinięć na tle pasma ±2σ modelu zerowego, oś ln T) poza pasmem ±2σ leżą wyłącznie oba punkty bloku 10^3 (z = +2,27 i +2,04, czyli tuż powyżej 2σ i poniżej progu 3σ), pozostałe mieszczą się w paśmie; oba rozwinięcia dają praktycznie identyczny przebieg.

## Iloraz wiarygodności GOE→GUE

[FAKT NUMERYCZNY] Skończone α nie jest istotnie lepsze od czystego modelu unitarnego w żadnym bloku: najmniejsza p-wartość ilorazu wiarygodności wynosi 0,19, a mieści się w przedziale od 0,19 do 1,0 dla obu rozwinięć. Dopasowane α̂ leży zwykle na dolnej krawędzi siatki (0,3–0,5) lub w punkcie 1,0, bez trendu wysokościowego. Kryterium anomalii z `docs/TESTY.md` (odchylenie powyżej 3σ spójnie na co najmniej trzech wysokościach i odporne na zmianę rozwijania) nie jest spełnione. [FAKT NUMERYCZNY] Wynik jest zatem negatywny: nie znaleziono śladu domieszki symetrii odwrócenia czasu w zakresie wysokości do 10^21.

## Test KS a adekwatność modelu zerowego

[FAKT NUMERYCZNY] Test Kołmogorowa–Smirnowa całego rozkładu odstępów odrzuca CUE(N_eff) dla bloków od n = 10^3 do n = 10^8 (N_eff < 4): p_KS wynosi 0,005 (co odpowiada dolnej granicy 1/201 przy 200 replikach, więc jest to w istocie nierówność p_KS ≤ 0,005) dla bloków od 10^3 do 10^7 (w obu rozwinięciach) oraz dla rozwinięcia lokalnego bloku 10^8, a 0,010 dla rozwinięcia gęstościowego bloku 10^8. Od n = 10^12 (N_eff = 5,63) p_KS mieści się w przedziale 0,22–0,88, zatem model zerowy nie jest odrzucany. Ponieważ przy z ≲ 2,3 wykładnik odpychania nie wykazuje odchylenia, rozbieżność dotyczy kształtu całego rozkładu, a nie jego ogona małych s.

[HIPOTEZA] Odrzucenie CUE(N_eff) przy N_eff < 4 wynika prawdopodobnie z dosłownego użycia macierzy CUE rozmiaru 2–3, którego rozkład odstępów różni się od granicznego bardziej, niż uwzględnia to model N_eff, a nie z domieszki GOE. Za tą interpretacją przemawia to, że iloraz wiarygodności GOE→GUE w tych blokach nie wykazuje żadnej istotności i że odchylenia a nie mają kierunku GOE. Hipotezy tej nie rozstrzyga Test 1; rozstrzygnie ją Test 2 (efektywny wymiar modelu Ĥ + λĜ), gdzie można porównać zera z modelem zerowym o ciągle regulowanym, a nie całkowitym rozmiarze. Do tego czasu wnioski z bloków o N_eff < 4 ograniczamy do stwierdzenia, że w nich model zerowy jest nieadekwatny jako całość, a nie do żadnej interpretacji fizycznej.

## Interpretacja w ramie TERJ

[ANALOGIA TERJ] W słowniku `docs/TERJ.md` domieszka GOE odpowiadałaby niezerowemu wkładowi operatora odwrócenia czasu τ̂ do statystyki widma. Nasz wynik negatywny oznacza, że przy obecnej czułości (σ_null ≈ 0,22–0,28 dla a przy 10^4 odstępów) taki wkład nie jest widoczny, co nie wyklucza wkładu mniejszego niż ta czułość. Nie stanowi to argumentu ani za RH, ani przeciw niej.

## Ograniczenia

Czułość na małą domieszkę GOE jest ograniczona liczbą odstępów: przy około 10^4 odstępów niepewność a przy modelu zerowym wynosi około 0,25, więc przejście GOE→GUE ze skromnym α pozostaje nierozróżnialne od czystego GUE. Najmniejsza rozróżnialna p-wartość symulacji (1/201) ogranicza opis odrzuceń KS do nierówności. Dziewięć bloków nie jest niezależnymi próbami w sensie gęstej siatki wysokości, a porównania wielokrotne nie były korygowane, co jest konserwatywne wobec wyniku negatywnego.
