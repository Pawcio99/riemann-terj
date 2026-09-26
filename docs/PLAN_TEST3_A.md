# Test 3, faza A: przepływ ciepła H_t i czasy zderzeń t_c (plan, bez uruchamiania)

## Kontekst
Poprawki Testu 2 są już zrobione. `src/test2_summary.py` zapisuje teraz w `fits.json` wartości `p_gof` i `gof_ref_N` przeliczone przy N̂ (przebieg w tle zakończony, fits.json zgadza się z summary.json). W REPORT.md wyjaśniono różnicę z-score: czat używał SD wariancji dla dużego N (0,0027), a przy CUE(2) SD wynosi ok. 0,0021. Poprawiono też autorów arXiv:1608.04638 i dodano informację o podpisie rys. 3 (N ≈ 11,3).

Na początku fazy A domknę trzy drobne sprawy. W REPORT.md zdjęcie znacznika [FAKT NUMERYCZNY] ze zdania „nie porównywano wariancji”, bo to zdanie nie jest wynikiem. W POSTEP.md wpis o poprawkach Testu 2 zastąpi nieaktualne „Do weryfikacji: arXiv:1608.04638”.

Cel fazy A: wiarygodny silnik H_t, walidacja na H_0 oraz pomiar czasów zderzeń t_c sąsiednich par dla pierwszych około 100 zer. Każdy t_c jest porównany z przybliżeniem izolowanej pary. Fazy B nie planuję.

## Nowe pliki
- `src/heatflow.py`: silnik H_t oraz pochodne ∂_x, ∂_x², ∂_t, bez I/O.
- `src/test3.py` z podkomendami `validate | pilot | tc`, na wzór `src/test2.py`. Dla `tc` używa `multiprocessing.Pool(≤6)`, jak `src/zeros.py`.
- Wyniki trafiają do `results/test3/*.json`, logi do `logs/test3_*.log`.
- Wysokości γ_n (n ≤ 110) pochodzą bezpośrednio z `flint.acb.zeta_zeros(1, 110)`, jak w `src/zeros._worker`, z kontrolą promienia kuli. Nie korzystam z float64 z data/.

## (1) H_t kwadraturą w mpmath z adaptacyjną precyzją
Φ(u) = Σ_n (2π²n⁴e^{9u} − 3πn²e^{5u}) exp(−πn²e^{4u}), a H_t(z) = ∫_0^∞ e^{tu²}Φ(u)cos(zu) du. Obie definicje wziąłem z TESTY.md.

Precyzja. Wynik jest rzędu e^{−πz/8}, więc precyzja robocza wynosi dps = D + ⌈πz/(8 ln 10)⌉ + 15 ≈ D + 0,171·z + 15, przy D = 20 cyfrach docelowych. Dla z ≈ 470 daje to około 115 dps.

Silnik referencyjny to `mp.quad` na przedziale [0, u_max], podzielonym w punktach kπ/z. Obcięcie u_max wyznaczam z warunku π e^{4u} > (dps + zapas)·ln 10, co daje u_max ≈ 1,1–1,3. Liczba wyrazów sumy Φ jest ustalana tym samym warunkiem.

Silnik produkcyjny to reguła trapezów w arytmetyce mpmath, z węzłami Φ(u_k) policzonymi raz dla danej precyzji. Wtedy H_t(x) = h·Σ' e^{t u_k²}Φ_k cos(x u_k), a pochodne dostaję analitycznie: ∂_x daje czynnik −u sin, ∂_x² czynnik −u² cos, a ∂_t czynnik u². Uzasadnienie: funkcja podcałkowa jest parzysta i analityczna w pasie |Im u| < π/8, a błąd aliasingu wynosi ≈ H_t(2π/h − z). Stąd warunek na krok h < 2π/(2z + 8D ln10/π), co daje około 200 węzłów. Tę ocenę wyprowadziłem sam, więc traktuję ją jako DO WERYFIKACJI i sprawdzam empirycznie. Ewaluacja kosztuje około 200 mnożeń mpf zamiast pełnego `mp.quad`.

Kontrola stabilności na każdej wartości końcowej: podwajam dps i jednocześnie połowię h. Wymagam względnej zmiany H poniżej 1e−12, a przy t_c zmiany poniżej 1e−10. Trapezy muszą się zgadzać z `mp.quad` na siatce punktów kontrolnych.

## Walidacja (`test3 validate`)
- Porównuję H_0(z) z ξ(1/2 + iz/2)/8, gdzie ξ(s) = ½ s(s−1) π^{−s/2} Γ(s/2) ζ(s) liczę przez `mp.zeta` i `mp.gamma`. Punkty: z ∈ {0, 10, 2γ_1, 50, 100, 200, 2γ_50}. Kryterium to błąd względny poniżej 1e−15 (ξ jest rzeczywiste dla rzeczywistego z, więc sprawdzam też część urojoną ≈ 0).
- Zera H_0 znajduję metodą Newtona/Brenta w x, startując z 2γ_n, dla n ≤ 50. Kryterium: |z_n − 2γ_n| < 1e−8. Dodatkowo liczba zmian znaku H_0 na [0, 2γ_50 + 1] ma wynosić dokładnie 50.
- Wynik zapisuję w `results/test3/validate.json`. Dalsze kroki uruchamiam dopiero po zaliczeniu walidacji.

## (2) t_c jako pierwiastek w t, bez liczenia zmian znaku na siatce
Dla przerwy j między z_j i z_{j+1}:
- x*(t) to lokalne ekstremum H_t w przerwie, czyli rozwiązanie ∂_x H_t(x) = 0 wyznaczane Brentem w x. Przy następnym t startuję z poprzedniego x*.
- Funkcja g(t) = H_t(x*(t)) zmienia znak dokładnie w chwili t_c. Gdy dwa zera schodzą z osi, ekstremum trwa dalej, ale jego wartość przechodzi przez 0. Pierwiastek jest prosty, bo dg/dt = ∂_t H = −∂_x²H ≠ 0 w punkcie x*.
- t_c wyznaczam Brentem (mpmath/scipy) z tolerancją 1e−10 w t. Na koniec powtarzam jedno obliczenie g z podwojonym dps i połowionym h.

Status przerwy:
- `collided`, jeśli jest zmiana znaku i x* przez cały czas pozostaje w przerwie.
- `absorbed`, jeśli ekstremum znika (∂_x² H_t → 0, zlanie punktów krytycznych) albo wychodzi poza przerwę, zanim g zmieni znak.

Parowanie idzie zachłannie według malejącego t_c, bo bliżej 0 zderzenie następuje wcześniej. Kandydata przyjmuję tylko wtedy, gdy żadne z jego zer nie zostało wcześniej zużyte. Pozostałe przerwy oznaczam `neighbor_collided_first`. Zderzeń drugiej generacji (z_{j−1} z z_{j+2}) w fazie A nie liczę, tylko je odnotowuję.

Kontrola spójności: ż = H''/H' liczone w zerach przy t = 0 porównuję z obciętą sumą 2Σ1/(z_j − z_k) po zerach ±z_k, k ≤ 110. To tylko kontrola, a różnicę zapisuję jako informację.

## (3) Przybliżenie izolowanej pary jako rząd wielkości i przedział startowy
- Obliczam t_c⁰ = −Δz²/8, gdzie Δz = z_{j+1} − z_j. Uzasadnienie: ∂_t H = −∂_x² H, więc dla P = x² − Δz²/4 dostajemy P_t = P − 2t, a zderzenie zachodzi przy t = −Δz²/8.
- Przedział startowy to [2·t_c⁰, 0,5·t_c⁰]. Jeśli brak zmiany znaku, rozszerzam go geometrycznie (×2, najwyżej 4 razy), a potem nadaję status `absorbed` lub `no_bracket`.
- Do raportu idą: t_c, t_c⁰, stosunek t_c/t_c⁰, x*(t_c), odległości do sąsiednich zer oraz status. Rząd wielkości dla niskich zer to Δγ ≈ 1,4–6,9, czyli Δz ≈ 2,8–13,8 i t_c⁰ ≈ −1 do −24.
- Przerwę (−z_1, z_1), dla której x* = 0 z symetrii, a t_c⁰ ≈ −400, liczę osobno jako pozycję opcjonalną.

## (4) Pilot i oszacowanie
- `python -m src.test3 pilot --nzeros 10 --out results/test3/tc_pilot.json` liczy przerwy 1–9, więc potrzebuje zera 11. Mierzy czas pojedynczej ewaluacji (trapezy i `mp.quad`) w funkcji z oraz czas jednego t_c, łącznie z kontrolą 2×dps. Wypisuje najwyżej 15 linii. Oczekiwany czas to kilka minut w pierwszym planie; jeśli ma być dłużej, pilot idzie w tle przez `nohup`.
- Oszacowanie dla 100 zer: koszt ewaluacji rośnie jak (liczba węzłów ∝ z)·M(dps ∝ z), więc skalowanie ekstrapoluję z pomiarów pilota przy z od 28 do 100. Zakładam około 100 ewaluacji na przerwę i Pool(6). Liczbę podam, zanim uruchomię `tc --nzeros 100`, które pójdzie w tle przez `nohup` do `logs/test3_tc.log`.
- Po etapie: najwyżej 10 linii w `docs/POSTEP.md` i propozycja `/compact`.

## Weryfikacja
- `validate` spełnia wszystkie kryteria (ξ, 50 zer z tolerancją 1e−8, liczba zmian znaku).
- Na pilocie: t_c jest stabilne przy 2×dps i h/2 z dokładnością do 1e−10. Znak g faktycznie zmienia się w t_c, co sprawdzam g(t_c ± 1e−6). Dla przerw wyraźnie izolowanych stosunek t_c/t_c⁰ jest bliski 1, a odchylenia są zgodne co do kierunku z asymetrią sąsiadów. To jest hipoteza, którą sprawdzam, a nie kryterium.
- Dla jednej przerwy wynik trapezów porównuję z pełnym `mp.quad`.

## Dopisek (2026-09-26): punkt 3 pominięty
Rozszerzenia zakresu do t = −400 dla najniższych szerokich par pokolenia ≥ 2 nie wykonano. Uzasadnienie: mediana ilorazu t_c/t_c⁰ dla zderzeń pokolenia 1 wynosi 1,4 (kwantyl 90%: 3,2), a t_c⁰ trzech najszerszych par cenzurowanych, (42,77), (21,42) i (6,21), leży poniżej −870, więc żadna z nich nie zderzyłaby się przed −400; dla (1,6) i (90,99) szanse byłyby co najwyżej umiarkowane, a zyskiem byłyby dwa punkty w zakresie t, w którym przybliżenie pary izolowanej nie ma sensu. Faza A zamknięta wynikiem results/test3/REPORT_A.md; liczby zbiera `python -m src.test3_summary --out results/test3/summary_A.json`.
