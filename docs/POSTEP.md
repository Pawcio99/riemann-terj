# Dziennik postępu

Każdy wpis: data, etap, najważniejsze liczby, komenda odtwarzająca, status. Najwyżej 10 linii na etap.

2026-09-26, pakiet startowy: kod sprawdzony w piaskownicy (python-flint 0.9.0); tests/baseline_check.py 6/6 PASS; zeros, stats, explicit, rp_model i odlyzko działają. Etap 0 na docelowej maszynie: do wykonania.

## Etap 0 (2026-09-26)
- .venv utworzony (wymagał python3.14-venv); baseline_check 6/6 PASS.
- Punkt 2 (20k zer): 19 999 odstępów, 13 <0,1, var 0,1565, min 0,0421 (zero 6709), a=3,21 [2,82; 3,67] — zgodne z referencją.
- Punkt 3 (10^8, 10k zer): od 42 653 549,76, 6 <0,1 na 9999, var 0,1696, a=3,18 [2,69; 3,81] — zgodne.
- Punkt 5: piki 2..31 bez fałszywych; ψ(100,5) dokładnie 94,045, z 500 zer 93,885 — zgodne.
- Punkt 6 (rp_scan): GUE p01 w ±20% dla λ≤0,6 (0,15: +16%, 0,3: +20%); λ=1,2: 0,0013 vs 0,0016 (−19%); λ=3 i ∞: 0,0013 vs 0,0008/0,0009 — poza ±20%, ale to małe p01 (szum); GOE PR/N 0,34, p01 ~0,008–0,009 OK. Wymaga uwagi przy Teście 2.

## Test 1 (2026-09-26)
- Zweryfikowano wzór N_eff = ln(T/2π)/√(12Λ), Λ = 1,57314 w Bogomolny i in. 2006 (arXiv math/0602270, wzór 19); przykład z artykułu (E = 1,3066·10^22) daje 11,30, zgodnie z wzorem.
- Dane: bloki 10^3…10^8 (src.zeros) i Odlyzko 10^12/10^21/10^22 (data/odl/zeros3-5, konwersja src.odlyzko; poprawiono parser nagłówków).
- Komenda: `python -m src.test1 --out results/test1.json --nrep 200 --nm 1000` (log: logs/test1.log, ~kilkanaście minut).
- [FAKT NUMERYCZNY] Wykładnik odpychania a: |z| ≤ 2,3 względem CUE(N_eff) we wszystkich 9 blokach, znak dodatni (bardziej GUE, nie GOE); iloraz wiarygodności GOE→GUE nieistotny (p_LR 0,19–1,0); oba rozwinięcia zgodne. Kryterium 3σ na ≥3 wysokościach niespełnione: brak anomalii.
- Zastrzeżenie: dla n ≤ 10^8 (N_eff < 4) test KS odrzuca CUE(N_eff) (p_KS ≈ 0,005), więc model zerowy z asymptotycznym N_eff jest tam nieadekwatny (nie jest to dowód domieszki GOE); od 10^12 (N_eff ≥ 5,6) p_KS 0,25–0,88.
- Artefakty Testu 1 uzupełnione (bez przeliczania): results/test1/summary.json, fig_a_vs_T.png, REPORT.md; komenda `python -m src.test1_summary`. Wynik negatywny: max |z| = 2,27, 0 przypadków >3σ.
- Odrzucenie KS przy N_eff < 4 wiązane z CUE rozmiaru 2–3, nie z GOE [HIPOTEZA]; rozstrzygnie Test 2. Proponuję /clear przed Testem 2.

## Test 2 (2026-09-26)
- Tablica CUE(N), N=2…40 + kotwice do 128, interpolacja splajnem w u=1/N²; N̂ = min KS z bootstrapem blokowym; komendy: `python -m src.test2 {table,check,calib,fit} --out results/test2/...` i `python -m src.test2_summary` (~30 min).
- [FAKT NUMERYCZNY] N̂/N_eff maleje z 2,26 (10³) do 1,00 (10²²) wraz z 1/ρ̄; z-score wariancji względem CUE(N_eff) od +11 do −1; wartości kontrolne z czatu odtworzone tylko dla bloków ≥ 10⁸.
- N ≥ 8 nieodróżnialne od ∞ przy 10⁴ odstępach (SD wariancji 0,0022–0,0027); N̂ > 128 raportowane jako dolna granica. p_GOF ≤ 0,0033 dla 10³–10⁵: sam dobór N nie tłumaczy odrzucenia KS z Testu 1.
- [HIPOTEZA] rozbieżność znika dla T → ∞ (wyraz wolny Δu −0,008 ± 0,019); pojedyncza prosta N̂ = βL nie jest interpretowana. Poprawki po przeglądzie: p_GOF w fits.json przeliczone przy N̂ (`python -m src.test2_summary`); różnica z-score względem czatu wyjaśniona innym SD wariancji; arXiv:1608.04638 = Bornemann, Forrester, Mays, N ≈ 11,3 z podpisu rys. 3 (porównanie wariancji nadal DO WERYFIKACJI). Plan fazy A Testu 3 zatwierdzony (bez uruchamiania). Proponuję /clear przed implementacją.

## Test 3, faza A: implementacja (2026-09-26)
- Nowe: src/heatflow.py (H_t trapezami w mpmath, dps adaptacyjne), src/test3.py (validate | pilot | tc). `python -m src.test3 validate --out results/test3/validate.json`: H_0 vs ξ/8 błąd 2,9e-25, |z_n−2γ_n| < 1e-8, 50 zmian znaku, zgodność z mp.quad 4e-40 — PASS.
- Błędy naprawione w trakcie: stałe okno skanu (zera dryfują w prawo dla t<0), filtr H·H''<0 (po t_c ekstremum jest minimum |H|), odwrócony znak testu zmiany znaku.
- Pilot `python -m src.test3 pilot --nzeros 10 --out results/test3/tc_pilot.json`: zderzenia przerw 2, 4, 7, 9, t_c/t_c0 ∈ [1,20; 1,57] (bez naruszeń niezmiennika ≥1), stabilność 1e-15; 5 przerw „absorbed”, każda sąsiaduje ze zderzoną.
- Oczekuje na zgodę: `tc --nzeros 100` (oszacowanie ~7 min szeregowo, ~1–3 min na 6 procesach). Proponuję /compact.

## Test 3, faza A: przebieg 100 zer (2026-09-26)
- `python -m src.test3 tc --nzeros 100 --extra 12 --workers 6 --out results/test3/tc.json` (222 s): 35 zderzeń, 64 „absorbed”, 0 „preempted”; t_c/t_c0 ∈ [1,077; 2,30] (mediana 1,33, bez naruszeń ≥1); stabilność ≤ 2e-13; korelacja ratio z Δz 0,57.
- H_t(0) > 0 na [−100, 0] (201 punktów, min 0,0336); przerwa (−z1, z1) nie zderza się (dowód: dodatnie składniki Φ, 2πn²e^{4u}>3).
- Kontrola liczby zer `python -m src.test3 count --tc results/test3/tc.json --out results/test3/count.json` (659 s): liczba zer w oknie (0, X_e(t)) nie rośnie i spada dokładnie o 2 przy 33 kolejnych zderzeniach do t = −2,38; potem niedobór −2 (t ≈ −2,5), −4 (−5,7), −10 (−9,9) względem przewidywania z pokolenia 1 — zderzenia kolejnych pokoleń lub błędnie sklasyfikowane „absorbed”; do rozstrzygnięcia.
- Z 64 „absorbed” 53 sąsiaduje ze zderzoną przerwą (pilot sugerował 100%).

## Test 3, faza A2: zderzenia kolejnych pokoleń (2026-09-26, implementacja + pilot)
- Nowe: src/test3_gen.py (symulacja zdarzeń: para sąsiadnich zer rzeczywistych zderza się, gdy najwyższy garb sH między nimi dochodzi do 0; nowe pary (L,R) po każdym przyjętym zderzeniu, numer pokolenia = 1 + max pokolenia usuniętych zer między nimi; krok t adaptacyjny). `src.test3 count` czyta teraz zdarzenia wszystkich pokoleń i kończy kontrolę przy zderzeniu przez brzeg okna.
- Pilot `python -m src.test3_gen --nzeros 10 --extra 8 --workers 6 --out results/test3/gen_pilot.json` (102 s): regresja gen-1 względem tc.json: max |Δt_c| 7e-14; przerwa 11 zmienia status „absorbed” → zderzenie t_c = −27,9 (stare „absorbed” było fałszywe); brak naruszeń ≥1. Pary pokolenia 2 (np. z1–z6 po zderzeniach 2–3, 4–5): 0 zderzeń do t = −100 (3× no_bracket, 2× absorbed przy −15 i −31).
- `count` na pilocie: liczba zer zgodna z przewidywaniem do t = −28,4 (12→2, −2 na zderzenie).
- Do zrobienia po zgodzie: pełny przebieg `python -m src.test3_gen --nzeros 100 --extra 12 --workers 6 --out results/test3/gen.json` i `count` na nim; jego wynik rozstrzygnie niedobory −2/−4/−10 z tc.json.

## Test 3, faza A2: statusy i start pełnego przebiegu (2026-09-26)
- Statusy: „absorbed” tylko z przyczyną w dzienniku (zderzenie zera pary z innym sąsiadem, t_c ≥ utrata garbu − 0,05; przyczyna ma pierwszeństwo także przed „censored”, bo garb żyje dalej jako punkt krytyczny); bez przyczyny „lost” i powtórzenie z dt/4, dt/16; wyjście poniżej tmin = „censored” (t_c < −100), nie brak zderzenia. Pilot: 9 zderzeń, 15 absorbed (wszystkie z przyczyną), 0 lost, 1 censored.
- Start (tło): `python -m src.test3_gen --nzeros 100 --extra 12 --workers 6 --out results/test3/gen.json` (log logs/test3_gen.log), potem automatycznie `python -m src.test3 count --tc results/test3/gen.json --out results/test3/count_gen.json` (log logs/test3_count_gen.log). Oczekiwane ok. 35–55 min.

## Test 3, faza A: wynik pełnego przebiegu (2026-09-26)
- `python -m src.test3_gen --nzeros 100 --extra 12 --workers 6 --out results/test3/gen.json` (1569 s) i `python -m src.test3 count --tc results/test3/gen.json --out results/test3/count_gen.json`. Zderzeń 53 (pokolenie 1: 48, pokolenie 2: 5; przy obu zerach ≤ 100: 46 = 42 + 4), pokolenie 3: 0. Statusy 155 par: 53 collided, 6 preempted, 92 absorbed (wszystkie z przyczyną), 1 lost (113–114, brzeg górny), 3 censored (1–6, 21–42, 77–90).
- Pokolenie 1 w zakresie (42 zderzenia): t_c/t_c0 min 1,077, mediana 1,386, q10 1,114, q90 3,17, max 16,07 (przerwa 55, t_c = −76,6); brak naruszeń ≥1; stabilność ≤ 2,5e-13, zmiana znaku wszędzie. Regresja względem tc.json: max |Δt_c| 2,3e-13 (41 wspólnych); 7 dawnych „absorbed” to zderzenia (m.in. przerwa 11: t_c = −27,89, ratio 4,62), 5 to preempted.
- Kontrola liczby zer: 50 punktów zgodnych z predykcją, liczba nie rośnie, ale ważna tylko do t = −48,78 (zderzenie przez brzeg okna); zderzenie 55–56 przy −76,6 pozostaje nieweryfikowane liczbą zer.
- Luka: `spawn` po cichu pomijał pary bez seedów; sąsiednie pary przeżywających zer (6,21), (42,77), (90,99), (99,114) nie mają rekordu (przeżywa 8 z 114 zer: 1, 6, 21, 42, 77, 90, 99, 114). Do naprawy przed rozszerzeniem zakresu.

## Test 3, faza A: punkt 1 (naprawa seedów, pilot) (2026-09-26)
- Zmiana w src/test3_gen.py: para bez seedu z sąsiadów nie jest już pomijana; zera L i R śledzone Newtonem od t = 0 do t_start (krok ≤ 0,1), garb szukany skanem H_x między nimi; niepowodzenie = status `no_seed` w JSON. Wynik zapisuje ścieżki garbów, listę przeżywających zer i `missing_survivor_pairs`; opcja `--force-track` ignoruje seedy (walidacja).
- Pilot `python -m src.test3_gen --nzeros 10 --extra 8 --workers 6 --out results/test3/gen_pilot.json` (115 s) i z `--force-track` (`gen_pilot_ft.json`, 120 s): 25 par, statusy i t_c identyczne (różnica 0), przeżywają zera 1 i 6, brak brakujących par, 0 `no_seed`. Pilot nie zawiera pary, która wcześniej byłaby pominięta, więc fallback jest sprawdzony tylko na startach par, nie na pominiętych (6,21), (42,77), (90,99), (99,114) — te pojawią się w pełnym przebiegu.
- Poprawka po pilocie: start pary z seedów wybierał garb sąsiedniej pary zamiast najwyższego garbu całego przedziału (okno skanu zbyt wąskie, np. para 11–16: 125,01 zamiast 117,05, H 2,2e-19 vs 1,9e-18). Teraz okno 0,8·rozpiętość + 1 wokół środka seedów; start z seedów i ze śledzenia zer zgodne co do 0,0 dla wszystkich 6 par pokolenia 2, te same zdarzenia. Wynik gen.json (1569 s) użył starego wyboru startu, więc jest zastąpiony przez punkt 2.

## Test 3, faza A: punkt 2, start (2026-09-26)
- Nowe: `python -m src.test3 count --dividers` (okno (0, X), X = garb żyjącej pary; predykcja p − 2·#zderzeń z t_c > t i oboma zerami < p; na pilocie zgodne w 9/9 punktach), src/test3_gridcheck.py (powtórka par z gęstością siatki ×2 i ×4).
- Start w tle: gen2.json (log logs/test3_gen2.log), potem gridcheck.json dla par 6–21, 42–77, 90–99, 99–114 (log logs/test3_gridcheck.log), potem count2.json (log logs/test3_count2.log). Oczekiwane 35–45 min + ok. 10 min + ok. 10 min.

## Test 3, faza A: punkt 2, wynik gen2 i błąd pary (42,77) (2026-09-26)
- gen2.json: 53 zderzenia, statusy 165 par (53 collided, 6 preempted, 99 absorbed, 1 lost, 6 censored, 1 no_seed); count2 (dividery) zgodny w 43/43 punktach do t = −77,13, zdarzenie 55–56 potwierdzone (1 punkt poniżej −76,6 z p ≥ 57). gridcheck (×2, ×4) dla (6,21), (90,99), (99,114): |Δ start| ≤ 2,4e-7, wynik identyczny; fallback bez seedów użyty dla (6,21) i (90,99).
- (42,77) nie ma rekordu. Przyczyna: klucz pamięci `spawned` = (i, j, t_c) bez sąsiadów L, R; zdarzenie 55–56 w pierwszej fali dostało L=50, R=67 (stąd jedyny `no_seed`: para (50,67)), a po odkryciu zderzeń 47–50 itd. jego sąsiedzi to 42 i 77, ale wpis już istniał. Naprawa: klucz z L, R, walidacja pary (i,j) == (L,R) w greedy i filtrze; opcja `--resume`. t_c⁰(42,77) = −2295 (nie −1160; to (21,42)).
- Start: `python -m src.test3_gen --nzeros 100 --extra 12 --workers 6 --resume results/test3/gen2.json --out results/test3/gen3.json` (log logs/test3_gen3.log; log pisze dopiero po fali).

## Test 3, faza A: podsumowanie (2026-09-26)
- gen3.json (wznowienie z gen2, 128 s): 166 par (113 gen-1 + 53 spawnowanych): 53 collided, 6 preempted, 99 absorbed (wszystkie z przyczyną), 7 censored, 1 lost (113–114, brzeg), 0 no_seed; brak brakujących par przeżywających zer (1, 6, 21, 42, 77, 90, 99, 114). Zdarzenia: 48 (gen 1) + 5 (gen 2), 46 przy obu zerach ≤ 100.
- count3 (dividery, p = 99): 43/43 punktów zgodnych do t = −77,13, zdarzenie 55–56 potwierdzone (1 punkt); gridcheck ×2, ×4 dla (6,21), (90,99), (99,114), (42,77): |Δ start| ≤ 3,5e-7, wyniki identyczne.
- censored (t_c⁰): (42,77) −2295, (21,42) −1161, (6,21) −872, (99,114) −343, (77,90) −283, (1,6) −275, (90,99) −107. Rozszerzenie do −400 nie ma sensu dla trzech pierwszych; do decyzji.
- Regresja względem tc.json: 12 zmian absorbed → collided/preempted (7 + 5) to poprawka błędu starego skryptu (zbyt grube kroki t, garb uciekał z okna), nie zmiana nazw; potwierdza ją kontrola liczby zer (7 nowych zderzeń w predykcji).

## Test 3, faza A: zamknięcie (2026-09-26)
- Faza A zamknięta bez punktu 3 (uzasadnienie w docs/PLAN_TEST3_A.md). Raport: results/test3/REPORT_A.md; liczby: `python -m src.test3_summary --out results/test3/summary_A.json` (src/test3_summary.py).
- Doprecyzowanie: 12 zmian statusu względem tc.json (7 collided + 5 preempted, z czego 11 w zakresie) to poprawka merytoryczna, nie zmiana nazw; zgodność preempted z partnerem pokolenia 2 jest bitowo dokładna (śledzony ten sam garb), więc to słabe potwierdzenie.

## Test 3, analiza geometrii (2026-09-26)
- Plan: docs/PLAN_TEST3_B.md (model zerowy GUE/CUE pominięty z uzasadnieniem). Kod: src/test3_geom.py; `python -m src.test3_geom --out results/test3/geom.json` (4 s), rysunek results/test3/fig_geom.png, ziarno bootstrapu 20260926.
- [FAKT NUMERYCZNY] 42 zderzenia pokolenia 1: T < 0 i t_c/t_c⁰ ≥ 1 w 42/42. Predyktor stałego T (x = −TΔ/4): dla x < 1 (36 par) R² 0,978, LOO 0,028, β = 0,68 [0,65; 0,74] zamiast 1; dla 6 par x ≥ 1 (przewiduje brak zderzenia). Reszta bez dopasowania jest monotoniczną funkcją δ (−0,03 … −0,41).
- neighbor_preempted dosłownie: 0 z 42 zderzonych (z konstrukcji), 4 z 4 preempted; uogólnienie (zero sąsiada uczestniczy w przyjętym zderzeniu z mniejszym |t_c|): 21 z 42 zderzonych, w tym wszystkie 6 par x ≥ 1. Zbiór ostrożny (n = 21): R² 0,991, LOO 0,013, β = 0,78 [0,74; 0,84]; selekcja faworyzuje małe |t_c|.

## Test 3, test reszty arytmetycznej (2026-09-26)
- Kod src/test3_arith.py (`corr`, `test`), wyniki results/test3/arith_corr.json, arith_test.json; opis w results/test3/REPORT_B.md. Naruszenia prawa Grama dla n ≤ 200: 126, 134, 195 (zweryfikowane liczbowo).
- [FAKT NUMERYCZNY] Geometria w kandydatach: ln|ζ′| 97,5% wariancji (LOO 97,0%), margines Grama 85,3%, faza Gramowska 11,4% (LOO ujemne). Trend w x (izotoniczny/splajn) usuwa >99% wariancji, ale ślad ln d_L w resztach pozostaje (Spearman −0,47, p ≈ 0,004).
- Wynik negatywny: żaden z 3 kandydatów istotny po Bonferronim (α = 0,0167); najmniejsze p = 0,046 (nominalnie, splajn: faza Gramowska i margines Grama); moc: |ρ| ≥ 0,51 przy n = 36. Isotoniczny trend w zbiorze ostrożnym (n = 21) jest praktycznie interpolacją (sd reszt 0,0002), więc bez wartości.

## Test 3: podsumowanie (2026-09-26; REPORT_A.md, REPORT_B.md; komendy w raportach)
- [TWIERDZENIE] (jak w docs/TESTY.md, tu niezweryfikowane ponownie): H_0(z) = ξ(1/2 + iz/2)/8, zera H_0 = 2γ_n; RH ⇔ Λ ≤ 0; Λ ≥ 0 (Rodgers–Tao), Λ ≤ 0,2 (Platt–Trudgian). [TWIERDZENIE elementarne] H_t(0) > 0 dla każdego t (składniki Φ dodatnie), więc para (−z₁, z₁) się nie zderza.
- [FAKT NUMERYCZNY] Silnik H_t (trapezy, mpmath): błąd względny względem ξ/8 2,9e-25, 50 zmian znaku dla n ≤ 50. Faza A: 166 par, 53 zderzenia (48 pokolenia 1, 5 pokolenia 2), 46 przy obu zerach ≤ 100; t_c/t_c⁰ w 42 zderzeniach pokolenia 1: min 1,077, mediana 1,386, q90 3,17, max 16,07, wszystkie ≥ 1; 7 par cenzurowanych (t_c < −100), 1 lost (brzeg), 6 preempted.
- [FAKT NUMERYCZNY] Kontrola liczby zer (dividery): 43/43 zgodnych do t = −77,13, zdarzenie 55–56 potwierdzone; dawne 12 „absorbed” to poprawka błędu skryptu (35 → 42 zderzeń), preempted potwierdzone tylko zgodnością t_c z partnerem pokolenia 2 (słabe).
- [FAKT NUMERYCZNY] Geometria: T < 0 i t_c/t_c⁰ ≥ 1 w 42/42; dla 36 par z x < 1 (stałe T): β = 0,678 [0,652; 0,744], R² = 0,978, LOO 0,028 (β = 1 odrzucone); 6 zderzeń ma x ≥ 1 (model: brak zderzenia); 21 z 42 par ma sąsiednie zero usunięte wcześniej.
- [FAKT NUMERYCZNY] Test arytmetyczny (36 par, trend w x izotoniczny/splajn, Bonferroni α = 0,0167): ln|ζ′| 97,5%, margines Grama 85% wariancji to geometria, faza Gramowska 11%; żaden kandydat istotny (najmniejsze p = 0,046 nominalnie); moc: |ρ| ≥ 0,51; w resztach zostaje ślad ln d_L (Spearman −0,47, p = 0,0034).
- [HIPOTEZA] β < 1 wynika z tego, że T zmienia się do zderzenia (sąsiedzi się poruszają, dla większego δ dłużej); możliwa asymetria wpływu lewego i prawego sąsiada; niesprawdzone (test: T w chwili pośredniej). Wynik arytmetyczny jest negatywny przy małej mocy i nie dowodzi braku związku z arytmetyką.
- Nie wykonano: rozszerzenia do t = −400 (uzasadnienie w PLAN_TEST3_A.md), modelu zerowego GUE/CUE (PLAN_TEST3_B.md), fazy B (para Lehmera, wymaga przybliżenia Polymath). Proponuję /clear przed następnym testem.
- Reanaliza (poza planem): hipoteza fluktuacyjno-dyssypacyjna |T|↔rozrzut reszt na 36 parach x<1 — odrzucona w kierunku przeciwnym (Spearman |resid|~|T| ρ=0,578, p=0,0002; wariancja reszt silne/słabe 0,00982/0,00191, F(17,17) p=0,0016); po kontroli δ korelacja cząstkowa spada do 0,044 — efekt to artefakt już znanej zależności reszty od δ, nie nowy mechanizm. `src/test3_fluctdiss.py`, `results/test3/fluctdiss.json`, sekcja w REPORT_B.md.
- Reanaliza (poza planem): N_eff(γ) vs |T| i jakość dopasowania na tych samych 36 parach — N_eff nie jest niemal stałe w tej próbie (zakres [0,299; 0,830]), koreluje z |T| niezależnie od δ (Spearman 0,518→0,815 po kontroli δ), ale nie z jakością dopasowania (|reszta|, ρ=−0,162, p=0,344, niekonfundowane). `src/test3_neff_check.py`, `results/test3/neff_check.json`, sekcja w REPORT_B.md. Sprawdzam, czy sensowny jest mini-test H_t na wyższych blokach Testu 2 — koszt precyzji rośnie liniowo z wysokością (heatflow.py), więc oceniam wykonalność benchmarkiem przed uruchomieniem czegokolwiek na noc.

## Freestyle: NEC dla metryki generowanej przez H (2026-09-27; eksploracja poza docs/TESTY.md, kod tylko w scratchpadzie)
- Model: ds² = −A dt² + B dx² + r(t,x)² dΩ², r = r0 + αH², r0 = 1; r_tt, r_tx liczone numerycznie z src/heatflow.py (różnice centralne w t, ε = 1e-3). Wyprowadzenie T_μν k^μ k^ν = −2(r_tt + 2r_tx + r_xx)/r zweryfikowane trzema niezależnymi kontrolami: redukcja do Morrisa–Thorne'a, zerowanie R na wektorze zerowym, stabilność przy ε/2 i 2×dps (przy α(z): zmiana T_kk ≤ 1e-7 względnie; wcześniejsza kontrola przy α = 1 była pusta dla r_tt, bo αH² ~ 1e-125 ginie przy r0 = 1).
- [FAKT NUMERYCZNY, eksploracyjny] 46 par (collided + preempted z geom.json), α(z) = e^{πz/4}: T_kk < 0 tylko w 2/46; pozostaje istotny trend resztkowy |T_kk| z wysokością (Spearman ρ = 0,334, p = 0,023; αH² ~ z: ρ = 0,70), więc zależności T_kk od |T| (ρ = −0,08) i δ (ρ = −0,36, p = 0,015) nie da się oddzielić od efektu skali. Przyczyna: H(z) ma poza wykładniczym zanikiem czynnik wielomianowy, którego uzasadnienie α nie uwzględniało.
- Zamknięte jako wynik nierozstrzygnięty; dalszych normalizacji nie próbowano: α dopasowane per para maskuje problem (T_kk przestaje być porównywalne między parami), a kontrola cząstkowa na δ i z_mid (same skorelowane) rozmywa go na dwie zmienne bez usunięcia. Status: konstrukcja geometryczna poprawna i zweryfikowana, ale nie daje interpretowalnego wyniku o związku naruszenia NEC z |T| i δ; to ani potwierdzenie, ani odrzucenie hipotezy, tylko granica tego ansatzu.
- Liczby nie są odtwarzalne z repozytorium (celowo: skrypty nec_*.py i nec_batch.json zostają w scratchpadzie sesji, nie w src/ ani tests/), więc nie wolno ich cytować w REPORT_*.md bez przeniesienia kodu jako osobnego, wyraźnie oznaczonego testu.
- [FAKT NUMERYCZNY, eksploracyjny] Obie pary z T_kk < 0 (przerwy 63 i 91: z_mid 339,0 i 442,2, ranga 30 i 43 z 46 przy zakresie 46,0–463,2; |T| 0,442 i 0,495, ranga 7 i 13 z 46) leżą w górnej części rozkładu wysokości, ale nie na samym skraju, i mają małe |T| (dolna trzecia część).

## Poza projektem: statystyka poziomów konektomu C. elegans (2026-09-28; eksploracja poza rejestrem przewidywań TERJ)
- Dane: rdzeń złączy szczelinowych hermafrodyty C. elegans (Cook i in. 2019, SI 5 poprawione w lipcu 2020; 272 neurony bez PHARYNX i SEX SPECIFIC, największa składowa 266 neuronów), statystyka stosunku kolejnych odstępów r̃ widma macierzy ważonej; oczekiwanie zapisane przed liczeniem w prereg.md, runda druga w prereg2.md.
- [FAKT NUMERYCZNY, eksploracyjny] Kontrole potoku (2000 realizacji, n = 266): GOE 0,5310, Poisson 0,3863. Wynik główny r̃ = 0,5079 wobec 0,5150 ± 0,0190 w 2000 grafach o tych samych stopniach z przetasowanymi wagami (z = −0,37, p = 0,70), czyli bez wyróżnienia względem modelu zerowego. Macierz jest rzeczywista symetryczna, więc klasą odniesienia jest GOE, nie GUE, i nie jest to ta sama klasa symetrii co zera ζ.
- [FAKT NUMERYCZNY, eksploracyjny] Spadek r̃ części lustrzanie symetrycznej (0,4306) i podwyższone IPR (0,0748 wobec 0,0112 dla GOE) występują w podobnej skali w modelu zerowym (0,4330 ± 0,0202, percentyl 46,8; IPR 0,0667 ± 0,0044, percentyl 97,4).
- [HIPOTEZA] Sektor nieparzysty L/R (88 poziomów): r̃ = 0,4325 wobec 0,5083 ± 0,0328, percentyl 1,8. Powstał w rundzie eksploracyjnej z kilkoma porównaniami naraz; przy około 8 niezależnych porównaniach szansa na co najmniej jedną wartość poniżej 2. percentyla wynosi około 15% (1 − 0,98⁸ ≈ 0,149), więc wymaga osobnego, prerejestrowanego testu.
- Ograniczenia: jedna realizacja widma jednego zwierzęcia; macierz synaps chemicznych przygotowana, ale niezbadana; P(s) po rozwinięciu wielomianem (stopnie 5, 7, 9) nie działa z powodu dwóch skrajnych wartości własnych (niedodatnie odstępy).
- Dane i skrypty leżą poza repozytorium, w katalogu dane-elegans/ obok projektu (notatka zamykająca NOTATKA.md tamże), więc te liczby nie są odtwarzalne z riemann-terj i nie należą do REPORT_*.md.
