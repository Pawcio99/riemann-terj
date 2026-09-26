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
