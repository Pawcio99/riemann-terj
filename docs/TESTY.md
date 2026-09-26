# Specyfikacja etapów

Czytaj tylko sekcję, nad którą pracujesz. Każdy test kończy się plikami `results/testN/summary.json` i `results/testN/REPORT.md` (proza po polsku, statusy twierdzeń jak w docs/ZASADY_PRACY.md) oraz wykresami PNG, których nie wyświetlasz w kontekście.

## Etap 0: reprodukcja wyników bazowych

Wyniki uzyskane wcześniej tym samym kodem (Arb/python-flint, jeden rdzeń) służą jako wartości referencyjne.

1. `python tests/baseline_check.py`: wszystkie kontrole PASS.
2. `python -m src.zeros --start 1 --count 20000 --workers 6 --out data/z_1_20k.npz`, potem `src.stats`. Oczekiwane: 19 999 odstępów, 13 odstępów krótszych niż 0,1 (0,065%), wariancja 0,1565, najmniejszy odstęp 0,0421 między zerami 6709 i 6710 przy t ≈ 7005,06 (para Lehmera z 1956 r.).
3. `python -m src.zeros --start 100000000 --count 10000 --workers 6 --out data/z_1e8_10k.npz` (w tle), potem `src.stats`. Oczekiwane: wysokość od 42 653 549,76, 6 odstępów krótszych niż 0,1 na 9999, wariancja 0,1696.
4. Wartości odniesienia z przybliżeń Wignera dla P(s<0,1): Poisson 0,0952, GOE 0,00782, GUE 0,00107. Wariancje: GUE 0,178, GOE 0,273.
5. `python -m src.explicit data/z_1_20k.npz --nzeros 500`: piki w 2, 3, 4, 5, 7, 8, 9, 11, 13, 17, 19, 23, 25, 27, 29, 31 i brak pików fałszywych; ψ(100,5): dokładnie 94,045, z 500 zer 93,885, z 5000 zer około 94,02.
6. `python -m src.rp_model --out results/rp_scan.json`. Oczekiwane w granicach szumu (±20% dla małych p01): dla Ĝ zespolonego p01 przy λ = 0; 0,03; 0,08; 0,15; 0,3; 0,6; 1,2; 3; ∞ wynosi około 0,087; 0,071; 0,025; 0,0095; 0,0035; 0,0017; 0,0016; 0,0008; 0,0009, a PR/N rośnie od 0,005 do 0,505. Dla Ĝ rzeczywistego p01 nasyca się przy około 0,008, a PR/N przy około 0,34.

Czas na i7-1165G7 z 6 procesami: punkt 2 około minuty, punkt 3 około minuty lub dwóch, punkt 6 około 20 sekund.

## Test 1 (operator τ̂): domieszka symetrii odwrócenia czasu

Pytanie: czy statystyka zer na rosnących wysokościach zawiera składnik klasy GOE (układy z symetrią T) ponad znane poprawki skończonej wysokości? Status hipotezy: [HIPOTEZA]; oczekiwanie standardowe to brak domieszki.

Dane: bloki po 10 000 zer od n = 10^3, 10^4, 10^5, 10^6, 10^7, 10^8 (`src.zeros`) oraz tabele Odlyzki z https://www-users.cse.umn.edu/~odlyzko/zeta_tables/ (po 10^4 zer od n = 10^12 + 1, 10^21 + 1 i 10^22 + 1; plik z pierwszymi 2 001 052 zerami). Format plików sprawdź przez `head -n 5` i konwertuj przez `src.odlyzko`.

Metoda. Po pierwsze, wykładnik odpychania a (MLE dla s < 0,25, `stats.repulsion_exponent`) z przedziałem bootstrap, porównany z tym samym estymatorem na próbkach modelu zerowego tej samej liczności. Po drugie, model zerowy: CUE o efektywnym wymiarze N_eff zależnym od wysokości T według Bogomolny, Bohigas, Leboeuf, Monastra (2006), J. Phys. A 39, 10743. Z pamięci N_eff = ln(T/2π)/√(12Λ) z Λ ≈ 1,57314 — DO WERYFIKACJI w artykule przed użyciem. Porównuj całe rozkłady (statystyka KS, χ² na histogramie), a p-wartości licz z symulacji, nie z tablic asymptotycznych. Po trzecie, model alternatywny: przejście GOE→GUE (`rmt.crossover_sample`, H = S + iαA) z dopasowaniem α; sprawdź testem ilorazu wiarygodności na symulacjach, czy skończone α jest istotnie lepsze od czystego modelu unitarnego.

Kryterium: anomalię zgłaszamy tylko wtedy, gdy odchylenie od modelu zerowego przekracza 3σ spójnie na co najmniej trzech wysokościach i przeżywa zmianę metody rozwijania widma.

## Test 2: efektywny wymiar modelu Ĥ + λĜ

Pytanie: jakiego rozmiaru N (i ewentualnie λ) potrzebuje model, żeby odtworzyć statystykę odstępów zer na wysokości T, i czy N rośnie liniowo w ln(T/2π)?

Metoda: dla każdego bloku z Testu 1 dopasuj N w modelu CUE(N) (minimum odległości KS), a dodatkowo parę (N, λ) w modelu Ĥ + λĜ z Ĝ zespolonym. Wykreśl N̂ względem ln(T/2π) z niepewnościami bootstrap i porównaj nachylenie z przewidywaniem Bogomolnego i in. (DO WERYFIKACJI). Nie myl tego z normalizacją Keatinga–Snaitha (N ≈ ln(T/2π) dla momentów |ζ|), która dotyczy innej wielkości.

Interpretacja w raporcie: [ANALOGIA TERJ] — liczba efektywnych stopni swobody „pola” rośnie logarytmicznie z energią.

## Test 3: przepływ de Bruijna–Newmana i „najsłabsze relacje”

Status (2026-09-26): faza A wykonana, wraz z analizą geometrii t_c i testem reszty arytmetycznej (opisy: results/test3/REPORT_A.md i results/test3/REPORT_B.md; plany: docs/PLAN_TEST3_A.md i docs/PLAN_TEST3_B.md; podsumowanie: wpis „Test 3: podsumowanie” w docs/POSTEP.md). Wynik testu arytmetycznego jest negatywny. Punkt rozszerzenia zakresu do t = −400 pominięto (uzasadnienie w docs/PLAN_TEST3_A.md). Faza B (para Lehmera, przybliżenie efektywne Polymath) pozostaje nierozpoczęta.

Definicje [TWIERDZENIE]: Φ(u) = Σ_{n≥1} (2π²n⁴e^{9u} − 3πn²e^{5u}) exp(−πn²e^{4u}), H_t(z) = ∫_0^∞ e^{tu²} Φ(u) cos(zu) du. H_0(z) = ξ(1/2 + iz/2)/8, więc zera H_0 to z = 2γ_n. Λ to najmniejsze t, przy którym wszystkie zera H_t są rzeczywiste; RH ⇔ Λ ≤ 0. Wiadomo, że Λ ≥ 0 (Rodgers–Tao 2018) i Λ ≤ 0,2 (Platt–Trudgian 2021). Przy t < 0 bliskie pary zer zderzają się i schodzą z osi rzeczywistej. Dynamika zer rzeczywistych: dz_j/dt = 2 Σ_{k≠j} 1/(z_j − z_k) przy sumowaniu symetrycznym; używaj jej tylko jako kontroli spójności torów.

Faza A (niskie wysokości). Zaimplementuj H_t kwadraturą mpmath z wysoką precyzją. Wynik jest wykładniczo mały względem całki z modułu (rzędu e^{−πz/8}), więc precyzję dobieraj adaptacyjnie i sprawdzaj stabilność przy podwojeniu dps. Walidacja: zera H_0 pokrywają się z 2γ_n dla n ≤ 50 z dokładnością 1e−8. Potem dla pierwszych około 100 zer śledź zera jako funkcję t < 0 (siatka t, zmiany znaku H_t(x) dla rzeczywistego x) i wyznacz czasy zderzeń t_c sąsiednich par. Czasy zderzeń dla niskich zer mogą być rzędu dziesiątek jednostek ujemnych; najpierw zbadaj zakres t na małej próbce.

Faza B (dopiero po zaliczeniu fazy A). Okolica pary Lehmera (zera 6709–6710, z ≈ 14010) wymaga przybliżenia efektywnego z pracy D.H.J. Polymath „Effective approximation of heat flow evolution of the Riemann ξ function, and a new upper bound for the de Bruijn–Newman constant” (2019). Odszukaj ją na arXiv i zweryfikuj wzory przed implementacją.

Analiza: zależność t_c od znormalizowanej odległości pary i odległości do sąsiadów, a następnie test, czy po uwzględnieniu tej geometrii lokalnej coś arytmetycznego (np. naruszenia prawa Grama, |ζ′(ρ)|) wyjaśnia resztę zmienności t_c. Oczekiwanie: dominuje geometria lokalna; każda istotna reszta arytmetyczna jest wynikiem wartym opisania.
