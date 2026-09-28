# Aneks do prerejestracji 3 (prereg3.md pozostaje niezmieniony)

Data: 2026-09-28
Dotyczy: prereg3.md, SHA-256 5df150658cf8be4ce4343ef593919ea19f81f960826398225b52b3d190ba2844
Zapisany po kroku 1 (kalibracja, chem_controls.json), przed jakimkolwiek liczeniem na danych.

STATYSTYKA_GLOWNA: mean_cos

(a) Statystyką główną jest ⟨cos θ⟩ według reguły „większa separacja”: separacja Poisson–Ginibre wynosi 3,50 sd pojedynczej realizacji dla ⟨cos θ⟩ wobec 3,42 sd dla ⟨r⟩ (definicja: |średnia_G − średnia_P| / √((sd_G² + sd_P²)/2), 2000 realizacji). Różnica jest w granicach błędu symulacji; wybór wynika z mechanicznego zastosowania reguły. ⟨r⟩ jest statystyką drugą, opisową.

(b) Odstępstwo od specyfikacji: kontrola zgodności z wartościami literaturowymi nie przeszła dla ⟨cos θ⟩ na pełnym zbiorze punktów górnej półpłaszczyzny (Ginibre −0,174 wobec ~−0,24; Poisson +0,040 wobec 0; tolerancja 0,03), bo przy ~130 punktach duża część leży przy brzegu (oś rzeczywista, okrąg). Kryterium kontroli przeformułowano na zgodność punktów wewnętrznych z odniesieniem. Definicja wnętrza użyta w diagnostyce (csr_diag.py, ziarno 7): widmo całej płaszczyzny (Ginibre rzeczywisty z wpisami N(0, 1/N), Poisson jednostajny w kole jednostkowym), najbliższy i drugi najbliższy sąsiad szukani wśród wszystkich wartości własnych w całej płaszczyźnie, uśrednianie tylko po punktach z |λ| < 0,6 oraz Im λ > 0,15; N = 272 (300 realizacji) i N = 1000 (60 realizacji). Wynik: Ginibre ⟨cos θ⟩ = −0,230 (N = 272) i −0,247 (N = 1000), Poisson +0,008 i −0,008, czyli zgodnie z odniesieniem w tolerancji 0,03 (⟨r⟩ wnętrza: 0,738 i 0,740 dla Ginibre, 0,669 i 0,666 dla Poissona). Samo przetwarzanie danych (pełny zbiór punktów górnej półpłaszczyzny) pozostaje bez zmian.

(c) Odniesieniem dla wyniku na danych jest model zerowy z zachowanymi stopniami wejściowymi i wyjściowymi, przetworzony identycznie, a nie wartości literaturowe.

(d) Diagnostyka była wykonana wyłącznie na symulacjach; dane nie były dotknięte poza liczbą krawędzi i zbiorem wag do kontroli (c) w kroku 1.
