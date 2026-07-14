# Project Work — Optimization Techniques for Machine Learning                                                                                                        
                                                                                                                                                                       
  Consegna del Project Work per il corso di *Optimization Techniques for                                                                                               
  Machine Learning*.  
                                                                                                                                                                       
  **Studente:** Michelangelo Gaetani Lovatelli — Intelligenza Artificiale                                                                                            
  **Docente:** Prof. Matteo Lapucci — A.A. 2025/2026                                                                                                                   
                                                                                                                                                                       
  Il lavoro implementa e confronta empiricamente tre algoritmi classici di                                                                                             
  ottimizzazione non vincolata su un sottoinsieme di 15 problemi                                                                                                
  tratti dalla libreria **MASTSIF** di CUTEst. La relazione discussa in                                                                                                
  sede d'esame è contenuta nel file [`report.pdf`](report.pdf); questo                                                                                                 
  README documenta il codice a supporto e ne descrive la riproducibilità.                                                                                              
                                                                                                                                                                       
  ---                                                       
                                                                                                                                                                       
  ## Algoritmi implementati                                                                                                                                            
   
  | Solver | Direzione | Line-search | Budget iter. |                                                                                                                  
  |---|---|---|---|                                         
  | Gradient Descent | `-∇f(x)` | Armijo backtracking | 5000 |
  | Newton | soluzione di `∇²f(x) d = -∇f(x)` (Cholesky → LU) con salvaguardia *gradient-related* e fallback a GD | Armijo backtracking | 1000 |                       
  | BFGS | `-H_k ∇f(x)` con update BFGS dell'inversa dell'Hessiana | Wolfe forte (`scipy.optimize.line_search`) | 2000 |                                               
                                                                                                                                                                       
  **Iperparametri adottati**                                                                                                                                           
                                                                                                                                                                       
  - Tolleranza sul gradiente: `ε = 1e-4` (criterio d'arresto `‖∇f(x_k)‖ ≤ ε`)                                                                                          
  - Armijo: `α₀ = 1`, `ρ = 0.5`, `c₁ = 1e-4`, max 50 backtracks per iterazione
  - Wolfe (BFGS): `c₁ = 1e-4`, `c₂ = 0.9`                                                                                                                              
  - Newton: soglia gradient-related `σ = 1e-4`                                                                                                                         
                                                                                                                                                                       
  La motivazione dettagliata di queste scelte e la loro coerenza con la                                                                                                
  letteratura di riferimento (Grippo–Sciandrone, 2011) sono discusse in
  sezione 2 della relazione.                                                                                                                                           
                                                            
  ---                                                                                                                                                                  
                                                            
  ## Struttura della consegna

  ```text
  opt_ml_project/
  ├── report.tex / report.pdf  # relazione finale (documento d'esame)                                                                                                  
  ├── main.py                  # esecutore del benchmark
  ├── plots.py                 # generazione delle 3 figure incluse nella relazione                                                                                    
  ├── test_env.py              # controllo dell'ambiente CUTEst
  ├── methods/                                                                                                                                                         
  │   ├── gradient_descent.py                               
  │   ├── newton.py                                                                                                                                                    
  │   └── bfgs.py                                           
  ├── results/                                                                                                                                                         
  │   ├── benchmark.csv        # metriche per (problema, solver) — dati usati in Tabella 2
  │   └── histories.pkl        # traccia iterazione-per-iterazione — dati usati in Figura 3                                                                            
  ├── figures/                                                                                                                                                         
  │   ├── plot1_cond_vs_iters.png    # Figura 1 della relazione                                                                                                        
  │   ├── plot2_n_vs_runtime.png     # Figura 2 della relazione                                                                                                        
  │   └── plot3_convergence.png      # Figura 3 della relazione                                                                                                        
  └── requirements.txt                                                                                                                                                 
  ```                                                                                                                                                                  
                                                                                                                                                                       
  ---                                                       

  ## Riproduzione dei risultati

  **Prerequisiti**

  - Python 3.10+
  - Installazione funzionante di CUTEst (`SIFDECODE`, `MASTSIF`, variabili
    d'ambiente `CUTEST`, `SIFDECODE`, `MASTSIF`, `ARCHDEFS`,                                                                                                           
    `PYCUTEST_CACHE`)                                                                                                                                                  
  - `gfortran` / `gcc` per la compilazione dei problemi SIF                                                                                                            
                                                                                                                                                                       
  **Ambiente Python**                                                                                                                                                  
   
  ```bash                                                                                                                                                              
  python -m venv .venv                                      
  source .venv/bin/activate
  pip install -r requirements.txt
  python test_env.py    # verifica che siano visibili i 92 problemi unconstrained
  ```                                                                                                                                                                  
   
  **Rigenerazione completa dei risultati**                                                                                                                             
                                                            
  ```bash                                                                                                                                                              
  python main.py        # riscrive results/benchmark.csv e results/histories.pkl
  python plots.py       # riscrive figures/plot{1,2,3}_*.png                                                                                                                                                
  ```                                                                                                                                                                  
                                                                                                                                                                       
  I risultati numerici presentati nella relazione e nel PDF committato                                                                                                 
  sono esattamente quelli prodotti dai comandi qui sopra su una macchina
  Apple Silicon con CUTEst installato via Homebrew; a parità di libreria                                                                                               
  non si osservano scostamenti significativi tra esecuzioni successive                                                                                                 
  (le differenze sono limitate al runtime).                                                                                                                            
                                                                                                                                                                       
  **Opzioni ausiliarie di `main.py`** (non necessarie per riprodurre i                                                                                                 
  risultati della relazione, incluse per completezza)                                                                                                                  
                                                                                                                                                                       
  | Flag | Descrizione |                                    
  |---|---|                                                                                                                                                            
  | `--problems P1 P2 …` | esegue solo sui problemi indicati |
  | `--all` | esegue su tutti i 92 problemi unconstrained del filtro |                                                                                                 
  | `--limit N` | limita ai primi N problemi |
  | `--out-dir DIR` | cartella di output alternativa (default `results/`) |                                                                                            
  | `--max-iter-{gd,newton,bfgs} N` | budget iterazioni personalizzato |
                                                                                                                                                                       
  ---                                                       
                                                                                                                                                                       
  ## Problemi selezionati                                   

  La scelta dei 15 problemi è motivata in sezione 3.2 della relazione ed è                                                                                             
  finalizzata a coprire (i) diverse strutture funzionali, (ii) dimensioni
  da `n = 2` a `n = 10⁴`, (iii) regimi di condizionamento da `κ ≈ 1` a                                                                                                 
  `κ ≈ 10¹²`.                                                                                                                                                          
                                                                                                                                                                       
  `ROSENBR`, `ALLINITU`, `BEALE`, `BOOTH`, `BOX3`, `DENSCHNB`,                                                                                                         
  `DIXMAANA1`, `ENGVAL1`, `FREUROTH`, `HUMPS`, `MARATOSB`, `NONDQUAR`,
  `SCHMVETT`, `TRIDIA`, `ZANGWIL3`                                                                                                                                     
                                                            
  ---                                                                                                                                                                  
                                                            
  ## Formato dei file di output

  **`results/benchmark.csv`** — una riga per ciascuna coppia (problema, solver):                                                                                       
   
  ```text                                                                                                                                                              
  problem, solver, n, iterations, runtime_s,                
  f_final, grad_norm, cond_hessian, converged                                                                                                                          
  ```
                                                                                                                                                                       
  **`results/histories.pkl`** — dizionario `{problema: {solver: {f_history,                                                                                            
  grad_history, cond_hessian, iterations, converged}}}`, consumato da
  `plots.py` per generare la Figura 3.                                                                                                                                 
                                                                                                                                                                       
  **`figures/`** — le tre figure incluse nella relazione, in scala
  logaritmica come richiesto dalla consegna:                                                                                                                           
                                                            
  1. `plot1_cond_vs_iters.png` — iterazioni vs `κ(∇²f(x*))`
  2. `plot2_n_vs_runtime.png` — runtime vs dimensione `n`                                                                                                              
  3. `plot3_convergence.png` — decrescita `f(x_k) − f*` per ROSENBR                                                                                                    
     (`n = 2`), DIXMAANA1 (`n = 3`) e TRIDIA (`n = 10⁴`), con                                                                                                          
     annotazione del condizionamento finale per ciascun solver
