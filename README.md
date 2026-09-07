# ⚽ FantaMantra Portfolio

Assistente live decision-support per l'asta del **Fantacalcio Mantra**, progettato per guidare le decisioni in tempo reale secondo principi di **Modern Portfolio Theory**, **Ricerca Operativa (MILP)** e **Statistica Bayesiana**.

Configurazione di riferimento predefinita: **10 partecipanti / 500 crediti / 25 giocatori per rosa** (completamente personalizzabile da interfaccia, incluso il *peso della panchina* λ che governa quanto concentrare il budget sui titolari).

---

## 💡 Filosofia: La Rosa come un Portafoglio di Investimento

Nelle aste tradizionali la maggior parte dei fantallenatori compie errori tipici di finanza comportamentale:
- **Overbidding iniziale** per inseguire i primi nomi chiamati a prezzi gonfiati dall'entusiasmo;
- **Panico di metà asta** quando il budget residuo appare insufficiente per i ruoli scoperti;
- **Mancata copertura tattica** delle posizioni rigide del sistema Mantra (es. carenza di `Ds`, troppe ali o assenza di prime punte `Pc`);
- **Incapacità di ricalcolare istantaneamente il valore relativo** dei giocatori rimasti al variare dell'inflazione di mercato e della liquidità residua della lega.

**FantaMantra Portfolio** risolve questi problemi trattando la rosa come un portafoglio ottimizzato di asset sportivi:
1. Ogni giocatore ha una **distribuzione di rendimento atteso** (Expected Fantasy Points) e di **rischio/volatilità**;
2. Ogni euro speso è soggetto a un vincolo di **ottimizzazione combinatoria** (Knapsack Problem multidimensionale);
3. Dopo ogni singolo colpo battuto all'asta (`giocatore + prezzo + MIO/ALTRI`), l'intero mercato viene **ricalcolato in tempo reale**.

Durante l'asta devi inserire soltanto 3 dati rapidissimi:
1. **Giocatore chiamato**
2. **Prezzo finale battuto**
3. **Acquirente** (`MIO` oppure `ALTRI`)

L'app aggiorna all'istante:
- La tua rosa e il budget residuo reale;
- La liquidità aggregata di tutti gli avversari;
- L'inflazione empirica per singolo ruolo Mantra;
- L'indice di scarsità del talento titolare rimasto;
- Il portafoglio ottimale aggiornato, il **modulo Mantra** verso cui conviene costruire, l'**undici titolare target** e l'**esposizione** di ogni giocatore;
- Il **MAX BID dinamico e matematicamente vincolato** (non superabile per legge di bilancio);
- I **prossimi target consigliati** e le chiamate strategiche di **budget-drain** (chiamare top player costosi non target per far prosciugare i crediti avversari).

---

## 📐 Architettura del Modello Matematico

Il motore analitico è strutturato in 4 livelli rigorosamente integrati:

```
[ Listone Ufficiale Excel + Storico Statistiche ]
                        ↓
            1. PROJECTION ENGINE
  (Historical FM + Team Momentum + Availability Prior + Risk/Upside)
                        ↓
             2. MARKET DYNAMICS
  (Base Price + Bayesian Role Inflation + Scarcity + Liquidity + Fit)
                        ↓
            3. PORTFOLIO OPTIMIZER
  (Best-XI MILP HiGHS: rosa + modulo + undici titolare
   + Monte Carlo su rilassamento LP per la portfolio exposure)
                        ↓
              4. LIVE ADVISOR
  (Dynamic Fair Value + Legal Max Bid + Next Calls + Budget Drain)
```

### 1. Projection Engine: Proiezioni e Bayesian Shrinkage

La proiezione dei punti fantacalcio non si basa su semplici medie storiche, ma integra contrazione bayesiana (*empirical shrinkage*):

- **Fantamedia Proiettata ($FM_{proj}$):**
  - Storico recente ponderato: $70\%$ stagione precedente ($2025/26$) + $30\%$ stagione antecedente ($2024/25$);
  - *Market Prior Anchor*: l'FVM ufficiale (percentile di mercato) converte la stima a priori in un intervallo plausibile ($5.65 - 7.75$ FM base), stabilizzando giocatori con pochi voti;
  - *Team Momentum*: regressione bayesiana della crescita/calo complessivo del club di appartenenza con ~20 partite fittizie di prior strength.
- **Disponibilità e Presenze Attese ($PV_{att}$):**
  - Per i giocatori senza storico consolidato (nuovi acquisti, giovani, seconde linee), la stima delle presenze non assume arbitrariamente che siano titolari: è calcolata tramite funzione sigmoide monotona sull'FVM:
    $$PV_{prior} = \text{clip}\left(1.5 + 30.0 \cdot \frac{\text{FVM}}{\text{FVM} + 35.0}, \, 1.0, \, 34.0\right)$$
    In questo modo un 3° portiere o riserva a FVM = 1 riceve $\approx 2.3$ presenze attese ($\approx 13$ punti attesi), mentre un top player da FVM $\ge 150$ riceve $\approx 26 - 32$ presenze attese.
  - Per chi ha presenze storiche, l'evidenza viene contratta verso il prior con peso dell'85%.
- **Expected Fantasy Points ($FP$):**
  $$FP = FM_{proj} \times PV_{att}$$
- **Rischio e Upside:**
  - *Risk*: funzione dell'esperienza su 60 giornate massime, titolarità nella passata stagione e pedigree di mercato;
  - *Upside*: proxy di asimmetria positiva tra segnale corrente, rank di mercato e volatilità intrinseca.

---

### 2. Market Dynamics: Inflazione e Valutazione Dinamica

- **Prezzo Base Calibrato:**
  Poiché l'FVM ufficiale Fantacalcio è calibrato su base 1000 crediti, il prezzo base di equilibrio per la tua lega (default 500) è:
  $$\text{Base Price} = \max\left(1.0, \, \text{FVM} \cdot \frac{\text{Starting Budget}}{1000}\right)$$
- **Inflazione di Ruolo con Posterior Bayesiano:**
  Per ogni vendita $k$, viene calcolato il log-ratio di spesa:
  $$r_k = \text{clip}\left(\ln\frac{\text{Prezzo}_k}{\text{Base}_k}, \, -0.70, \, +0.70\right)$$
  L'inflazione complessiva di mercato e quella di ciascun ruolo vengono aggiornate tramite contrazione bayesiana rispetto a un prior neutro di forza $M_0 = 5$ osservazioni fittizie:
  $$\hat{\mu}_{role} = \frac{0.35 \cdot \hat{\mu}_{overall} \cdot M_0 + \sum_{i=1}^{n} r_i}{M_0 + n}, \quad \text{Multiplier} = \exp(\hat{\mu}_{role})$$
- **Scarsità di Ruolo Pesata per Qualità:**
  A differenza dei modelli naïf che contano tutte le riserve non draftabili del listone, la scarsità è calcolata esclusivamente sul pool di talento con $\text{FVM} \ge 2$, pesato per $\sqrt{\text{FVM}}$. Man mano che i titolari di un reparto si esauriscono, il moltiplicatore di scarsità sale fino a $+22\%$.
- **Liquidità Aggregata:**
  Monitora il rapporto tra crediti residui totali della lega e slot ancora da acquistare rispetto al valore iniziale:
  $$\text{Liquidity Mult} = \text{clip}\left(\left(\frac{\text{Budget Residuo Lega} / \text{Slot Residui Lega}}{\text{Budget Iniziale} / \text{Slot Iniziali}}\right)^{0.22}, \, 0.88, \, 1.15\right)$$
- **Fit Tattico con Penalità di Saturazione:**
  Valuta l'utilità marginale del giocatore per la tua specifica rosa. Se hai bisogno di un ruolo, il moltiplicatore premia l'acquisto (fino a $+14\%$). Se un reparto è già saturo (es. hai già acquistato i 3 portieri), il fit per ulteriori portieri crolla a $0.15$, azzerando la raccomandazione di acquisto.

---

### 3. Portfolio Optimizer: il Miglior Undici, non la Somma dei 25

Al Fantacalcio si schierano **11 giocatori su 25**. Un ottimizzatore che massimizza la somma degli Expected FP dell'intera rosa attribuisce quindi al 25° rincalzo — che non scenderà mai in campo — lo stesso peso dell'attaccante titolare. Con un budget di 500 crediti da spalmare su 25 slot, quella formulazione conduce sistematicamente a una sola conclusione: **nessun top player è mai conveniente**.

L'effetto era misurabile e severo: sul listone ufficiale 2026/27 l'esposizione di *ogni* giocatore sopra FVM 100 (Malen, Martinez L., Calhanoglu, Hojlund, Thuram, Paz N.) risultava esattamente $0.00$ su tutti gli scenari, e la rosa ottima era composta da 25 giocatori pressoché identici da ~20 crediti.

Il modello attuale ottimizza invece il **miglior undici schierabile**, trattando la panchina come un bene complementare pesato $\lambda$.

#### Formulazione

Tre famiglie di variabili binarie:

| Variabile | Significato |
| --- | --- |
| $x_i$ | il giocatore $i$ fa parte dei 25 |
| $m_f$ | $f$ è il modulo Mantra verso cui si costruisce (**variabile decisionale**, non un'assunzione) |
| $w_{i,g}$ | il giocatore $i$ occupa uno slot del gruppo di requisiti $g$ dell'undici |

$$\max \; \sum_{i=1}^N \Big( \lambda \cdot FP_i - c_r \text{Risk}_i + c_u \text{Upside}_i + c_v \text{Value}_i + \epsilon_i \Big) x_i \; + \sum_{(i,g)} (1-\lambda) \cdot FP_i \cdot w_{i,g}$$

Un **titolare vale i suoi Expected FP pieni**; un rincalzo solo la frazione $\lambda$.

Sotto i vincoli:

1. **Completamento rosa:** $\sum_i x_i = 25$, con $x_i = 1$ fissato per i giocatori **già acquistati** (che sono costo affondato e non consumano budget residuo);
2. **Budget residuo:** $\sum_i \text{PrezzoDinamico}_i \cdot x_i \le \text{Budget Residuo}$, sommato sui soli giocatori ancora disponibili;
3. **Modulo unico:** $\sum_f m_f = 1$;
4. **Portieri esatti:** esattamente $3$ portieri totali in rosa (mai più di 3!);
5. **Copertura Tattica Mantra Integrale** sui 25 (panchina inclusa):
   - Difesa centrale: almeno $4$ difensori centrali `Dc`;
   - Fascia destra: almeno $2$ terzini destri `Dd`;
   - Fascia sinistra: almeno $2$ terzini sinistri `Ds`;
   - Esterni di centrocampo: almeno $2$ esterni `E` o ali `W`;
   - Mediana: almeno $2$ mediani `M`;
   - Regia/Mezzali: almeno $3$ centrocampisti `C`;
   - **Prime Punte di ruolo:** almeno $2$ centravanti `Pc` garantiti;
   - Rifinitura/Attacco: almeno $3$ giocatori tra `W`, `T` e `A`;
6. **Diversificazione reale:** massimo 4 giocatori dello stesso club di Serie A;
7. **Capienza degli slot:** $\sum_i w_{i,g} = \text{count}_g \cdot m_{f(g)}$ — ogni gruppo di requisiti del modulo scelto va riempito esattamente;
8. **Un giocatore, uno slot:** $\sum_g w_{i,g} \le x_i$.

Il vincolo (8) è la parte non negoziabile: è un **matching bipartito** giocatori→slot, **lo stesso** che `engine/tactics.py` usa per il semaforo dei moduli nella pagina *La mia rosa*. Ottimizzatore e interfaccia condividono così un'unica definizione di "undici schierabile". Senza di esso lo stesso giocatore coprirebbe più caselle contemporaneamente e il solutore costruirebbe formazioni impossibili da mandare in campo.

#### Il parametro λ (peso della panchina)

Regolabile da **Dati & setup**, default $\lambda = 0.30$. Misurato sul listone reale a parità di budget:

| $\lambda$ | FP del miglior XI | Prezzo max | Giocatori > 40 cr | Carattere |
| --- | --- | --- | --- | --- |
| — *(vecchio obiettivo)* | 1283 | 25 | 0 | rosa piatta, nessun titolare vero |
| $0.05$ | 1825 | 72 | 7 | aggressivo, panchina fragile |
| **$0.30$** | **1760** | **49** | **7** | **equilibrato (default)** |
| $0.60$ | 1532 | 39 | 0 | prudente, robusto alle assenze |

Il passaggio all'obiettivo best-XI vale **+37% di Expected FP sull'undici titolare** rispetto alla formulazione precedente, a parità di crediti spesi. Sotto $\lambda \approx 0.30$ il guadagno sull'XI si appiattisce mentre la panchina si assottiglia rapidamente: è il punto di equilibrio consigliato.

#### Portfolio Exposure e prestazioni

L'esposizione nasce da perturbazioni Monte Carlo sulle proiezioni ($\epsilon_i \sim \mathcal{N}(0, 0.12 \cdot \sigma_i)$), che misurano in quale frazione degli scenari un giocatore rientra nella rosa target.

Il MILP esatto risolve in $\approx 1.2$ s su 531 giocatori (~5.000 variabili binarie), troppo per rieseguirlo a ogni simulazione. Su questo modello — la cui struttura è dominata dal matching di assegnamento — il **rilassamento LP risulta però intero e coincide con l'ottimo** (verificato: zero variabili frazionarie, stesso valore obiettivo, top-25 identica) risolvendo in $\approx 0.08$ s. Le simulazioni usano quindi l'LP, il MILP resta per la rosa consigliata: `analyze_auction` completa in **2-3 secondi** anche a listone pieno.

---

### Analisi Tattica della Rosa (`engine/tactics.py`)

La pagina *La mia rosa* valuta la rosa contro i **7 moduli ufficiali Mantra** (`4-3-3`, `4-2-3-1`, `3-4-2-1`, `3-5-2`, `4-3-1-2`, `3-4-3`, `4-4-2`).

Per ciascun modulo i gruppi di requisiti vengono espansi negli 11 slot individuali e si calcola il **matching bipartito massimo** giocatori→slot (algoritmo di Kuhn con cammini aumentanti, `_max_assignment`). Un modulo è `playable` solo se tutti e 11 gli slot trovano un titolare distinto.

> Contare i giocatori idonei gruppo per gruppo, in modo indipendente, **non** è sufficiente: una rosa con due soli centrocampisti `C` soddisferebbe contemporaneamente sia `C ×2` sia `M/C ×1` del 4-3-3, risultando erroneamente schierabile con 2 uomini per 3 caselle. Il matching elimina il doppio conteggio.

La pagina restituisce inoltre il **depth chart** per ruolo Mantra e la percentuale di copertura di ogni modulo, pesata per la profondità della rosa.

---

### 4. Live Advisor & Dynamic Max Bid Rigoroso

Durante la chiamata di un calciatore, l'Advisor determina:

1. **Fair Value:** Valore intrinseco dinamico moltiplicato per il fit tattico e il premio di portafoglio;
2. **MAX BID Legale e Assoluto:**
   Nel Fantacalcio ogni slot deve essere obbligatoriamente coperto a un costo minimo di 1 credito. Pertanto, il rilancio massimo consentito su un singolo giocatore è rigorosamente limitato da:
   $$\text{Max Bid Legale} = \max\left(0, \, \text{Budget Residuo} - (\text{Slot Rimanenti} - 1)\right)$$
   L'app non ti consiglierà **MAI** una cifra che ti impedirebbe di completare la rosa. Se la rosa è completa ($0$ slot rimanenti), il `max_bid` è pari a $0$.
3. **Segnale Operativo:**
   - 🟢 **BUY** (Esposizione $\ge 35\%$): perno della rosa ottimale, da comprare con decisione;
   - 🟡 **VALUE** (Esposizione $\ge 12\%$ o ottimo value index): ottimo colpo al prezzo giusto;
   - 🔴 **DISCIPLINA** (Bassa esposizione): giocatore non essenziale nel piano tattico, non rilanciare oltre il tetto;
   - 🚫 **ROSA COMPLETA / OUT OF BUDGET**: avvisi espliciti di sicurezza.
4. **Strategia "Budget Drain":**
   Individua i giocatori con alto costo dinamico che il modello **non** vuole acquistare. Chiamare questi giocatori costringe gli avversari a consumare liquidità preziosa su profili inefficienti.

---

## 🖥️ Interfaccia

| Sezione | Cosa trovi |
| --- | --- |
| **Asta live** | Selettore del giocatore chiamato con **filtri rapidi per reparto e per squadra**, scheda del consiglio (fair value, max bid, exposure, segnale operativo) e registrazione della vendita con **pulsanti quick-bid** (prezzo 1 · al max bid · al fair value). |
| **Chi chiamo?** | Target consigliati, **l'undici verso cui stai costruendo** con il modulo Mantra scelto dall'ottimizzatore e il segno ✅ sui giocatori già tuoi, chiamate di budget-drain. |
| **La mia rosa** | Metriche di rosa e budget, **compatibilità con i 7 moduli ufficiali Mantra** (semaforo + ruoli ancora scoperti) e **depth chart** per ruolo. |
| **Mercato** | Inflazione per ruolo, liquidità della lega e storico delle vendite registrate. |
| **Dati & setup** | Configurazione lega, **slider del peso panchina λ**, scommesse su club e caricamento di un nuovo listone. |
| **Modello** | Spiegazione discorsiva di come ragiona il motore. |

> **Nota operativa.** Il `max_bid` è deliberatamente prudente: resta entro circa $\pm 20\%$ dal prezzo di listino FVM. Serve a impedirti di sforare il budget e a dirti *su chi* concentrare la spesa, non a segnalare affari da pagare il doppio. Su un giocatore dell'undici target, arrivare al tetto consigliato e rilanciare ancora un po' è una scelta ragionevole.

---

## 📂 Dati Inclusi e Gestione Dataset

- **Fonte Primaria Inclusa:**
  La cartella `data/` include direttamente il **listone ufficiale Fantacalcio 2026/27** (`Quotazioni_Fantacalcio_Stagione_2026_27.xlsx`).
  L'applicazione estrae specificamente le colonne Mantra (`RM`, `Qt.A M`, `Qt.I M`, `FVM M`), rilevando in automatico la vera riga di testata nel foglio `Tutti`.
- **Arricchimento Statistico Online:**
  All'avvio, l'app arricchisce automaticamente i giocatori con le statistiche ufficiali delle ultime 3 stagioni di Serie A con parsing avanzato multi-colonna e salvaguardia dei valori decimali.
- **Aggiornamento da Interfaccia:**
  Nella sezione **Dati & setup** puoi caricare in qualsiasi momento un nuovo Excel o CSV ufficiale per sostituire il listone senza toccare una riga di codice.

---

## 🚀 Guida Rapida al Deploy

### Deploy su Google Cloud Run (Progetto `dani-lab-507314`)

Dalla Cloud Shell o dal terminale con credenziali gcloud attive:

```bash
cd fantamantra
chmod +x deploy.sh
./deploy.sh
```

Per proteggere l'applicazione con un PIN riservato:
```bash
export APP_PIN='il-tuo-pin-segreto'
./deploy.sh
```

Il servizio Cloud Run viene configurato con:
- Servizio: `fantamantra`
- Regione: `europe-west8` (Milano)
- Database: Cloud Firestore Native per la persistenza degli eventi
- CPU/Memoria: 1 vCPU / 1 GiB

### Esecuzione Locale

```bash
cd fantamantra
pip install -r requirements.txt
streamlit run app.py
```
oppure con `uv`:
```bash
uv run --with streamlit --with beautifulsoup4 --with lxml --with openpyxl streamlit run app.py
```

---

## 🧪 Test Suite

Il progetto include una suite di test approfondita in `tests/`:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -v
```
oppure con `uv`:
```bash
PYTHONPATH=. uv run --with pytest --with beautifulsoup4 --with lxml --with openpyxl pytest -v
```

La suite valida:
- Parsing numerico con corretta conservazione dei decimali italiani ed europei;
- Normalizzazione e copertura del listone ufficiale Mantra 2026/27;
- Stima non distorta delle presenze per giocatori privi di storico;
- Tetto esatto di 3 portieri nel modello di ottimizzazione;
- Presenza obbligatoria delle prime punte `Pc`;
- Rispetto rigoroso del vincolo di bilancio su `max_bid`;
- **Undici realmente schierabile**: l'ottimizzatore produce 11 titolari distinti, sottoinsieme dei 25, su un modulo che `evaluate_formations` conferma `playable`;
- **Effetto di λ**: un peso panchina più basso concentra effettivamente la spesa sui titolari;
- **Giocatori già acquistati** mantenuti in rosa e giocatori venduti agli avversari esclusi dal pool;
- **Nessun doppio conteggio nei moduli**: rosa incompleta non schierabile, rosa completa schierabile, polivalenti (`A/Pc`, `M/C`) utilizzabili sullo slot che serve;
- Persistenza dello stato e del dataset su storage locale e Firestore.

Le pagine Streamlit sono verificate end-to-end con `streamlit.testing.v1.AppTest`.
