# FantaMantra Portfolio

Assistente live per asta Fantacalcio **Mantra**, pensato di default per **10 partecipanti / 500 crediti**.

## Filosofia

La rosa è trattata come un portafoglio: projection, rischio/upside, vincoli Mantra, budget, diversificazione per club e ri-ottimizzazione dopo ogni vendita.

Durante l'asta inserisci solo:

1. giocatore chiamato;
2. prezzo finale;
3. `MIO` oppure `ALTRI`.

Dopo ogni vendita l'app aggiorna pool residuo, inflazione per ruolo, scarsità, liquidità, portafogli ottimali, exposure, max bid e suggerimenti su chi chiamare.

## Dati

La V1 include direttamente in `data/` il **listone ufficiale Fantacalcio 2026/27** fornito dall'utente:

- `RM` → ruoli Mantra;
- `Nome` e `Squadra`;
- `Qt.A M` / `Qt.I M` → quotazioni Mantra;
- `FVM M` → FVM Mantra.

Questo Excel è la **fonte primaria** dell'anagrafica e dei valori Mantra. L'app rileva automaticamente la vera riga di intestazione del workbook ufficiale (sotto il titolo del foglio `Tutti`) e usa esplicitamente le colonne Mantra, non quelle Classic.

Le pagine pubbliche Fantacalcio vengono usate solo come arricchimento best-effort per le statistiche 2026/27, 2025/26 e 2024/25. Se Internet o il parsing non sono disponibili, l'app continua a funzionare usando il listone incluso e l'FVM come prior.

In **Dati & setup** puoi inoltre caricare in futuro un nuovo Excel/CSV ufficiale per sostituire il listone incluso senza cambiare codice. Il dataset elaborato viene poi salvato su Firestore, così durante l'asta non dipendi dal sito.

## Deploy nel progetto `dani-lab`

Apri Cloud Shell, carica/scompatta il progetto e lancia:

```bash
cd fantamantra
chmod +x deploy.sh
./deploy.sh
```

Default:

- project: `dani-lab`
- region: `europe-west8` (Milano)
- service: `fantamantra`
- Firestore: `(default)` in Native mode

Puoi proteggere l'app con un PIN applicativo prima del deploy:

```bash
export APP_PIN='scegli-un-pin'
./deploy.sh
```

Il servizio Cloud Run resta pubblico a livello HTTP, ma se `APP_PIN` è valorizzato l'interfaccia richiede il PIN. Per un'app personale senza dati sensibili è la soluzione più semplice per una serata d'asta.

## Esecuzione locale

```bash
./run_local.sh
```

oppure:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modello V1

- **Projection**: storico 25/26 + 24/25, evidenza 26/27 con shrinkage, FVM come prior, team momentum con shrinkage.
- **Risk/upside**: proxy derivati da continuità storica, campione disponibile e segnale corrente.
- **Market**: `FVM * 500 / 1000` come prezzo base, poi posterior di inflazione per ruolo, scarsità e liquidità aggregata.
- **Portfolio**: mixed-integer optimization con `scipy.optimize.milp` e coperture Mantra aggregate.
- **Exposure**: più ottimizzazioni con perturbazioni delle proiezioni.
- **Advisor**: fair value + fit + exposure → max bid dinamico e next call.

### Limite importante della V1

Non tracciamo a chi, tra gli altri nove, è stato venduto un giocatore. Questo rende l'input rapidissimo ma significa che il modello conosce la liquidità **aggregata**, non i budget/ruoli residui di ogni avversario.

## Backup e recovery

Su Cloud Run gli eventi sono salvati in Firestore. Refresh, chiusura browser o sostituzione dell'istanza non cancellano l'asta. Puoi anche annullare l'ultima vendita dalla UI.
