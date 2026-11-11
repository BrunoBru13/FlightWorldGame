# FlightWorld

Simulatore di volo 3D scritto in Python con [Panda3D](https://www.panda3d.org/), sviluppato da **NovaGame**.

Sei aerei (Cessna 172, Boeing 737-800, Airbus A320, Airbus A380, Concorde, jet privato) e otto grandi
aeroporti (Roma, New York, Milano, Londra, Parigi, Dubai, Los Angeles, Tokyo) su un'isola generata
proceduralmente, con città, passerelle d'imbarco, schianti, multiplayer e un hangar per personalizzare
colori, livree, scritte, motore e peso degli aerei.

## Comandi

Il mouse indica dove vuoi andare (destra/sinistra = virata, alto/basso = salita o discesa).

| Tasto | Azione |
|---|---|
| E | motori |
| W | acceleratore |
| S | freni |
| P | retromarcia |
| G | carrello |
| R | quota fissa |

## Installazione (Windows)

Scarica l'ultimo installer `FlightWorld_Setup_x.y.z.exe` dalla pagina [Releases](../../releases) e avvialo.
Non servono i permessi di amministratore: il gioco si installa nella tua cartella utente.

## Esecuzione dal codice sorgente

Serve Python 3.10 o successivo.

```
pip install -r requirements.txt
python FlightWorldGame.py
```

## Rete e privacy

Il codice multiplayer fa due cose di rete, solo quando crei o ti unisci a una partita:

- contatta `https://api.ipify.org` per conoscere il tuo indirizzo IP pubblico, che serve a comporre il codice da
  dare agli amici;
- prova ad aprire una porta sul router tramite UPnP, per permettere ai giocatori di collegarsi.

Non viene inviato nessun altro dato e non c'è nessuna telemetria. Il gioco salva le scelte dell'hangar nel file
`hangar.json`, accanto all'eseguibile.

## Compilazione

L'installer viene creato automaticamente da GitHub Actions (PyInstaller + Inno Setup) quando si pubblica una
nuova release con un tag del tipo `v1.0.0`. Lo script dell'installer è `installer.iss`.

## Licenza

Vedi il file [LICENSE](LICENSE).
