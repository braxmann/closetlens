# Kleiderschrank

Ein digitaler Kleiderschrank: Kleidung fotografieren, automatisch freistellen und beschreiben lassen,
durchsuchen, Outfits vorschlagen – und langfristig Outfits an einem 3D-Avatar anprobieren.

> Status: **M3 – Mobile Web-Frontend im Retro-Look** (siehe [LERNPFAD.md](LERNPFAD.md))

## Architektur

```
 Handy/Browser ──► Frontend (mobile-first, Retro-Look)
                        │ HTTP
                        ▼
                  api/      FastAPI – nur Ein-/Ausgabe
                        │
                  services/ Fachlogik
                   │            │
            storage/         jobs/      eine GPU-Warteschlange
         SQLite + Dateien       │
                          models/   Schnittstellen: BackgroundRemover · AttributeRecognizer ·
                                    Embedder · TryOn · Avatar (+ Fakes für Tests)
```

Jedes Modell steckt hinter einer Schnittstelle (`typing.Protocol`) und ist austauschbar,
ohne dass API oder Datenbank sich ändern.

## Datenschutz

- Alles läuft lokal im Heimnetz, ohne Login.
- Fotos von mir dürfen auf gemietete Cloud-GPUs, Fotos anderer Personen nie.
- Fotos, Datenbank und Modellgewichte sind per `.gitignore` vom Repo ausgeschlossen.

## Einrichtung

```bash
# Mac
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Windows (PowerShell) – in requirements.txt rembg[gpu] statt rembg[cpu] verwenden
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Außerdem: [Ollama](https://ollama.com) mit einem Vision-Modell, z. B. `ollama pull qwen2.5vl:3b`.

## Starten

```bash
.venv/bin/uvicorn wardrobe.api.main:app --host 0.0.0.0 --port 8001
```

- Am Rechner: <http://127.0.0.1:8001/>
- Am Handy im selben WLAN: `http://<IP des Rechners>:8001/` (Mac: `ipconfig getifaddr en0`)
- `--host 0.0.0.0` macht den Server im Heimnetz erreichbar. Ohne diese Option antwortet er nur dem Rechner selbst.
- JSON-API und Testoberfläche: <http://127.0.0.1:8001/docs>
- Tests: `.venv/bin/python -m pytest`
