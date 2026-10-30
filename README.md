# Kleiderschrank

Ein digitaler Kleiderschrank: Kleidung fotografieren, automatisch freistellen und beschreiben lassen,
durchsuchen, Outfits vorschlagen – und langfristig Outfits an einem 3D-Avatar anprobieren.

> Status: **M1 – Foto freistellen und beschreiben** (siehe [LERNPFAD.md](LERNPFAD.md))

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
