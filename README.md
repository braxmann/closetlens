# Kleiderschrank

Ein digitaler Kleiderschrank: Kleidung fotografieren, automatisch freistellen und beschreiben lassen,
durchsuchen, Outfits vorschlagen – und langfristig Outfits an einem 3D-Avatar anprobieren.

> Status: **M5 – Embeddings** abgeschlossen (Zero-Shot, ähnliche Teile, Textsuche), als Nächstes M6 (siehe [LERNPFAD.md](LERNPFAD.md))

## Architektur

```
 Handy/Browser ──► Frontend (mobile-first, heller Lookbook-Stil)
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
# PyTorch mit CUDA zuerst, sonst installiert pip die CPU-Version
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
# onnxruntime passend zu PyTorchs CUDA 12.8 (ab 1.27 ist es für CUDA 13 gebaut und fällt still auf die CPU zurück)
.venv\Scripts\python -m pip install "onnxruntime-gpu==1.26.*"
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

## Auswertung (M4)

Wie gut erkennt das lokale VLM (`qwen2.5vl:3b` über Ollama) die Attribute? Gemessen auf einem eingefrorenen Testset
aus dem öffentlichen Datensatz *Fashion Product Images* (Studiofotos eines Online-Shops).

| Attribut | n | Abdeckung | Genauigkeit (95 %-Intervall) | Macro-F1 | Label-Fehler laut Audit (+ strittig) |
|---|---|---|---|---|---|
| category | 831 | 100 % | **86 %** (83–88) | 0,86 | 0 % (+2 %) |
| main_color | 1182 | 100 % | **67 %** (64–69) | 0,69 | 4 % (+14 %) |
| formality | 200 | 12 % | 83 % (64–93) | 0,75 | 0 % (+7 %) |

1,9 s pro Bild auf einer RTX 2070.

**Methodik**
- **Labels:** Die Labels des Datensatzes werden auf die eigene Taxonomie abgebildet ([label_mapping.py](wardrobe/evaluation/label_mapping.py)).
  Unklare Werte wie „Tops“, „Kurtas“ oder „Orange“ werden ausgeschlossen statt auf „other“ gelegt.
- **Gruppen:** Produkte, die sich nur in der Farbe unterscheiden, bilden eine Gruppe (Name ohne Farbwörter). Gruppen liegen
  komplett im Test- oder im Trainingsteil, damit ein späteres eigenes Modell (M6) keine Geschwister der Testbilder sieht.
- **Stichprobe:** pro Attribut ausgeglichen (bis 100 pro Klasse), höchstens ein Produkt pro Gruppe und Klasse, fester Seed.
  Kleine Klassen behalten mindestens die Hälfte ihrer Gruppen fürs Training.
- **`null`-Antworten** zählen nicht als Fehler. Die Abdeckung wird getrennt angegeben (selektive Klassifikation).
- **Label-Audit:** 100 Testbilder, modellgestützt geprüft. Drei Fehler in der Zuordnung wurden danach behoben.

**Ergebnisse im Detail**
- Größter Kategoriefehler: T-Shirts werden zu 55 % als „shirt“ erkannt, meist Polos.
- Bei Farben weicht das VLM oft auf „multicolor“ aus.
- `formality` beantwortet es kaum.

**Einschränkungen**
- Studiofotos statt Handyfotos, oft mit Model, das mehrere Kleidungsstücke trägt.
- Etwa jedes fünfte Farb-Label ist strittig.

**Reproduzieren**
```bash
python -m scripts.prepare_dataset                        # Datensatz laden
python -m scripts.make_test_set                          # Testset einfrieren (Seed 0)
python -m scripts.run_recognizer --model qwen2.5vl:3b    # VLM-Lauf, fortsetzbar
python -m scripts.evaluate --run qwen2.5vl-3b            # Bericht nach data/evaluation/
```

## Zero-Shot mit Embeddings (M5)

Statt das VLM zu fragen, werden Bild und kurze Texte („shirt“, „a photo of a blue colored garment“) in denselben
Vektorraum eingebettet. Gewählt wird der Text mit der höchsten Kosinus-Ähnlichkeit ([zero_shot.py](wardrobe/models/zero_shot.py)).
Gleiches Testset wie in M4:

| Attribut | VLM qwen2.5vl:3b | Marqo-FashionSigLIP | SigLIP2 B/16-384 |
|---|---|---|---|
| category | 86 % (83–88) | **93 %** (91–94) | **93 %** (91–94) |
| main_color | 67 % (64–69) | 75 % (72–77) | 76 % (73–78) |
| formality | 83 % bei 12 % Abdeckung | 77 % bei 100 % | 81 % bei 100 % |
| Zeit pro Bild (RTX 2070) | 1,9 s | **0,01 s** | 0,03 s |

**Methodik**
- **Klassentexte** wurden nur auf einem eigenen Validierungsset getunt (783 Bilder aus den Trainingsgruppen,
  [make_validation_set.py](scripts/make_validation_set.py)) und danach einmal auf dem Testset gemessen.
- Das bloße Wort („t-shirt“) trennt T-Shirt und Hemd viel besser als „a photo of a t-shirt“:
  T-Shirt → Hemd fiel von 36–44 auf 7–8 von 100. Bei der Farbe blieb vom Gewinn auf dem Validierungsset (+3 Punkte)
  auf dem Testset nur etwa 1 Punkt übrig.
- „other“ ist keine wählbare Klasse. Zero-Shot antwortet immer, deshalb ist die Abdeckung bei formality 100 %.
  Es kann dabei „smart“ antworten, das im Testset nicht vorkommt und immer als Fehler zählt.

**Einschränkungen**
- Marqo wurde vermutlich auch auf diesem Datensatz trainiert (Marqo misst selbst auf „KAGL“), die Zahl ist eher zu optimistisch.
- Zero-Shot erkennt keine Marke.

**Reproduzieren**
```bash
python -m scripts.make_validation_set
python -m scripts.run_recognizer --recognizer zero-shot --model hf-hub:Marqo/marqo-fashionSigLIP [--set validation]
python -m scripts.evaluate --run marqo-fashionSigLIP [--set validation]
```
