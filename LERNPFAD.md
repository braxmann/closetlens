# Lernpfad: Digitaler Kleiderschrank

Vom Kleidungsfoto bis zum Outfit am eigenen 3D-Avatar, in Stufen, die jede für sich vorzeigbar sind.
Vorgänger und Lernarchiv: `VisionLLMmini` (YOLO-Kaskade, Pose, MediaPipe-Hände, VLM per Ollama, Threads).

## So arbeiten wir

Jeder Schritt läuft so ab: **Lesen** (mit Leitfrage) → **Vorhersage** (im Kopf) → **Teilaufgabe** (du schreibst) →
**Review** (Rückfragen, abgestufte Hinweise) → **Experiment**.
Wird eine Phase übersprungen, gibt es einen Hinweis.

Vor jeder größeren Stufe recherchieren wir kurz den aktuellen Stand, den VRAM-Bedarf und die Lizenzen, weil sich das Feld schnell bewegt.

## Rahmen

- **Hardware:** Der Windows-PC (NVIDIA, 8 GB VRAM) ist der Server im Heimnetz. MacBook (Apple Silicon) und Smartphone sind Clients.
  Entwickelt wird auf beiden.
- **Modelle:** lokal bevorzugt. Cloud-GPUs und Ollama-Cloud nur für Fotos von mir, **nie** für Fotos anderer Personen.
  Biometrische Identifikation nur bei mir selbst.
- **Kein Login**, nur im Heimnetz. Fernzugriff später eventuell über Tailscale.
- **Architektur:** siehe [README.md](README.md). Modelle stecken hinter Schnittstellen; API, Jobs, Speicher und Frontend sind getrennt.
- **Tests** mit pytest ab M2.
- **Sprache:** Code auf Englisch (Namen, Kommentare, Docstrings, Attributwerte), Doku und Lernpfad auf Deutsch.

---

## Übersicht

| Stufe | Meilenstein | Ergebnis |
|---|---|---|
| 1 Kleiderschrank | **M1** Freistellen + Beschreiben | `analyze.py foto.jpg` → PNG + JSON |
| | **M2** Backend + Jobs + Tests | Upload per API, Attribute landen in SQLite |
| | **M3** Mobile Web-Frontend (Retro) | Am Handy fotografieren → Teil erscheint im Schrank |
| 2 Training + Data Science | **M4** Testdatensatz + VLM messen | Genauigkeit pro Attribut, die Messlatte |
| | **M5** Embeddings | Ähnliche Teile, Textsuche, Zero-Shot |
| | **M6** Eigener Klassifikator | Eigenes Modell vs. VLM auf demselben Testset |
| | **M7** Verkleinern | Genauigkeit vs. Größe vs. Latenz |
| | **M8** Outfits + Auswertungen | Vorschläge, „was ziehe ich nie an“ |
| 3 2D-Anprobe | **M9** Try-on auf meinem Foto | Outfit wählen → Bild von mir darin |
| 4 Avatar | **M10** 3D-Avatar im Browser | drehbarer Avatar *(grob, nach Recherche)* |
| 5 Outfit am Avatar | **M11** Outfit am Avatar | 2D-Anprobe → Avatar *(grob, nach Recherche)* |

Laufend, nach jeder Stufe: README, Architekturdiagramm und Demo-GIF ergänzen.
Die Auswertung aus M4, M6 und M7 wird der Kern des Portfolios.

---

## Stufe 1 – Kleiderschrank

### M1 – Foto freistellen und beschreiben (1–2 Sessions)

**Lernziel:** das erste lauffähige Stück der Pipeline, von Anfang an hinter Schnittstellen.
**Neu:** Segmentierung vs. Matting, Alphakanal, `typing.Protocol`, eine feste Taxonomie im Schema.
**Fertig, wenn:** Aus 5 eigenen Kleidungsfotos werden 5 freigestellte PNGs und 5 JSON-Dateien mit plausiblen Attributen.

#### Schritt 0 – Einrichtung
Venv anlegen und `requirements.txt` installieren (siehe README). Fotografier 5 Kleidungsstücke:
eins flach auf dem Boden, eins auf dem Bügel, eins mit Muster, eins dunkel auf dunklem Grund und eins, wie du willst.
Leg die Fotos nach `data/photos/`. Der Ordner ist in der `.gitignore`.

#### Schritt 1 – Freistellen ausprobieren, noch ohne eigenen Code
Lesen: die README von [rembg](https://github.com/danielgatis/rembg), nur die Abschnitte zu CLI, Python-Nutzung und Modellen.
- Leitfragen: Was ist der Unterschied zwischen **Segmentierung** (jedes Pixel bekommt eine Klasse) und **Matting**
  (jedes Pixel bekommt eine Deckkraft von 0–1)? Warum ist das bei Haaren, Fransen oder Strick wichtig?
- Vorhersage: Welches deiner 5 Fotos wird am schlechtesten freigestellt?
- Aufgabe: Stell dasselbe Foto mit der CLI und 3 Modellen frei, z. B. `u2net`, `isnet-general-use` und `birefnet-general`.
  Vergleich Ergebnis und Zeit.
- Experiment: `u2net_cloth_seg` ist auf Kleidung spezialisiert. Was macht es anders?

#### Schritt 2 – Schnittstelle `BackgroundRemover`
Lesen: [typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol), nur die Einleitung und das erste Beispiel.
- Leitfrage: Worin unterscheidet sich ein `Protocol` von einer Basisklasse mit Vererbung? Warum ist das für Fakes in Tests praktisch?
- Aufgabe: In `wardrobe/models/` ein Protocol `BackgroundRemover` mit **einer** Methode (Eingabe: Bild, Ausgabe: Bild mit Alphakanal)
  und eine Implementierung mit rembg. Das Modell wird **einmal** geladen, nicht bei jedem Aufruf.
  Tipp: `new_session` in der rembg-README.

#### Schritt 3 – Taxonomie und Schema
Die wichtigste Design-Entscheidung von M1, denn auf diesem Schema bauen M4 und M6 auf.
- Leitfrage: Warum sind feste Werte (`Literal["t-shirt", "shirt", ...]`) besser als freier Text, sobald du später **messen** und **trainieren** willst?
- Aufgabe: Ein pydantic-Schema `GarmentAttributes` mit `category`, `main_color` und `pattern`, dazu optional
  `material` und `formality`. Leg die Werte bewusst fest. Wenige, klar trennbare Klassen sind besser als viele unscharfe.
- Erinnerung aus VisionLLMmini: Die Reihenfolge der Felder zählt, und Prompt und Schema müssen dieselben Namen verwenden.

#### Schritt 4 – Schnittstelle `AttributeRecognizer` + VLM-Implementierung
Gleiches Muster wie `frage_vlm` aus `vlm_hand.py`, aber jetzt hinter einem Protocol.
- Vorhersage: Was passiert, wenn du ein freigestelltes PNG mit transparentem Hintergrund als **JPEG** ans VLM schickst?
- Aufgabe: Protocol plus Ollama-Implementierung, mit dem Modellnamen als Parameter.
- Experiment: Originalfoto vs. freigestellt auf weißem vs. freigestellt auf grauem Hintergrund. Welche Variante beschreibt das VLM am besten?

#### Schritt 5 – `scripts/analyze.py`
- Lesen: [argparse-Tutorial](https://docs.python.org/3/howto/argparse.html), nur die ersten Abschnitte.
- Aufgabe: `analyze.py foto.jpg [weitere.jpg ...]` stellt frei, beschreibt und schreibt `output/<name>.png` und `output/<name>.json`.
  Das Skript kennt nur die Schnittstellen. Welche Implementierung benutzt wird, entscheidet **eine** Stelle im Skript.
- Fertig-Check: die 5 Fotos aus Schritt 0. Notier dir, wo das VLM danebenliegt. Das ist dein erster Eindruck für M4.

### M2 – Backend mit Hintergrund-Jobs und Tests (2–3 Sessions)
**Neu:** SQLModel/SQLite, Dateiablage, Dependency Injection in FastAPI, Job-Warteschlange mit Status, pytest mit Fakes, `TestClient`.
**Du baust:** Endpunkte zum Hochladen, Auflisten, Abrufen und Korrigieren eines Teils, und einen GPU-Worker, der Jobs nacheinander abarbeitet.
**Fertig, wenn:** `pytest` durchläuft (mit Fake-Modellen, ohne GPU) und ein Upload per curl nach ein paar Sekunden Attribute in der DB hat.

### M3 – Mobile Web-Frontend im Retro-Look (2 Sessions)
**Neu:** mobile-first CSS, Kamera-Upload im Browser (`<input type="file" accept="image/*" capture>`), Status-Polling, Zugriff im Heimnetz.
**Design:** schlicht gebaut (FastAPI-Templates + htmx oder Vanilla JS), aber mit eigenem Charakter, z. B. Retro-OS, Pixel-Schrift
oder Kaufhauskatalog. Die Richtung legen wir zu Beginn von M3 fest.
**Nachtrag (14.11.2026):** Der Retro-OS-Look (98.css) wurde durch einen hellen, minimalistischen Lookbook-Stil ("Atelier") ersetzt.
Nur `app.css` und die Templates haben sich geändert; Routen und htmx-Logik sind gleich geblieben.
**Du baust:** Galerie, Filter, Detailansicht mit **Korrigieren der Attribute**. Die Korrekturen sind später dein Ground Truth.
**Fertig, wenn:** Du am Handy ein Teil fotografierst und es freigestellt und beschrieben im Schrank erscheint.

## Stufe 2 – Eigenes Training und Data Science

### M4 – Testdatensatz und VLM-Auswertung (1–2 Sessions)
**Neu:** train/val/test-Split, Leakage (alle Fotos desselben Teils gehören in denselben Split), Klassen-Ungleichgewicht,
Macro-F1, Konfusionsmatrix.
**Fertig, wenn:** Es eine Tabelle mit der VLM-Genauigkeit pro Attribut gibt. Sie ist die Messlatte für alles Weitere.

**Erledigt (Nov 2026).** Ergebnisse und Methodik im [README](README.md#auswertung-m4). Was dabei aufgefallen ist:
- T-Shirt → „shirt“ (55 %): zu gut zwei Dritteln Polos (Definitionsfrage), der Rest echte Modellfehler.
- „multicolor“ als Ausweichantwort (Präzision 0,24), Braun mit nur 40 % Recall.
- formality: Das VLM antwortet in 88 % der Fälle `null`.
- Fotos mit mehreren Kleidungsstücken erklären nur wenige Fehler; häufiger sind 3/4-Shorts → „pants“,
  Hoodies/Strickjacken zwischen jacket und sweater und zweifelhafte „Brown“-Labels.

**Offen, als Experiment nach M5:**
1. ~~Validierungsset aus den Trainingsgruppen ziehen.~~ Erledigt in M5 (`scripts/make_validation_set.py`). Prompt-Varianten nur dort testen, nie am Testset.
2. Prompt v2 mit festgelegten Definitionen: Polos zählen als `t-shirt`. `main_color` ist die Grundfarbe des Stoffs,
   Drucke und Akzente gehören in `pattern`, „multicolor“ nur ohne dominierende Farbe.
3. Prompt v2 einfrieren und auf dem Testset vergleichen: 3B gegen ein größeres lokales VLM (Qwen2.5-VL 7B / Qwen3-VL),
   optional ein Cloud-Modell. Vorher klären, ob die Datenschutzregel öffentliche Datensätze ausnimmt.

### M5 – Embeddings: Ähnlichkeit und Textsuche (2 Sessions)
**Neu:** Embedding-Raum, Kosinus-Ähnlichkeit, Normalisierung, Zero-Shot-Klassifikation, Vektorsuche.
**Fertig, wenn:** „Ähnliche Teile“ und eine Suche wie „blaues Leinenhemd“ funktionieren und die Zero-Shot-Zahlen neben den VLM-Zahlen stehen.

**Recherche (Nov 2026):** Drei Modelle, alle Apache-2.0 und unter 1 GB VRAM, werden auf dem M4-Testset verglichen:

| Modell | Rolle |
|---|---|
| SigLIP 2 base (`google/siglip2-base-patch16-384`) | allgemeines Modell, Basislinie; dieselbe Basis wie ZooClaw |
| Marqo-FashionSigLIP (ViT-B/16, 224 px, 768-dim) | Mode-spezifisch |
| ZooClaw-FashionSigLIP2 (SigLIP2-base, 384 px, 768-dim) | Mode-spezifisch, Preprint Juni 2026 |

Leitfrage: Wie viel bringt Mode-spezifisches Training, und wie nah kommt Zero-Shot an das VLM heran?

**Entscheidung:** Verglichen werden SigLIP2 (`hf-hub:timm/ViT-B-16-SigLIP2-384`) und Marqo, beide über open_clip und damit über
denselben `OpenClipEmbedder`. ZooClaw bräuchte einen eigenen Embedder für transformers und bleibt eine Option, falls die
Ergebnisse knapp sind.

- **Testset-Effekt:** Marqo misst auf „KAGL“, und das ist unser Datensatz (Fashion Product Images, Kaggle).
  Die Trainingsdaten sind nicht offengelegt. Gute Marqo-Zahlen auf unserem Testset sind deshalb eher zu optimistisch.
  ZooClaw ist auf einem nicht öffentlichen Katalog trainiert und nicht auf KAGL gemessen.
- **Vergleichszahlen der Hersteller sind selbst gemessen.** Unabhängig (LookBench) zeigt sich nur: Mode-Feintuning schlägt allgemeine Modelle.
- Marqo-FashionSigLIP 2 ist nicht frei verfügbar.
- **Speicherung:** keine Vektordatenbank, ein numpy-Array reicht (30.000 × 768 ≈ 90 MB, Suche = ein Matrixprodukt).

**Teil 1 erledigt (Nov 2026): Zero-Shot.** `OpenClipEmbedder` und `ZeroShotAttributeRecognizer` gebaut, Ergebnisse im
[README](README.md#zero-shot-mit-embeddings-m5). Was dabei aufgefallen ist:
- Zero-Shot schlägt das VLM bei category (93 % gegen 86 %) und main_color (75 % gegen 67 %), bei etwa 100-facher Geschwindigkeit.
- Marqo und SigLIP2 liegen gleichauf. Marqo ist dreimal schneller und wird deshalb der Standard.
- Die Formulierung der Klassentexte zählt: „a photo of a t-shirt“ und „a photo of a shirt“ liegen im Textraum fast
  aufeinander, das bloße Wort trennt sie. Getunt nur auf dem Validierungsset, danach einmal auf dem Testset gemessen.
- Gewinne auf dem Validierungsset schrumpfen auf dem Testset (Farbe: +3 → +1 Punkt). Genau dafür trennt man die Sets.
- Größter verbleibender Kategoriefehler: Pullover → Jacke (20 %).

**Teil 2 erledigt: App-Integration.** Der Worker speichert das Marqo-Embedding des freigestellten Bildes in SQLite
(`embedding` als float32-Bytes, dazu `embedding_model`, weil Vektoren verschiedener Modelle nicht vergleichbar sind).
Die Detailansicht zeigt „Ähnliche Teile“, die Galerie hat eine Textsuche. Brute Force (ein Matrixprodukt) reicht bei
ein paar hundert Teilen; bei Millionen bräuchte man einen Vektorindex wie pgvector/HNSW.
- Deutsche Suchtexte funktionieren: „rotes kariertes Hemd“ findet dasselbe wie „red checked shirt“.
- Die Suche findet nie „nichts“, nur weniger Ähnliches. Deshalb höchstens 12 Treffer.
- Noch offen: Erkenner in der App von VLM auf Zero-Shot umstellen? Zero-Shot ist genauer und schneller, erkennt aber keine Marke.

### M6 – Eigener Attribut-Klassifikator (3 Sessions)
**Neu:** Transfer Learning, Linear Probe auf Embeddings (zuerst!), Feintuning eines kleinen Backbones, Augmentation, Overfitting.
Lizenzen öffentlicher Mode-Datensätze vorher prüfen.
**Fertig, wenn:** Eigenes Modell und VLM auf demselben Testset verglichen sind: Genauigkeit, Latenz, VRAM.
Dein Modell ist eine weitere Implementierung von `AttributeRecognizer`.
**Reihenfolge:** Linear Probe auf dem besten M5-Modell → nur wenn nötig LoRA-Feintuning (passt auf 8 GB) → volles Feintuning eher nicht
(kontrastives Training braucht große Batches). Trainiert wird nur auf den Trainingsgruppen aus `split.parquet`.
Achtung: Training auf Shopfotos verbessert nicht automatisch die Erkennung auf Handyfotos.

### M7 – Verkleinern (1–2 Sessions)
**Neu:** Quantisierung (z. B. int8 mit ONNX Runtime), Distillation (das VLM als Lehrer).
**Fertig, wenn:** Eine Tabelle oder Kurve zeigt, was dich jedes eingesparte MB an Genauigkeit kostet.

### M8 – Outfits und Auswertungen (2 Sessions)
**Neu:** Recommender-Grundlagen (Regeln + Embedding-Kompatibilität), Cold Start, einfache Statistik.
**Fertig, wenn:** Die App 3 Outfits vorschlägt (passend zu Saison und Farben) und eine Statistikseite hat („Getragen“-Protokoll, „was ziehe ich nie an“).

## Stufe 3 – 2D-Anprobe

### M9 – Anprobe auf meinem Foto (2–3 Sessions)
**Neu:** Diffusion (Rauschen → Bild, Conditioning), Human Parsing, VRAM-Management.
**Kandidat:** FASHN VTON 1.5 (Apache-2.0, laut Model Card ca. 8 GB VRAM, also an der Grenze). Die GPU-Warteschlange muss
Ollama entladen, bevor sie die Anprobe startet. Für Fotos von mir ist auch eine Cloud-GPU erlaubt.
**Fertig, wenn:** Ein in der App gewähltes Outfit ein Bild von mir in diesem Outfit ergibt.

## Stufe 4 und 5 – Avatar (grob, wird nach Recherche ausgearbeitet)

### M10 – 3D-Avatar im Browser
- **Weg A:** SMPL-X (parametrisches Körpermodell: Form β, Pose θ), an meine Fotos angepasst, Viewer mit three.js.
  Lizenz: nur nichtkommerzielle Forschung. Für das Portfolio ist das okay, gehört aber in die README.
- **Weg B:** LHM (Einzelbild → animierbarer Gaussian-Splat-Avatar) auf einer gemieteten Cloud-GPU. VRAM-Bedarf vorher prüfen.
- **Neu:** parametrische Körpermodelle, Gaussian Splatting, glTF/3D im Browser.

### M11 – Outfit am Avatar
Zuerst pragmatisch über die Kette 2D-Anprobe (M9) → LHM. Optional danach der Forschungsweg: Schnittmuster (ChatGarment)
und Stoffsimulation in Blender auf einem SMPL-X-Körper.

---

## Ausblick (nicht eingeplant)
Das eigene Zimmer mit Möbeln und echten Maßen in 3D digitalisieren.

Kleidungsfotos per Bildbearbeitungs-Modell glätten (keine Falten, wie perfekt ausgebreitet). Frühestens nach M9, weil es dieselben
Diffusion-Grundlagen braucht. Nur als zusätzliche Anzeige-Version speichern; Erkennung und Messung laufen weiter auf dem echten Foto,
weil generierte Bilder Logos und Muster verändern können.
