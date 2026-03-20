# RAG-Pipeline

## Deine Aufgabe

Baue eine einfache RAG-Pipeline (Retrieval-Augmented Generation), die die Dokumente im Ordner `data/` einliest, in eine Vektordatenbank indexiert und damit Fragen zu diesen Dokumenten beantworten kann. Bei einer Anfrage soll die Pipeline relevante Inhalte aus den Dokumenten finden und daraus mit Hilfe eines LLMs eine Antwort erzeugen.

Das Ergebnis soll ein einfacher Python-Einstiegspunkt oder eine kleine CLI sein, die eine Frage entgegennimmt und eine Antwort auf Basis der bereitgestellten Dokumente ausgibt. Ein Frontend, eine Web-App oder eine grafische Oberfläche sind nicht erforderlich.

Am Ende dieses Dokuments stehen **Beispielfragen** zur Orientierung. Es gibt keine Bewertung, ob diese Fragen richtig beantwortet wurden.

> **Wichtig:** Bitte beachte die [Hinweise zur Verwendung der kostenlosen OpenRouter-Modelle](#llm-zugriff--openrouter) im Setup.

## Einreichung

Stelle uns deinen fertigen Code so bereit, dass wir ihn einsehen können: Entweder machst du dein GitHub-Repository **öffentlich (public)**, oder du fügst uns als **Collaborator** zu deinem Repository hinzu. Bitte **keine** Commits oder Pushes ins ursprüngliche Vorlagen-Repository.

## Technischer Rahmen

Die konkrete Umsetzung ist dir überlassen. An folgendem Setup kannst du dich orientieren:

- **LangChain:** Die Verwendung ist ausdrücklich erwünscht, aber nicht zwingend. LangChain kann insbesondere beim Laden der Dokumente, beim Chunking, bei Embeddings sowie beim Aufbau der Retrieval- und LLM-Pipeline hilfreich sein.
- **Vektordatenbank:** Frei wählbar. Eine lokale Lösung ist ausreichend, zum Beispiel Chroma.
- **LLM:** Die Nutzung von OpenRouter ist ausdrücklich erwünscht, da dort einige Modelle kostenlos verfügbar sind und sich die Aufgabe damit ohne zusätzliche Kosten umsetzen lässt. Andere Anbieter oder Modelle können aber ebenfalls verwendet werden.

## Erwartete Umsetzung

Die Lösung sollte mindestens folgende Punkte abdecken:

1. Dokumente aus dem Ordner `data/` einlesen
2. Inhalte sinnvoll in Chunks aufteilen
3. Die Chunks per Embeddings in einer Vektordatenbank speichern
4. Bei einer Frage relevante Chunks abrufen
5. Die gefundenen Inhalte zusammen mit einem LLM zur Antwortgenerierung verwenden

## Was im Repository enthalten ist

### `main.py`

`main.py` enthält ein minimales Beispiel für die Anbindung eines LLMs über OpenRouter mit LangChain.

Nach dem Kopieren von `.env.example` nach `.env` und dem Eintragen des API-Keys sollte sich das Beispiel über `uv run main.py` beziehungsweise `python main.py` ausführen lassen.

### `data/`

Im Ordner `data/` befinden sich die Dokumente, die von deiner Pipeline verarbeitet werden sollen.

---

## Setup

### Repository klonen

Klone dieses Repository auf deinen Rechner (oder forke es auf GitHub in dein eigenes Konto und klone deinen Fork). Arbeite in **deiner** Kopie. Bitte **keine** Commits oder Pushes ins ursprüngliche Vorlagen-Repository.

```bash
git clone https://github.com/lukasflaig/rag-pipeline-exercise.git
cd rag-pipeline-exercise
```

### Voraussetzungen

- Python 3.13 oder höher
- `uv` (empfohlen) oder alternativ `pip`

### Abhängigkeiten installieren

#### Mit `uv`

```bash
uv sync
```

Dadurch werden die Abhängigkeiten aus dem `pyproject.toml` in einer virtuellen Umgebung installiert.

#### Mit `pip`

```bash
pip install -r requirements.txt
```

### LLM-Zugriff / OpenRouter

Wenn du OpenRouter verwendest:

1. Konto bei [OpenRouter](https://openrouter.ai/settings/keys) anlegen und API-Key erzeugen
2. `.env.example` nach `.env` kopieren
3. API-Key in der `.env`-Datei bei `OPENROUTER_API_KEY` eintragen

Falls du „No endpoints available matching your guardrail restrictions“ siehst: [Privacy-Einstellungen](https://openrouter.ai/settings/privacy) prüfen und ggf. Data-Collection-Einschränkungen lockern.

**Achtung:** Da die Modelle kostenlos sind, kann es bei einigen zu sehr langen Ladezeiten kommen (Überlastung). In Tests haben `openai/gpt-oss-20b:free` und `openrouter/free` funktioniert. `openrouter/free` ist ein Router, der automatisch ein verfügbares kostenloses Modell wählt. Wenn etwas nicht geht, verschiedene Free-Modelle ausprobieren.

Kostenlose Modelle: [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free). Beispiele:

- `openrouter/free` – Router, wählt automatisch ein kostenloses Modell
- `openai/gpt-oss-20b:free` – Open-Source-Modell von OpenAI
- ...

---

## Projektstruktur

```text
rag-pipeline-exercise/
├── README.md
├── main.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── data/
    ├── ...
```

---

## Beispielfragen

Beispiele für mögliche Nutzerfragen:

- Was ist FastAPI, und wofür eignet es sich typischerweise? (`What is FastAPI?.docx`)
- Welche drei RAG-Paradigmen werden unterschieden (`Retrieval-Augmented Generation for Large Language Models - A Survey.pdf`)
- Was ist Azure AI Foundry? (`Getting Started with Azure AI Foundry.docx`)
- Welche Dokumentformate kann Docling verarbeiten? (`Docling - An Efficient Open-Source Toolkit for AI-driven Document Conversion.pdf`)

