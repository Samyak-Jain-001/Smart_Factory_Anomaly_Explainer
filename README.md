# Smart Factory Anomaly Explainer

Minimal setup and config to run the project.

---

## 1. Dataset: How to Download & Place It

This project uses the **“Condition monitoring of hydraulic systems”** dataset.

1. Download the dataset archive from its original public source (search for  
   `"Condition monitoring of hydraulic systems dataset"`).
2. Extract all `.txt` files into:

   ```text
   data/condition+monitoring+of+hydraulic+systems/


So the folder contains:

* `PS1.txt`, `PS2.txt`, `PS3.txt`, `PS4.txt`, `PS5.txt`, `PS6.txt`
* `EPS1.txt`, `FS1.txt`, `FS2.txt`
* `TS1.txt`, `TS2.txt`, `TS3.txt`, `TS4.txt`
* `VS1.txt`, `CE.txt`, `CP.txt`, `SE.txt`
* `profile.txt`

3. Make sure this folder is **not tracked by git** (add to `.gitignore`):

   ```gitignore
   data/condition+monitoring+of+hydraulic+systems/
   ```

If `HYDRAULIC_DATA_DIR` is not set (see below), the code assumes this default path.

---

## 2. Environment Variables

### Core

* `HYDRAULIC_DATA_DIR`
  Path to the folder containing the hydraulic dataset `.txt` files.

  ```bash
  HYDRAULIC_DATA_DIR=/path/to/data/condition+monitoring+of+hydraulic+systems
  ```

  If not set, defaults to:

  ```text
  <project_root>/data/condition+monitoring+of+hydraulic+systems
  ```

* `OLLAMA_HOST`
  URL of the Ollama server.

  ```bash
  # default if not set
  OLLAMA_HOST=http://localhost:11434
  ```

* `OLLAMA_MODEL`
  Name of the Ollama model to use, e.g.:

  ```bash
  OLLAMA_MODEL=llama3
  ```

---

## 3. Local Run

```bash
# example (PowerShell)
set HYDRAULIC_DATA_DIR=C:\path\to\data\condition+monitoring+of+hydraulic+systems
set OLLAMA_HOST=http://localhost:11434
set OLLAMA_MODEL=llama3

python smart_factory_anomaly_explainer.py
```

---

## 4. Docker

### Build

```bash
docker build -t smart-factory-anomaly-explainer .
```

### Run (Docker Desktop on Windows/macOS)

```bash
docker run --rm ^
  -e HYDRAULIC_DATA_DIR=/app/data/condition+monitoring+of+hydraulic+systems ^
  -e OLLAMA_HOST=http://host.docker.internal:11434 ^
  -e OLLAMA_MODEL=llama3 ^
  smart-factory-anomaly-explainer
```

### Run (Linux, host network)

```bash
docker run --rm \
  --network=host \
  -e HYDRAULIC_DATA_DIR=/app/data/condition+monitoring+of+hydraulic+systems \
  -e OLLAMA_HOST=http://localhost:11434 \
  -e OLLAMA_MODEL=llama3 \
  smart-factory-anomaly-explainer
```

```
::contentReference[oaicite:0]{index=0}
```
