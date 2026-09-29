# Batch Background Remover

Fully automated, free, offline batch image background removal using **rembg + U2Net**.

Drop images into a folder → run one command → get transparent PNGs.

---

## Features

| Feature | Details |
|---|---|
| **Offline** | After the first model download (~170 MB), no internet needed |
| **Free** | Open-source model (U2Net, Apache-2.0 license) |
| **Batch** | Processes 500–2,000+ images sequentially |
| **Smart skip** | Skips already-processed images (tracks size + mtime) |
| **Atomic writes** | Writes to temp file, then renames — safe against interruptions |
| **EXIF rotation** | Auto-corrects orientation before processing |
| **Watch mode** | Optional folder monitoring for new images |
| **Error resilient** | Logs failures, continues processing remaining images |

---

## Quick start (Windows)

### 1. Install Python 3.11+

Download from <https://www.python.org/downloads/> and check **"Add Python to PATH"** during installation.

### 2. Open a terminal in this folder

```
cd "E:\Projects\Assets BG"
```

### 3. Create a virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

### 5. Place images

Copy your images into:

```
Assets BG\assets\
```

### 6. Run

```
python main.py
```

Transparent PNGs will appear in `Assets BG\output\`.

---

## Folder structure

```
Assets BG/
├── assets/            ← input images (JPG, PNG, WEBP, BMP, TIFF)
├── output/            ← transparent PNG results
├── models/            ← cached AI model (auto-downloaded on first run)
├── main.py            ← the script
├── requirements.txt
└── README.md
```

---

## Command-line options

| Command | Description |
|---|---|
| `python main.py` | Process all images once |
| `python main.py --force` | Reprocess even if output exists |
| `python main.py --watch` | Monitor folder for new images |
| `python main.py --model u2netp` | Use the lighter/faster model |
| `python main.py --recursive` | Scan subdirectories |
| `python main.py --input "D:\Images"` | Custom input folder |
| `python main.py --output "D:\Processed"` | Custom output folder |
| `python main.py --help` | Show all options |

You can combine flags:

```
python main.py --recursive --force --model u2netp
```

---

## Supported formats

| Input | Output |
|---|---|
| JPG / JPEG | PNG (RGBA) |
| PNG | PNG (RGBA) |
| WEBP | PNG (RGBA) |
| BMP | PNG (RGBA) |
| TIFF / TIF | PNG (RGBA) |

All outputs are saved as transparent PNG with the alpha channel.

---

## How skip / resume works

The script maintains a lightweight manifest (`.processing_manifest.json`) in the output folder that tracks each processed file's size and modification time.

- If an output already exists **and** the source file hasn't changed → **skipped**.
- If the source image is modified → **reprocessed** automatically.
- If the script is interrupted → restart it; only remaining images are processed.
- Use `--force` to override and reprocess everything.

---

## GPU acceleration (optional)

By default the script runs on CPU. To use an NVIDIA GPU:

```
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

The script will auto-detect CUDA and use the GPU. If unavailable, it falls back to CPU.

---

## Error handling

- **Corrupted / unreadable images** → logged to `failed_images.log`, processing continues.
- **Permission errors** → logged, processing continues.
- **Interruptions** → safe; temp files are cleaned up, manifest is persisted.
- All errors are logged with timestamps in `failed_images.log`.

---

## Model information

| Model | Size | Speed | Quality | License |
|---|---|---|---|---|
| `u2net` (default) | ~170 MB | Slower | Higher | Apache-2.0 |
| `u2netp` | ~4 MB | Faster | Good | Apache-2.0 |
| `isnet-general-use` | ~170 MB | Medium | High | Apache-2.0 |
| `silueta` | ~170 MB | Medium | High | Apache-2.0 |

The model is downloaded automatically on the first run and cached in the `models/` folder.

**License**: U2Net is released under the [Apache License 2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE). Free for commercial and personal use. Redistribution must include the license notice.

---

## Performance notes

Processing speed depends on your hardware and image resolution. Rough estimates:

- **CPU (modern i7)**: ~2–5 images/min at full resolution
- **GPU (RTX 3060+)**: ~15–30 images/min

The script processes one image at a time to minimize memory usage. For 2,000 images on CPU, expect several hours.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'rembg'` | Run `pip install -r requirements.txt` |
| Model download fails | Ensure internet access for the first run |
| Out of memory | Use `--model u2netp` for the lighter model, or reduce image resolution |
| Script is slow | Use GPU acceleration (see above) or `u2netp` model |
| Permission denied on output | Ensure the output folder is writable |

---

## License

This script is provided as-is for personal and commercial use. The U2Net model is licensed under Apache-2.0.
