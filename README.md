<div align="center">

# 📸 Edge Auto Photo Enhancement AI

**Autonomous, zero-touch RAW reconstruction & neural color grading for Raspberry Pi 5 & macOS**
image enhancement pipeline engineered for high-fidelity RAW reconstruction and automated color grading.
Designed to run on edge hardware (Raspberry Pi 5) as well as desktop workstations (macOS / Linux),
the system allows photographers to offload heavy post-processing directly from a smartphone or laptop in the field via a private, peer-to-peer network.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow Lite](https://img.shields.io/badge/LiteRT%20%2F%20TFLite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Tailscale](https://img.shields.io/badge/Tailscale-P2P%20WireGuard-2D3748?style=for-the-badge&logo=tailscale&logoColor=white)](https://tailscale.com/)


## 💡 Motivation & Philosophy

As a photographer with over a decade of experience behind the lens, I built this system to eliminate two persistent bottlenecks in the photographic workflow:

1. **The Backlog Problem:** Sifting through hundreds of unedited exposures after a shoot often results in massive archives sitting unprocessed for weeks.
2. **Field Turnaround & Immediate Sharing:** Delivering polished, professional-grade images while still on location usually requires carrying a laptop, card readers, and spending valuable time manually nudging sliders in desktop software.

Generic AI filters often push saturation blindly or introduce destructive edge halos. To overcome this, the models in this pipeline were **custom-trained on my own curated dataset accumulated over 10+ years of shooting**. They replicate a cohesive, balanced editing style—executing both broad global balance (dynamic range recovery, tonal mapping) and nuanced local adjustments (clarity, localized micro-contrast, skin/sky tonal separation) that hold up to strict photographic standards.

---

## ✨ Key Features

* **Wide Format Ingestion:** Native 16-bit decoding for professional cameras and smartphone DNGs (`.arw`, `.cr2`, `.cr3`, `.nef`, `.dng`) along with standard consumer formats (`.jpg`, `.png`, `.tif`).
* **Luminance-Isolated Processing:** Sharpens and denoises exclusively on the luminance channel ($Y$) in YUV space, eliminating chromatic noise and edge discoloration.
* **Seam-Free Tiling Engine:** Slices large 24–45+ MP sensors into overlapping $512 \times 512$ patches and blends them using linear/Gaussian falloff, avoiding OOM (Out Of Memory) crashes on SBCs.
* **Autonomous Edge Deployment:** Runs self-hosted on a Raspberry Pi 5 without cloud fees or telemetry.
* **Field P2P Mobile Ingestion:** Built-in FastAPI PWA connected over Tailscale (WireGuard) for direct, high-speed upload from iOS/Android devices in the field.

---

## ⚙️ Architecture

```text
[Camera / Mobile / Mac]
       │
       │  (Encrypted P2P Upload via Tailscale & Web PWA)
       ▼
[FastAPI Asynchronous Engine] ──► [Background Processing Queue]
                                           │
                                           ▼
                    1. Ingestion & 16-bit Linearization
                       * rawpy (LibRaw) for RAWs | PIL/OpenCV for standard formats
                                           │
                                           ▼
                    2. Color Space Transform (RGB ──► YUV)
                       * Isolates luminance (Y) from chrominance (U, V)
                                           │
                                           ▼
                    3. AI Detail Recovery & Denoising (NAFNet Architecture)
                       * Deployed via LiteRT (TFLite) with XNNPACK vector acceleration
                                           │
                                           ▼
                    4. Overlapping Patch Segmentation & Blend
                       * Prevents out-of-memory errors on high-MP frames
                                           │
                                           ▼
                    5. Style & Color Enhancement Model
                       * Custom-trained grading network for global/local tone tuning
                                           │
                                           ▼
                    6. Export & Compression
                       * High-bitrate JPEG / 16-bit TIFF ready for web delivery
                                           │
       ▲                                   ▼
       │                            [Output Cache]
[Single File Download / Batch ZIP Archive]
