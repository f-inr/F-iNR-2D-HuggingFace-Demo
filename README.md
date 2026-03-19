---
title: F-INR 2D Image Demo
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10
app_file: app.py
pinned: false
---

# F-INR 2D Image Demo - HuggingFace Demo

This folder contains a self-sustained Gradio demo for the Tensor Decomposition (FINR) image reconstruction project.

## Project Structure

- `app.py`: The Gradio application entry point.
- `model.py`: JAX/Flax implementation of the F-INR (CP) and Baseline INR models.
- `utils.py`: Utility functions for data generation and loss calculation.
- `requirements.txt`: Python dependencies.

## How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the application:

    ```bash
    python app.py
    ```

3. Open the local URL in your browser.

## Features

- Supports CP (Tensor Decomposition) and Baseline (MLP) architectures.
- Choice of backends: SIREN, ReLU, WIRE, FINER.
- Configurable rank and training epochs.
- Real-time training progress and image reconstruction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Cite

```bibtex
@InProceedings{Vemuri_2026_WACV,
      author = {Vemuri, Sai Karthikeya and B\"uchner, Tim and Denzler, Joachim},
      title = {F-INR: Functional Tensor Decomposition for Implicit Neural Representations},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      month = {March},
      year = {2026},
      pages = {6557-6568}
}
```
