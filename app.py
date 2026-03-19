# MIT License

# Copyright (c) [2026] [Tim Büchner, Sai Karthikeya Vemuri, Joachim Denzler]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import requests
from io import BytesIO
from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import DecompositionType, EmbeddingType, MLPType, get_model_2D
from utils import img_loss, img_train_generator

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def train_model(img, decomp_type, backend_type, embedding_type, rank, epochs, update_freq, diff_boost):
    img_array = np.array(img).astype(np.float32) / 255.0

    backend = MLPType[backend_type]

    if embedding_type == "None":
        embedding = EmbeddingType.PE000
    elif embedding_type == "Positional":
        embedding = EmbeddingType.PE100
    elif embedding_type == "Hash":
        embedding = EmbeddingType.HE
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    # Forcing CP decomposition as per requirement, while preserving the user dropdown
    decomp = DecompositionType.CP

    features = [256] * 4
    model = get_model_2D(backend=backend, embedding=embedding, decomp=decomp, rank=rank, features=features)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, 2)

    # CP Decomposition (Always used)
    train_data = img_train_generator(img_array)
    x_coords, y_coords, u_channels = train_data
    params = model.init(subkey, x_coords[:1], y_coords[:1])
    optim = optax.adam(0.001)
    apply_fn = jax.jit(model.apply)

    l_fn = img_loss(apply_fn, *train_data)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(l_fn)(params)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    opt_state = optim.init(params)

    # Calculate model size and compression ratio
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    model_size_bytes = num_params * 4
    img_size_bytes = np.array(img).shape[0] * np.array(img).shape[1] * 3
    compression_ratio = img_size_bytes / model_size_bytes if model_size_bytes > 0 else 0

    info_str = f"Original image size: {img_size_bytes / 1024:.2f} KB | Model size: {model_size_bytes / 1024:.2f} KB | Compression ratio: {compression_ratio:.2f}x"
    print(info_str)

    loss_history = []
    for i in range(epochs):
        loss, params, opt_state = train_step(params, opt_state)
        loss_history.append([i, float(loss)])

        if stop_requested:
            break

        if i % update_freq == 0 or i == epochs - 1:
            u = apply_fn(params, x_coords, y_coords)
            ut = jnp.transpose(jnp.array(u), (1, 2, 0)).clip(0, 1)
            recon = (np.array(ut) * 255).astype(np.uint8)
            diff = np.abs(np.array(ut) - img_array) * diff_boost
            diff_img = (np.clip(diff, 0, 1) * 255).astype(np.uint8)
            yield Image.fromarray(recon), Image.fromarray(diff_img), f"Epoch {i}: loss {loss:.4f}\n{info_str}", loss_history

    # Final execution will already cover up to the last state due to the loop


# Load sample image
remote_url = "https://f-inr.github.io/static/images/0886.png"

try:
    response = requests.get(remote_url, timeout=10)
    sample_img = Image.open(BytesIO(response.content))
    print(f"Loaded remote image from {remote_url}")
except Exception as e:
    print(f"Failed to load remote image: {e}. Falling back to local search.")
    image_files = list(Path(__file__).parent.glob("*.png")) + list(Path(__file__).parent.glob("*.jpg"))
    sample_img_path = image_files[0] if image_files else None

    if sample_img_path is None or not sample_img_path.exists():
        sample_img = None
    else:
        sample_img = Image.open(sample_img_path)

with gr.Blocks() as demo:
    gr.HTML("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h1 style="font-weight: 900; font-size: 2.5rem; margin-bottom: 0.5rem">
            F-INR: Functional Tensor Decomposition for Implicit Neural Representations
        </h1>
        <h3 style="font-weight: 400; font-size: 1.2rem; margin-bottom: 0.5rem">
            Sai Karthikeya Vemuri, Tim Büchner, Joachim Denzler
        </h3>
        <p style="font-size: 1.1rem; color: #555; margin-bottom: 1rem">
            Winter Conference on Applications of Computer Vision (WACV 2026)
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem;">
            <a href="https://openaccess.thecvf.com/content/WACV2026/papers/Vemuri_F-INR_Functional_Tensor_Decomposition_for_Implicit_Neural_Representations_WACV_2026_paper.pdf" target="_blank" style="text-decoration: none;">
                <button style="background-color: #333; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    📄 Paper (PDF)
                </button>
            </a>
            <a href="https://f-inr.github.io" target="_blank" style="text-decoration: none;">
                <button style="background-color: #333; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    🌐 Project Page
                </button>
            </a>
            <a href="https://github.com/f-inr/" target="_blank" style="text-decoration: none;">
                <button style="background-color: #333; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    💻 GitHub
                </button>
            </a>
        </div>
        <img src="https://f-inr.github.io/static/images/fig1_overview.png" alt="F-INR Overview" style="max-width: 100%; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <p style="text-align: justify; margin-bottom: 2rem;">
            Implicit Neural Representations (INRs) model signals as continuous, differentiable functions. However, monolithic INRs scale poorly with data dimensionality. <strong>F-INR</strong> factorizes a high-dimensional INR into a set of compact, axis-specific sub-networks using functional tensor decomposition. This demo allows you to train an INR model on a <strong>2D image</strong> (as showcased in our paper) and compare our approach to a standard baseline.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Training Configuration")
            input_img = gr.Image(value=sample_img, type="pil", label="Input Image")

            decomp = gr.Dropdown(choices=["CP"], value="CP", label="Decomposition Type", info="Select 'CP' for F-INR versions of the standard INR models. Only CP available for Image task.")
            with gr.Row():
                backend = gr.Dropdown(choices=["SIREN", "RELU", "WIRE", "FINER"], value="RELU", label="Model Backend")
                embedding = gr.Dropdown(choices=["None", "Positional", "Hash"], value="Positional", label="Input Encoding")

            rank = gr.Dropdown(choices=[2**i for i in range(1, 10)], value=128, label="Tensor Rank", info="Rank for decomposition (only for CP).")

            with gr.Row():
                epochs = gr.Slider(minimum=1000, maximum=50_000, value=15_000, step=1000, label="Epochs")
                update_freq = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="UI Update")

            diff_boost = gr.Slider(minimum=1, maximum=100, value=1, step=1, label="Error Map Boost (Multiplier)")

            with gr.Row():
                btn = gr.Button("🚀 Start Training", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop")

            logs = gr.Textbox(label="Training Status & Compression Info", lines=3)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Reconstruction Results")
            output_img = gr.Image(label="Reconstructed Image", type="pil", interactive=False)
            diff_img_ui = gr.Image(label="Difference Error Map", type="pil", interactive=False)

            loss_plot = gr.LinePlot(label="Training Loss Curve", x="Epoch", y="Loss", y_lim=[0, 0.05])

    stop_event = gr.State(False)

    def run_training(img, decomp, backend, embedding, rank, epochs, update_freq, diff_boost):
        # split loss tuple for pandas dataframe or list of dicts
        import pandas as pd

        global stop_requested
        stop_requested = False

        img_recon, img_diff, text_update, df = None, None, "", pd.DataFrame(columns=["Epoch", "Loss"])

        for img_recon, img_diff, text_update, l_hist in train_model(img, decomp, backend, embedding, int(rank), epochs, update_freq, diff_boost):
            df = pd.DataFrame(l_hist, columns=["Epoch", "Loss"])
            yield img_recon, img_diff, text_update, df
            if stop_requested:
                break

        metrics_text = ""
        if img_recon is not None and img is not None:
            img_arr = np.array(img.resize(img_recon.size)) if img.size != img_recon.size else np.array(img)
            recon_arr = np.array(img_recon)

            psnr_val = peak_signal_noise_ratio(img_arr, recon_arr, data_range=255)
            ssim_val = structural_similarity(img_arr, recon_arr, channel_axis=-1, data_range=255)
            metrics_text = f"\n\n📈 Final Metrics:\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"

        if stop_requested:
            final_text = text_update + "\n\n🛑 TRAINING STOPPED!" + metrics_text
        else:
            final_text = text_update + "\n\n✅ TRAINING COMPLETED!" + metrics_text

        if img_recon is not None:
            yield img_recon, img_diff, final_text, df

    def stop_training():
        global stop_requested
        stop_requested = True

    def recalculate_diff(recon_img, input_img, boost):
        if recon_img is None or input_img is None:
            return None
        recon = np.array(recon_img).astype(np.float32) / 255.0
        orig = np.array(input_img).astype(np.float32) / 255.0
        # If sizes mismatch, return none
        if recon.shape != orig.shape:
            # Resize orig to match
            orig = np.array(Image.fromarray((orig * 255).astype(np.uint8)).resize((recon.shape[1], recon.shape[0]))).astype(np.float32) / 255.0
        diff = np.abs(recon - orig) * boost
        diff_img = (np.clip(diff, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(diff_img)

    btn.click(run_training, inputs=[input_img, decomp, backend, embedding, rank, epochs, update_freq, diff_boost], outputs=[output_img, diff_img_ui, logs, loss_plot])
    stop_btn.click(fn=stop_training, inputs=None, outputs=None)

    diff_boost.change(fn=recalculate_diff, inputs=[output_img, input_img, diff_boost], outputs=[diff_img_ui])

    demo.queue()
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Citrus())  # type: ignore
