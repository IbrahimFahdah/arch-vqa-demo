# Plan Reader: a small vision model that reads floor plans

A local demo of **Qwen3-VL-4B fine-tuned to read architectural floor plans**: it finds the doors on a plan and
draws them as boxes, counts them, and answers questions such as room areas, room names or which rooms share a
wall. Everything runs on your own machine. Your drawings are never uploaded anywhere.

The fine-tuning is a small LoRA adapter (66 MB) on top of the official Qwen3-VL-4B model, run in 4-bit with
[llama.cpp](https://github.com/ggml-org/llama.cpp).

## Quick start

```bash
git clone https://github.com/IbrahimFahdah/arch-vqa-demo.git
cd arch-vqa-demo
bash setup.sh          # one time: ~3.4 GB download
bash run.sh            # then open http://localhost:7860
```

In the page, pick an example plan (or upload your own), choose a model, and press **Find doors**, or type a
question and press **Ask**. Stop the app with Ctrl+C.

On Windows, run the commands in **Git Bash** (comes with [Git for Windows](https://git-scm.com/download/win)).

While this repository is private, the model downloads need the [GitHub CLI](https://cli.github.com/) logged in
with an account that has access (`gh auth login`); `setup.sh` uses it automatically when a plain download fails.

## Requirements

- Python 3.8 or newer (standard library only, nothing to `pip install`)
- `curl` and `bash` (built in on macOS and Linux, included in Git Bash on Windows)
- Disk: about 3.5 GB, or 4.5 GB with `--full`
- Speed depends on the hardware:

| Hardware | Time to find doors on one plan |
|---|---|
| NVIDIA GPU with 6 GB or more | about 10-30 seconds |
| Apple Silicon Mac (M1 or later) | probably under a minute (not tested yet) |
| CPU only | several minutes |

`setup.sh` picks the right llama.cpp build: CUDA if `nvidia-smi` works, Metal on Apple Silicon, otherwise CPU.
To force another build (e.g. Vulkan for AMD or Intel GPUs): `LLAMA_BUILD=win-vulkan-x64 bash setup.sh` or
`LLAMA_BUILD=ubuntu-vulkan-x64 bash setup.sh`.

Tested so far on Windows 11 with an NVIDIA GPU. macOS and Linux use the same scripts but have not been tried
yet; please open an issue if something fails.

## Which model?

| Model | Extra download | Doors found on FloorPlanCAD / CubiCasa test plans | Best for |
|---|---|---|---|
| **fpc** (default setup) | 66 MB | 80% / 75% | Questions about the plan; finding doors at lower accuracy |
| **fpcvis** (`bash setup.sh --full`) | +900 MB | 84% / 88% | Finding doors: the most accurate |
| **base** (always included) | none | 28% on FloorPlanCAD | Seeing what the fine-tuning changed |

"Doors found" = share of the labelled doors that got a box, on plans the models never saw in training (4-bit,
greedy decoding). Of the boxes the fine-tuned models draw, 85-92% are on a real door.

Why two models: fpcvis also trained the part of the model that looks at the image. That made it better at doors
but slightly worse at reading text on real drawings, and it needs its own 836 MB vision file. fpc keeps the
original vision part, so it only adds a 66 MB file.

`setup.sh --full` can be run later; it only downloads what is missing. The app lists the models that are installed.

## Examples

`examples/` has plans to try (see [examples/README.md](examples/README.md) for sources and sample questions):

- `synthetic_1` to `synthetic_6`: computer-generated plans, never used in training
- `cubicasa_416`, `cubicasa_3162`, `cubicasa_11962`: real plans from the CubiCasa5K test set. These plans were
  not trained on, but the CubiCasa drawing style was.

## Limitations

- It is a 4-billion-parameter research proof of concept, not a production tool. Check every answer.
- Door finding is the strongest skill. Counting windows and reading plans in styles far from the training data are
  much weaker.
- On real drawings, the fine-tuned models are not better than the base model at reading text (title blocks,
  dimensions, notes); fpcvis is a little worse.
- Large plans with many doors can make the model repeat boxes. The app then asks again with a setting that avoids
  repeats, and warns you if the answer is still cut off.

## How it works

`run.sh` starts `app/app.py`, a small Python web server on `localhost:7860`. It starts llama.cpp's
`llama-server` on `localhost:8080` with the 4-bit base model, a vision file and the adapter, and forwards the
page's questions to it. "Find doors" asks the model for a JSON list of boxes (0-1000 coordinates), which the
page draws on the plan. Both servers listen on localhost only.

Settings (environment variables): `APP_PORT` (7860), `LLAMA_PORT` (8080), `GPU_LAYERS` (99; lower it if the GPU
runs out of memory), `IMAGE_MAX_TOKENS` (1024), `LLAMA_SERVER` (path to your own llama-server).
Server log: `logs/llama-server.log`.

## Training data and credits

The adapters were trained with LoRA ([LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) on question-answer
pairs generated from:

- synthetic floor plans drawn by code, with answers taken from the generator's data
- [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) (CC BY-NC-SA 4.0): Kalervo et al., "CubiCasa5K: A Dataset
  and an Improved Multi-Task Model for Floorplan Image Analysis", SCIA 2019
- [FloorPlanCAD](https://floorplancad.github.io/) (annotations CC BY-NC 4.0): Fan et al., "FloorPlanCAD: A
  Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting", ICCV 2021

Base model: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache 2.0) by the Qwen team.

## Licence

- Code (`app/`, `setup.sh`, `run.sh`): MIT, see [LICENSE](LICENSE).
- Fine-tuned adapters and vision file (release downloads): **CC BY-NC-SA 4.0, non-commercial use only**, because
  of the training data. See [MODEL_LICENSE.md](MODEL_LICENSE.md).
- The base model is downloaded from Qwen's official repository under its own Apache 2.0 licence.
