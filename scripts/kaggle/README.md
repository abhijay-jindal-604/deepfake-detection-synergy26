# Kaggle T4 timing kit (PHASE5 §5-6, PHASE6 §7)

What it answers, on the same GPU class the organisers use (NVIDIA T4, sm_75):
1. Does our torch build run on a T4? (sm_75 in the compiled arch list + a real FP16 kernel call)
2. Wall time of `python inference_script.py ...` for all 3,000 test images, start to finish (imports, bundle load,
   decode, inference, CSV). Efficiency points: < 5 min = 20. Our budget: <= 2.5 min.
3. Does the CSV pass `validate_predictions.py`, and is a re-run byte-identical?
4. How far is the T4 CSV from the local-GPU CSV (max |Δconfidence|, label flips)?
The T4 CSV (`predictions.csv` from run 1) is the one to submit (PHASE5 §6.2).

Files: `make_kit.py` (local, packs the kit), `t4_run.py` (runs on Kaggle), `t4_timing.ipynb` (the notebook).

## Steps

### A. Local (you, ~5 min)
```bash
cd <repo>
export DF_DATA_ROOT=$PWD/resources/extracted/dataset/hackathon_dataset
# 1. the bundle you want to time (Phase 5): outputs/bundle/model_weights.pt
# 2. local predictions with that bundle (for the cross-GPU diff)
.venv/bin/python inference_script.py --test_dir $DF_DATA_ROOT/test/images \
    --weights outputs/bundle/model_weights.pt --output outputs/predictions_local.csv
# 3. pack the kit (folder + zip, ~60 MB of images + the bundle)
.venv/bin/python scripts/kaggle/make_kit.py --bundle outputs/bundle/model_weights.pt \
    --local_pred outputs/predictions_local.csv
# -> outputs/kaggle_kit.zip  (sha256 of every file recorded in kit_manifest.json)
```

### B. Kaggle dataset (you, ~5 min)
1. kaggle.com -> **Datasets -> New Dataset**. Title e.g. `dfd-t4-kit`. Drag in `outputs/kaggle_kit.zip`
   (Kaggle unpacks zips; `t4_run.py` finds the kit anywhere under `/kaggle/input`).
2. Visibility: **Private** (it contains the test images; they are only used for inference).
3. Create, then wait until the dataset shows "ready" (processing a 3,000-file dataset takes a minute or two).
4. For a later bundle: on the dataset page -> **New Version** -> upload the new zip (replace all files).

### C. Notebook (you, ~10 min)
1. **Code -> New Notebook -> File -> Import Notebook** -> upload `scripts/kaggle/t4_timing.ipynb`.
2. Right panel: **Session options -> Accelerator: GPU T4 x2**; **Internet: On** (needs a phone-verified account;
   only the install cell uses the network). **Add Input** -> your `dfd-t4-kit` dataset.
3. **Run All**. Cell 2 prints `nvidia-smi` + the kit path; cell 4 builds a Python 3.10 venv with `uv` and installs
   `requirements-infer.txt` (~2-4 min, torch wheel ~1 GB) and prints the arch list (must contain `sm_75`);
   cell 6 runs `t4_run.py`; cell 7 prints the report and offers `t4_results.zip`.
4. Download `t4_results.zip` (link under the last cell, or the Output tab) and drop it in `outputs/kaggle/`.

### D. Read the result
`t4_out/t4_report.md` (also printed at the end) gives:
- GPU name / `sm_75` / driver, torch + CUDA + cuDNN versions, sm check PASS/FAIL;
- a table per run: wall s/min, load s, inference s, img/s, efficiency band (run 1 = cold cache = the honest number);
- validator PASS, re-run byte-identical, `<= 2.5 min` met or not;
- vs local: max/mean |Δconf|, number > 1e-3, label flips (IDs and confidences in `t4_report.json`), Spearman;
- `hard failures: none` -> OK to submit `t4_out/predictions.csv`.
`t4_run.py` exits non-zero on: kit sha256 mismatch, unsupported GPU arch, inference error, validator FAIL,
non-identical re-run. A large cross-GPU diff is informational (FP16 differs between GPU generations), but label
flips far from the threshold would mean a bug: report them.

## If something fails
| symptom | fix |
|---|---|
| `uv venv --python 3.10` cannot download Python | skip the venv: `!pip install -q -r {KIT}/requirements-infer.txt` into Kaggle's own Python and run `!python {KIT}/t4_run.py`; note the Python version in the report (the brief asks for 3.10) |
| pip cannot find `torch==2.11.0+cu128` | Internet off, or the PyTorch index is unreachable: turn Internet on and re-run the install cell |
| `no kernel image is available` / sm check FAIL | the torch build lacks sm_75: pin the newest cu12x build whose `torch._C._cuda_getArchFlags()` lists sm_75 in `requirements-infer.txt`, rebuild the kit, re-run parity (`scripts/parity_test.py`) |
| wall time > 2.5 min | PHASE5 §5 cut order: drop flip TTA -> drop the slowest member -> `--batch_size 128` (re-time) |
| `DataLoader workers failed` warning | small `/dev/shm`; the script already retries with `num_workers=0` (slower; the time reported includes it) |
| re-run not byte-identical | report it; check `inference_run*.log` for warnings (non-finite FP16 logits fall back to FP32) |

## Local smoke test (CPU, no Kaggle)
```bash
.venv/bin/python scripts/kaggle/make_kit.py --bundle <bundle> --local_pred <csv> --limit_images 48 --out /tmp/kit
cd /tmp && unzip -q kit.zip -d kit_in
.venv/bin/python /tmp/kit_in/kit/t4_run.py --kit_dir /tmp/kit_in --out /tmp/t4_out --allow_cpu --batch_size 16
```
