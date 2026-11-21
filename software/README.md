# Software

## Trained 4-bit VGG16 model

A trained 4-bit VGG16 checkpoint (**92% CIFAR-10 test accuracy**) is published as a GitHub Release:

https://github.com/knaziri/ece284-fa26-final-project-group-6/releases/tag/vgg16-4bit-v1

The checkpoint is not stored in git (it's 169 MB, over GitHub's file size limit), so you need to download it separately.

### Download

**Option A: GitHub CLI** (recommended)

If you haven't set up `gh` yet, install it ([cli.github.com](https://cli.github.com), or `sudo apt install gh` on Ubuntu) and run `gh auth login` once.

From the `software/` directory:

```bash
gh release download vgg16-4bit-v1 -p "model_best.pth.tar" -D result/VGG16_quant_4bit/
```

**Option B: Browser**

Open the release link above, click `model_best.pth.tar` under **Assets**, and move the file to:

```
software/result/VGG16_quant_4bit/model_best.pth.tar
```

This is the path `VGGNet_Hardware_Mapping.ipynb` loads from. `software/result/` is in `.gitignore`, so the file won't be committed by accident.

### Use the matching code

The model was trained with code changes that the evaluation needs too:

- `models/quant_layer.py`: the activation quantizer clips at 0 and 1, so every conv layer's input is an unsigned 4-bit integer (0–15).
- `VGGNet_Hardware_Mapping.ipynb`: inputs are **not** normalized (only `transforms.ToTensor()`), and the first layer's `act_alpha` is initialized to 1.0.

If you evaluate with the old normalized transforms, accuracy drops a lot even though nothing crashes. To make sure you have the exact code the model was trained with, check out the release tag:

```bash
git fetch --tags
git checkout vgg16-4bit-v1
```

(This puts you in "detached HEAD" mode, which is fine for running and evaluating. Run `git switch -c <branch-name>` if you want to make changes from there.)

### Evaluate

Open `VGGNet_Hardware_Mapping.ipynb` and run cell 0 (model and data setup), then cell 2 (load the checkpoint and test). **Skip cell 1**: it retrains from scratch for 150 epochs and overwrites `result/VGG16_quant_4bit/`. Cell 2 should print about 92% test accuracy.

The checkpoint is a full training checkpoint (`state_dict`, `optimizer`, `epoch`, `best_prec`), so it loads with the default `strict=True` and can also be used to resume training. On a machine without an NVIDIA GPU, load it with `torch.load(PATH, map_location="cpu")`, since the notebook currently calls `.cuda()`.
