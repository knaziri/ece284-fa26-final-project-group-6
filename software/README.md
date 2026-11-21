# Software

## Trained models

The trained checkpoints are published as GitHub Releases. They aren't stored in git because they're over GitHub's file size limit, so you need to download them separately.

| Model | Notebook | Release tag | Test accuracy |
|---|---|---|---|
| 4-bit VGG16 | `VGGNet_Hardware_Mapping.ipynb` | [`vgg16-4bit-v1`](https://github.com/knaziri/ece284-fa26-final-project-group-6/releases/tag/vgg16-4bit-v1) | ~92% |
| 4-bit VGG16 with a squeezed 8x8 layer | `VGGNet_Squeezed_8x8.ipynb` | [`vgg16-squeeze8-v1`](https://github.com/knaziri/ece284-fa26-final-project-group-6/releases/tag/vgg16-squeeze8-v1) | ~92% |

Both use 4-bit weights and 4-bit unsigned input activations (0–15) for every conv layer, on CIFAR-10.

### Setting up the GitHub CLI

If you haven't set up `gh` yet, install it ([cli.github.com](https://cli.github.com), or `sudo apt install gh` on Ubuntu) and run `gh auth login` once. You can also download from the release pages in a browser instead.

### Use the matching code

The models were trained with code changes that the evaluation needs too:

- `models/quant_layer.py`: the activation quantizer clips at 0 and 1, so every conv layer's input is an unsigned 4-bit integer (0–15).
- Both notebooks: inputs are **not** normalized (only `transforms.ToTensor()`), and the first layer's `act_alpha` is initialized to 1.0.

If you evaluate with the old normalized transforms, accuracy drops a lot even though nothing crashes. To make sure you have the exact code a model was trained with, check out its release tag, for example:

```bash
git fetch --tags
git checkout vgg16-squeeze8-v1
```

(This puts you in "detached HEAD" mode, which is fine for running and evaluating. Run `git switch -c <branch-name>` if you want to make changes from there.)

All checkpoints are full training checkpoints (`state_dict`, `optimizer`, `epoch`, `best_prec`), so they load with the default `strict=True` and can also be used to resume training. On a machine without an NVIDIA GPU, load them with `torch.load(PATH, map_location="cpu")`, since the notebooks currently call `.cuda()`.

`software/result/` is in `.gitignore`, so downloaded checkpoints won't be committed by accident.

## 4-bit VGG16 (`vgg16-4bit-v1`)

### Download

From the `software/` directory:

```bash
gh release download vgg16-4bit-v1 -p "model_best.pth.tar" -D result/VGG16_quant_4bit/
```

Or, in a browser, open the [release page](https://github.com/knaziri/ece284-fa26-final-project-group-6/releases/tag/vgg16-4bit-v1), click `model_best.pth.tar` under **Assets**, and move it to `software/result/VGG16_quant_4bit/model_best.pth.tar`.

### Evaluate

Open `VGGNet_Hardware_Mapping.ipynb` and run cell 0 (model and data setup), then cell 2 (load the checkpoint and test). **Skip cell 1**: it retrains from scratch for 200 epochs and overwrites `result/VGG16_quant_4bit/`. Cell 2 should print about 92% test accuracy.

## 4-bit VGG16 with a squeezed 8x8 layer (`vgg16-squeeze8-v1`)

This model (`VGG16_quant_squeeze()` in `models/vgg_quant.py`) narrows one layer so it maps onto the 8x8 systolic array without tiling:

- `features[24]`: 256 → 8 channels (conv → BN → ReLU)
- `features[27]`: **8 → 8 channels, conv → ReLU with no BatchNorm**. This is the layer mapped onto the array.
- `features[29]`: 8 → 512 channels (conv → BN → ReLU)

### Download

From the `software/` directory:

```bash
gh release download vgg16-squeeze8-v1 -p "model_best.pth.tar" -D result/VGG16_quant_squeeze/
```

Or, in a browser, open the [release page](https://github.com/knaziri/ece284-fa26-final-project-group-6/releases/tag/vgg16-squeeze8-v1), click `model_best.pth.tar` under **Assets**, and move it to `software/result/VGG16_quant_squeeze/model_best.pth.tar`.

Both releases name their file `model_best.pth.tar`, so make sure each one goes into its own folder.

### Evaluate and verify

Open `VGGNet_Squeezed_8x8.ipynb` and run:

1. **Section 1** (setup).
2. **Skip sections 2 and 3.** Section 2 loads the `vgg16-4bit-v1` checkpoint as a starting point for training, and section 3 fine-tunes for 100 epochs and overwrites `result/VGG16_quant_squeeze/`. You only need them to retrain.
3. **Section 4** (evaluate). It should print about 92% test accuracy.
4. **Sections 5–9** (verification). They map `features[27]` onto the 8x8 array, compute `psum_recovered` (including ReLU), and compare it with the prehooked input of `features[29]`. The cell should print a max error below 1e-3 (about 1e-5 is expected) and `PASS`.

To retrain this model, first download the `vgg16-4bit-v1` checkpoint (see above), then run all the sections in order.
