# CIFAR-10 Image Classification with Convolutional Neural Networks

🇬🇧 English · 🇪🇸 [Versión en español](README.es.md)

Project IV of the Deep Learning module of the **Master in Data Science & AI (Nuclio Digital School, 2026)**.

The goal is to train a Keras convolutional network that classifies CIFAR-10 images with at least **80% test accuracy**, through **10 iterative experiments**. Each experiment introduces one change, justified by what was observed in the previous one.

**Result:** from **60.58%** with the baseline to **87.18%** with the best model (Project 9). The 80% target is reached in Project 5.

![Comparison](figures/comparativa.png)

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32×32 colour images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

- 40,000 training images
- 10,000 validation images (the last ones of the original training set)
- 10,000 test images, used only for the final accuracy

Pixels are normalised to [0, 1].

## Methodology

For each experiment I analysed the loss and accuracy curves: the epoch of the best model, the gap between training and validation, and whether the validation loss keeps decreasing, plateaus or rises. This identifies the main problem and drives the next change.

From Project 1 onwards, these common settings apply:

- Early Stopping on `val_loss` with `patience=15` and `restore_best_weights=True`.
- Maximum of 100 epochs, raised to 200 from Project 6.
- Network rebuilt from scratch in every experiment, with `clear_session()` and a fixed seed (`set_random_seed(42)`).
- `he_uniform` initialisation and `padding='same'` in the convolutional layers.
- Adam optimiser and `sparse_categorical_crossentropy`.

## Results

| P. | Change | Test accuracy | Best epoch / epochs | Parameters |
|---|---|---|---|---|
| 0 | Baseline (assignment code) | 60.58% | 20 / 20 | 263,402 |
| 1 | First VGG block + 128-unit dense layer | 64.98% | 5 / 20 | 1,060,138 |
| 2 | Second VGG block | 69.46% | 9 / 24 | 591,274 |
| 3 | Dropout | 77.83% | 46 / 61 | 591,274 |
| 4 | Remove Dropout from conv blocks | 73.26% | 24 / 39 | 591,274 |
| 5 | Third VGG block | 81.82% | 44 / 59 | 550,570 |
| 6 | Learning rate 0.0005 | 83.12% | 94 / 109 | 550,570 |
| 7 | Data augmentation | 85.81% | 132 / 147 | 550,570 |
| 8 | Batch size 256 | 84.58% | 73 / 88 | 550,570 |
| 9 | 512-unit dense layer | **87.18%** | 191 / 200 | 1,341,226 |
| 10 | Learning rate 0.001 | 85.14% | 67 / 82 | 1,341,226 |

The chain of decisions:

- **P0 → P1.** The baseline is undertrained and 99.5% of its parameters sit in the dense layer. A VGG block is added.
- **P1 → P2.** Early overfitting caused by a huge dense layer. A second VGG block halves the dense layer's parameters.
- **P2 → P3.** Overfitting persists. Dropout is added (+8.37 points, the largest gain).
- **P3 → P4 → P5.** The model plateaus. Removing Dropout from the conv blocks makes it worse, which points to a lack of capacity. A third VGG block crosses 80%.
- **P5 → P6.** The validation loss oscillates. The learning rate is lowered to 0.0005.
- **P6 → P7.** The validation loss is almost twice the training loss. Data augmentation is added (horizontal flips and 10% shifts).
- **P7 → P8 → P9.** The model learns slowly and training accuracy stays at 85%. A batch size of 256 does not help; widening the dense layer to 512 units gives the best result.
- **P9 → P10.** Going back to a 0.001 learning rate speeds up learning, but the final model is worse. Each learning rate works best at a different stage of training.

### Per-class accuracy (Project 10 model)

![Per-class accuracy](figures/clases.png)

Vehicles are almost always recognised: automobile 95.7% and ship 93.3%. Errors concentrate on animals, especially cat (66.7%) and bird (75.3%).

## Repository structure

```
├── notebooks/Proyecto_IV_CIFAR10.ipynb   # Full code for the 11 experiments (Google Colab)
├── results/resultados_iteracion.json     # Test accuracy, training histories and parameter counts
└── figures/                              # Comparison, per-class accuracy and per-experiment curves
```

The notebook and code comments are in Spanish.

## How to run

1. Open the notebook in Google Colab with a GPU (T4).
2. Download [`cifar-10-python.tar.gz`](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and upload it to your Google Drive as `cifar-10-batches-py.tar.gz`. Then adjust the `! cp` path and the `ruta_googledrive` variable.
3. Run the cells in order. From Project 7 (data augmentation) each training run takes about an hour.

The notebook is published without outputs. The results of each run are in `results/` and the curves in `figures/curvas/`.

## Limitations and next steps

- Each experiment was trained only once, so differences of about one point may be due to chance.
- The best model (Project 9) reached the 200-epoch limit before Early Stopping triggered, so it probably had room for improvement.
- Next steps:
  - Retrain Project 9 with more epochs.
  - Combine both learning rates with a `LearningRateScheduler`.
  - Compute the confusion matrix.
  - Try transfer learning (VGG16).

## Tech stack

Python · TensorFlow / Keras · NumPy · Matplotlib · Google Colab

---

**Author:** Carmen Candial Garcés
