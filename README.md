# MNIST Digit Classifier Cascade

This project trains **10 separate binary classifiers** (one for each digit `0–9`) on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and performs **cascade prediction**:

* Model `0`: predicts "is this digit 0?"
* If **no**, go to Model `1`: predicts "is this digit 1?"
* …
* Until one model predicts **yes**, or fallback to the model with the highest probability.

---

## 🚀 Features:

* Train **individual models per digit** (binary classification).
* Evaluate a **cascade prediction pipeline** across all 10 models.
* Predict your **own handwritten digit images** (JPG/PNG).
* Uses **PyTorch + torchvision** for training and evaluation.
* Includes preprocessing (grayscale, resize, normalization, auto-invert).

---

## 📂 Project Structure:

```
MNIST_Model/
│── digit_cascade.py      # Main script (training & prediction)
│── models/               # Saved models (digit_0.pt, digit_1.pt, ...)
│── data/                 # MNIST dataset (auto-downloaded)
│── my_scan.jpg           # Example custom input image
│── README.md             # Project documentation
```

---

## ⚙️ Installation:

1. Clone this repo:

   ```bash
   git clone https://github.com/Manvith-kumar16/MNIST_Model.git
   cd MNIST_Model
   ```

2. Install dependencies:

   ```bash
   pip install torch torchvision pillow tqdm numpy matplotlib
   ```

---

## 📊 Training:

### Train all 10 digit models

```bash
python digit_cascade.py --train-all --epochs 5 --batch-size 256
```

### Train only one digit (e.g., digit 2)

```bash
python digit_cascade.py --train-digit 2 --epochs 5
```

Models will be saved in the `models/` folder.

---

## 🔍 Testing

### Evaluate cascade on MNIST test set

```bash
python digit_cascade.py --test-cascade --models-dir models --threshold 0.5 --limit 1000
```

* `--limit` = number of test samples (optional, for speed).

---

## 🖼️ Predict Your Own Image:

Place your handwritten digit image in the project folder (e.g., `my_scan.jpg`) and run:

```bash
python digit_cascade.py --predict-file 7.jpg --models-dir models --threshold 0.5
```

Example with a subfolder:

```bash
python digit_cascade.py --predict-file images/4.jpg --models-dir models
```

---

## 📌 Notes:

* Dataset: [MNIST in CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or auto-download via `torchvision.datasets.MNIST`.
* Images must be **single digits**, clear handwriting, and ideally black digit on white background.
* Preprocessing will auto-resize to **28×28** pixels.

---

## ⚡ Future Improvements:

* Train a single **multiclass classifier** (10 outputs) for better performance.
* Add support for **EMNIST** (letters + digits).
* Improve preprocessing (centering, deskewing, noise removal).

---

## 📜 License

This project is open-source under the MIT License.
