# Herbal Leaf Classification

A pattern-recognition course project that classifies five Indonesian herbal leaf species — **Jambu** (guava), **Kunyit** (turmeric), **Paku** (fern), **Singkong** (cassava), and **Sirih** (betel) — from smartphone photos using classical (non-deep-learning) computer vision features and machine learning. The project combines texture features (GLCM, LBP) with two different shape representations (Fourier Descriptor and geometric shape descriptors) in a *hybrid feature extraction* pipeline, then compares SVM, KNN, and Naive Bayes classifiers across ten feature-combination scenarios to identify the best-performing setup. The dataset (279 self-collected images across the 5 classes, later augmented) is included in the repository, along with a full written report (`laporan_klasifikasi_daun_herbal.pdf`).

This was built for the *Pengenalan Pola* (Pattern Recognition) course at the Faculty of Engineering, Universitas Mataram (FT-UNRAM), by a 3-person group (Kelompok 2).

## Features

- Custom image dataset loader that reads class-labeled folders and resizes every image to 128×128
- Preprocessing pipeline: background removal via `rembg` (U²-Net salient object detection), grayscale conversion, Otsu-style fixed thresholding, and external contour detection
- Train-only data augmentation (horizontal flip, ±15° rotation) applied after the train/test split to avoid data leakage
- Hybrid feature extraction combining:
  - **GLCM** (Gray Level Co-occurrence Matrix) texture features
  - **LBP** (Local Binary Pattern) texture features
  - **Fourier Descriptor** shape features
  - Classic geometric **shape descriptors** (via `regionprops`)
- Ten systematically tested feature-combination scenarios (single features and hybrid combinations)
- Three classifiers trained and compared per scenario: SVM (linear kernel), KNN, and Gaussian Naive Bayes
- Model/scaler/label-encoder persistence with `joblib` for reuse without retraining
- Inference demo on new, unseen photos (the `CITRA BARU/` folder) with per-class confidence scores, using the notebook's selected best combination (Scenario 8 — GLCM + Shape Descriptor — with SVM)

## Tech Stack

- **Python** (Jupyter Notebook)
- **OpenCV** (`cv2`) — image I/O, color conversion, resizing, thresholding, morphology, contour detection
- **NumPy** / **Pandas** — array and tabular data handling
- **scikit-image** — `graycomatrix`/`graycoprops` (GLCM), `local_binary_pattern` (LBP), `regionprops` (shape descriptors)
- **rembg** — automatic background removal (U²-Net)
- **scikit-learn** — `SVC`, `KNeighborsClassifier`, `GaussianNB`, `LabelEncoder`, `StandardScaler`, `train_test_split`
- **Matplotlib** / **Seaborn** — visualization
- **joblib** — model persistence
- **tqdm** — progress bars

## Getting Started

### Prerequisites

No `requirements.txt` is included in the repository; based on the notebook's imports you will need Python 3 with:

```
opencv-python numpy pandas matplotlib seaborn scikit-image scikit-learn rembg joblib tqdm
```

Install them with pip:

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image scikit-learn rembg joblib tqdm
```

### Installation

```bash
git clone https://github.com/unproduktif/herbal-leaf-classification.git
cd herbal-leaf-classification
```

The dataset is already included under `DATASET/`, organized by class folder, so no separate download step is required.

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook klasifikasi_daun_herbal.ipynb
```

The notebook will, in order: load and inspect the dataset, split into train/test (80:20, stratified), remove backgrounds, augment the training set, convert to grayscale, threshold, detect contours, extract all 10 feature scenarios, train and evaluate SVM/KNN/Naive Bayes for each, and finally run inference on the sample images in `CITRA BARU/`.

## Project Structure

```
herbal-leaf-classification/
├── klasifikasi_daun_herbal.ipynb     # main notebook: preprocessing, feature extraction, training, evaluation
├── laporan_klasifikasi_daun_herbal.pdf  # written project report
├── DATASET/                          # labeled training/testing images, one folder per class
│   ├── JAMBU/
│   ├── KUNYIT/
│   ├── PAKU/
│   ├── SINGKONG/
│   └── SIRIH/
└── CITRA BARU/                       # extra unseen images used for the inference demo
```
