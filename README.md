# 🌍 Land Type Classification using Sentinel-2 Satellite Images

![GitHub repo size](eurosat-overview.png)


## 📌 Project Overview
This project builds a **Deep Learning Computer Vision system** to classify land types using **Sentinel-2 multispectral satellite imagery**.

### Classes:
- 🌾 Annual Crop
- 🌲 Forest
- 🌱 Herbaceous Vegetation
- 🛣 Highway
- 🏭 Industrial
- 🌄 Pasture
- 🌳 Permanent Crop
- 🏠 Residential
- 🌊 River
- 🟦 Sea / Lake


## 🚀 Workflow
![GitHub repo size](workflow.png)

## 📂 Dataset
- Sentinel-2 imagery  
- EuroSAT Dataset  
- Collected data for Egypt 

**Features:**
- RGB + NIR  
- NDVI  


## 🧹 Preprocessing
- Resize & Normalize  
- Data Augmentation:
  - Rotation  
  - Flipping  
  - Cropping  


## 🧠 Models
- CNN (Baseline)  
- EfficientNet (Transfer Learning + Fine tuning)  


## ⚙️ Training
- Loss: Cross-Entropy  
- Optimizers: Adam
- Techniques:
  - Early Stopping  
  - Dropout  
  - BatchNorm  


## 📈 Evaluation
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  


## 🚀 Deployment
- FastAPI / Flask API  
- Input: Satellite Image  
- Output: Land Type  


## 📡 MLOps
- Monitor performance  
- Detect drift  
- Retraining pipeline  

---

## 📁 Structure
```
Land-Type-Classification/
│
├── data/
│   ├── sample/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│
├── models/
│   ├── trained_model.h5
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│
├── deployment/
│   ├── app.py
│
├── results/
│   ├── plots/
│   ├── metrics/
│
├── requirements.txt
├── README.md
```


## ▶️ Usage
```bash
pip install -r requirements.txt
python src/train.py
```


## 🌱 Future Work
- Vision Transformers  
- More datasets  
- Cloud deployment  

python -m src.main