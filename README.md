# Surface Defect Detection for Manufacturing QA

AI-powered quality inspection system for detecting and localizing surface defects in manufactured metal products using YOLOv8.

## 🎯 Project Overview

This project implements a state-of-the-art deep learning system for automated defect detection in manufacturing, achieving 90%+ mAP on the NEU-DET Surface Defect Database.

### Key Features
- Real-time defect detection using YOLOv8
- 6 defect type classification with bounding boxes
- Trained on Google Colab with free GPU
- Object detection with precise defect localization
- XML to YOLO format conversion

## 📊 Dataset

**NEU-DET Surface Defect Database**
- Object detection format (Pascal VOC XML)
- 1,800+ annotated images
- 6 defect classes:
  1. Crazing
  2. Inclusion
  3. Patches
  4. Pitted Surface
  5. Rolled-in Scale
  6. Scratches
- Source: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

## 🚀 Training Details

- **Model**: YOLOv8n (Nano)
- **Platform**: Google Colab (Tesla T4 GPU)
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640×640
- **Training Time**: ~2-3 hours
- **Optimizer**: Adam
- **Learning Rate**: 0.001

## 📈 Results

- **mAP@0.5**: 92%+
- **mAP@0.5:0.95**: 75%+
- **Precision**: 90%+
- **Recall**: 88%+
- **Inference Speed**: 45 FPS (GPU)

### Model Capabilities
- Detects multiple defects per image
- Provides bounding box coordinates
- Real-time quality inspection
- Accurate defect classification

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.5.1
- **Object Detection**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV 4.12.0
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Annotation Conversion**: XML (Pascal VOC) to YOLO format
- **Platform**: Google Colab (Free GPU)

## 📂 Project Structure

```
surface-defect-detection/
├─ README.md
├─ data.yaml
├─ data/
│  └─ NEU-DET/
│     ├─ train/
│     │  ├─ images/
│     │  └─ annotations/
│     └─ validation/
│        ├─ images/
│        └─ annotations/
├─ yolo_dataset/
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ labels/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ models/
│  └─ defect_detection/
│     ├─ weights/
│     │  ├─ best.pt
│     │  └─ last.pt
│     ├─ results.png
│     ├─ confusion_matrix.png
│     ├─ PR_curve.png
│     ├─ F1_curve.png
│     ├─ P_curve.png
│     └─ R_curve.png
└─ runs/
   └─ detect/
```

Legend:
- data/: original NEU-DET dataset (Pascal VOC XML + images)
- yolo_dataset/: converted YOLO format (images + labels)
- models/defect_detection/: training outputs and weights
- runs/: additional Ultralytics run artifacts

## 🎓 Usage

### Training in Google Colab

1. Open notebook in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Upload NEU-DET dataset ZIP file
4. Run all cells sequentially
5. Model automatically trains and saves

### Inference with Trained Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference on single image
results = model.predict('test_image.jpg', conf=0.25)

# Display results with bounding boxes
results[0].show()

# Get predictions
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"Class: {model.names[class_id]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Bounding Box: {bbox}")
```

### Batch Inference

```python
from ultralytics import YOLO
import os

model = YOLO('best.pt')

# Predict on multiple images
image_folder = 'test_images/'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

results = model.predict(image_files, conf=0.25, save=True)

print(f"Processed {len(results)} images")
print(f"Results saved to: runs/detect/predict/")
```

## 🏆 Project Highlights

✅ **Professional-grade defect detection system**  
✅ **90%+ accuracy** on challenging manufacturing data  
✅ **Real-time inference** with bounding box localization  
✅ **Trained entirely on free cloud GPU**  
✅ **Production-ready** object detection model  
✅ **Automatic XML to YOLO conversion**  
✅ **Complete end-to-end pipeline**  

## 📊 Training Pipeline

1. **Data Upload**: Manual upload of NEU-DET ZIP file
2. **Extraction**: Automatic extraction to `/content/data/`
3. **Structure Verification**: Check train/validation splits
4. **Annotation Analysis**: Parse XML files for class information
5. **Format Conversion**: Convert Pascal VOC XML to YOLO .txt
6. **Dataset Split**: Create train/val/test splits
7. **Configuration**: Generate data.yaml
8. **Training**: YOLOv8 training with GPU (2-3 hours)
9. **Evaluation**: Validate on test set
10. **Visualization**: Generate performance metrics

## 📊 Training Visualizations

The project generates:
- Training curves (loss, mAP, precision, recall over epochs)
- Confusion matrix (class-wise performance)
- Precision-Recall curves (per-class evaluation)
- F1 score curves
- Sample predictions with bounding boxes

## 🔄 Data Conversion Process

**Original Format (NEU-DET):**
- XML annotations (Pascal VOC format)
- Separate annotation and image folders

**Converted Format (YOLO):**
- .txt label files (one per image)
- Format: `class_id x_center y_center width height` (normalized 0-1)

**Conversion Script:**
- Parses XML bounding boxes
- Normalizes coordinates to 0-1 range
- Converts to YOLO format
- Creates mirror directory structure

## ⚙️ Configuration

**data.yaml** (YOLO dataset config):
```yaml
path: /content/yolo_dataset
train: images/train
val: images/val
test: images/test
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled_in_scale
  5: scratches
nc: 6
```

## 🐛 Troubleshooting

**Issue: GPU not available**
- Solution: Runtime → Change runtime type → GPU

**Issue: Out of memory during training**
- Solution: Reduce batch size from 16 to 8

**Issue: Training disconnects after 90 minutes**
- Solution: Keep browser tab active or use Colab Pro

## 📝 License

MIT License

## 🙏 Acknowledgments

- Northeastern University for the NEU-DET Surface Defect Database
- Ultralytics for YOLOv8 framework
- Google Colab for free GPU access
- Kaggle for dataset hosting

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NEU Surface Defect Database Paper](https://www.sciencedirect.com/science/article/pii/S0924271619300031)
- [Object Detection with YOLO](https://arxiv.org/abs/2304.00501)

---

⭐ **If you found this project helpful, please star the repository!**

**Training Stats:**
- Training Time: 2-3 hours with T4 GPU
- Total Images: 1,800+
- Model Size: ~6 MB (YOLOv8n)
- Inference Speed: 45 FPS on GPU
