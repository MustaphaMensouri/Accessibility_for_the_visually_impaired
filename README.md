# Object Recognition with Voice Output for Visually Impaired Users

An accessible, real-time object recognition system designed to help visually impaired individuals identify everyday objects through voice descriptions.

## 📋 Overview

This project aims to develop an AI-powered solution that detects common objects and provides immediate voice descriptions, enhancing autonomy and safety for people with visual impairments. With over 2.2 billion people worldwide suffering from visual deficiency, this open-source tool strives to make assistive technology more accessible.

## ✨ Key Features

- **Real-time Object Detection**: Identifies furniture, food products, signage, and everyday items
- **Voice Output**: Generates clear, concise audio descriptions adapted to context
- **Offline Functionality**: Works without internet connection for privacy and reliability
- **Low Latency**: Optimized for immediate responses
- **Lightweight**: Designed to run on resource-constrained devices
- **Open Source**: Free and accessible to all

## 🎯 Project Goals

### General Objective
Develop a real-time object recognition system with integrated voice output to assist visually impaired users in their daily activities.

### Specific Objectives
- Detect common objects (furniture, products, signage)
- Provide immediate voice descriptions
- Ensure minimal latency and offline operation
- Maintain user privacy
- Create an affordable, accessible solution

## 🛠️ Technology Stack

### Detection Models
- **YOLOv8** (Primary): Fast and efficient for real-time detection on limited hardware
- **Detectron2** (Alternative): For precise segmentation tasks
- **MobileNet-SSD** (Mobile): Optimized for low-power mobile devices

### Frameworks
- PyTorch
- TensorFlow
- OpenCV

### Text-to-Speech
- **gTTS**: Cloud-quality voice synthesis
- **pyttsx3**: Offline voice generation

## 📊 Dataset

### Sources
- **COCO Dataset**: 80 classes, 330K annotated images
- **Open Images**: 600 classes, 9M images

### Current Statistics
- **Total Images**: 4,840
- **Classes**: 50 (furniture, food, signage, electronics)
- **Average per Class**: ~100 images
- **Split**: Train (70%), Test (20%)

### Data Organization
The dataset has been collected and organized by team members covering diverse everyday situations:
- **Member 1**: Furniture (chairs, tables, lamps)
- **Member 2**: Food products (bottles, fruits, groceries)
- **Member 3**: Signage (traffic lights, signs)

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone https://github.com/MustaphaMensouri/Accessibility_for_the_visually_impaired.git
cd object-recognition-visually-impaired

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# For webcam demo
python demo_webcam.py
```

## 📈 Project Progress

### Week 1 - Framework
- ✅ Project scope validated
- ✅ Review of similar tools and projects
- ✅ Initial technology choices: YOLOv8, OpenCV, PyTorch, gTTS

### Week 2 - Data Preparation
- ✅ Dataset selection: COCO, Open Images
- 🔄 Data cleaning and annotation in progress
- 🔄 Class exploration and distribution analysis

## 🔜 Next Steps

1. **Finalize Dataset**: Complete annotations, balance classes, define evaluation protocols
2. **Train Models**: Compare YOLOv8 vs Detectron2 on latency/robustness metrics
3. **Optimize Inference**: Fine-tune for local, real-time performance
4. **Integrate TTS**: Build offline voice synthesis prototype (pyttsx3)
5. **User Prototype**: Develop webcam demo and conduct targeted user testing

## 🌟 Our Approach vs. Existing Solutions

While similar projects exist (Seeing AI, Google Lookout, Be My Eyes), our solution addresses key limitations:

| Feature | Existing Solutions | Our Approach |
|---------|-------------------|--------------|
| Internet Dependency | Often required | **Fully offline** |
| Cost | Proprietary/paid | **Open-source/free** |
| Latency | Variable | **Optimized for real-time** |
| Privacy | Cloud-dependent | **Local processing** |

## 👥 Team

- **Ait Dihim Nassim**
- **El-hamidy Abderahman**
- **Mensouri Mustapha**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open-source and available under the [Apache License](LICENSE).

## 📧 Contact

For questions or suggestions, please open an issue or contact the team.

---

**Final Goal**: Provide an accessible, open-source, privacy-respecting AI solution to enhance autonomy for visually impaired individuals.
