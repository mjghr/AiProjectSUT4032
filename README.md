# AI Project SUT 4032 - Computer Engineering Department

## 🎓 Academic Information

- **Course**: Artificial Intelligence
- **Department**: Computer Engineering
- **Semester**: Spring 2025
- **Institution**: Sharif University of Technology

### 👥 Team Members

- **Nazanin Yousefi** - 401110172
- **Mohammad Javad "Mahan" Gharegozlou** - 401170134
- **MohammadMahdi NouriBorji** - 401170061

---

## 📚 Project Overview

This repository contains two comprehensive phases of an AI project focusing on advanced deep learning techniques for computer vision and reinforcement learning applications.

### Phase 1: Image Segmentation with U-Net Architectures

Implementation and comparison of three state-of-the-art neural network architectures for semantic segmentation:

- **U-Net**: Standard encoder-decoder architecture
- **Attention U-Net**: Enhanced with attention mechanisms for improved feature focusing
- **Residual Attention U-Net**: Combines residual blocks with attention mechanisms for better gradient flow

### Phase 2: Reinforcement Learning with Soft Actor-Critic (SAC)

Implementation of the Soft Actor-Critic algorithm for continuous control tasks, featuring:

- Off-policy learning with entropy regularization
- Actor-critic architecture with dual Q-networks
- Experience replay for stable learning

---

## 🔬 Phase 1: Image Segmentation

### 📊 Dataset

- **Massachusetts Roads Dataset**: Aerial imagery for road segmentation
- **Task**: Binary segmentation of road pixels
- **Image Size**: 256×256 pixels
- **Split**: Train/Validation/Test sets

### 🏗️ Architecture Details

#### U-Net

- **Encoder**: Contracting path with convolutional blocks and max pooling
- **Decoder**: Expanding path with upsampling and skip connections
- **Bottleneck**: Feature compression at the lowest resolution
- **Output**: Sigmoid activation for binary segmentation

#### Attention U-Net

- **Attention Gates**: Focus on relevant features in skip connections
- **Gating Mechanism**: Uses decoder features to guide attention
- **Improved Performance**: Better feature selection and localization

#### Residual Attention U-Net

- **Residual Blocks**: Enhanced gradient flow through skip connections
- **Combined Benefits**: Residual learning + attention mechanisms
- **Superior Performance**: Best results among the three architectures

### 📈 Performance Metrics

Models are evaluated using:

- **Dice Score**: Measures overlap between prediction and ground truth
- **IoU Score**: Intersection over Union for segmentation quality
- **Binary Cross-Entropy**: Standard classification loss
- **Combined Loss**: Sum of Dice + IoU + BCE losses

### 🎯 Performance Targets

| Model                    | Validation Loss | IoU Score | Dice Score |
| ------------------------ | --------------- | --------- | ---------- |
| U-Net                    | < 1.0           | > 0.5     | > 0.6      |
| Attention U-Net          | < 0.9           | > 0.6     | > 0.65     |
| Residual Attention U-Net | < 0.8           | > 0.65    | > 0.7      |

### 🔧 Technical Implementation

- **Framework**: PyTorch
- **Optimizer**: Adam (LR: 5e-4)
- **Scheduler**: Linear learning rate decay
- **Batch Size**: 8
- **Epochs**: 25
- **Data Augmentation**: Normalization and tensor conversion

---

## 🎮 Phase 2: Reinforcement Learning

### 🌟 Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that maximizes both expected return and entropy:

- **Entropy Regularization**: Encourages exploration through policy entropy
- **Soft Target Updates**: Stable learning with target networks
- **Dual Q-Networks**: Reduces overestimation bias

### 🏃‍♂️ Environment

- **Task**: HalfCheetahBulletEnv-v0
- **Type**: Continuous control locomotion
- **Objective**: Learn to run efficiently
- **Action Space**: Continuous joint torques
- **Observation Space**: Joint positions, velocities, and orientations

### 🧠 Network Architectures

#### Actor Network

- **Input**: State observations
- **Output**: Mean and standard deviation for Gaussian policy π(a|s) = N(μ(s), σ(s))
- **Activation**: ReLU hidden layers, tanh output squashing
- **Purpose**: Policy learning with exploration

#### Critic Networks (Dual)

- **Input**: State-action pairs
- **Output**: Q-value estimates Q(s,a)
- **Architecture**: Two separate networks to reduce overestimation
- **Loss**: Mean Squared Error with target values

#### Value Network

- **Input**: State observations
- **Output**: State value V(s)
- **Purpose**: Baseline for policy updates and target computation
- **Target Network**: Soft updates for stability

### ⚙️ Hyperparameters

- **Learning Rate**: 1e-4 (critics), 1e-4 (actor)
- **Discount Factor (γ)**: 0.99
- **Entropy Coefficient (α)**: 0.1
- **Soft Update Rate (τ)**: 0.005
- **Replay Buffer**: 1,000,000 transitions
- **Batch Size**: 256
- **Update Frequency**: Every 2 steps

### 📊 Training Process

- **Episodes**: 2000
- **Warmup Period**: 10,000 steps before learning
- **Experience Replay**: Store and sample transitions for stable learning
- **Performance Tracking**: Episode scores and moving averages

---

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
torch
torchvision
numpy
matplotlib
opencv-python
pillow
pandas
tqdm

# For RL (Phase 2)
gym==0.21
pybullet==3.2.5
```

### Installation

```bash
git clone <repository-url>
cd AiProjectSUT4032
pip install -r requirements.txt  # if available
```

### Running the Code

#### Phase 1: Image Segmentation

```bash
jupyter notebook Final_Project_Phase_1.ipynb
```

- Load the Massachusetts Roads Dataset
- Train U-Net, Attention U-Net, and Residual Attention U-Net
- Compare performance metrics
- Visualize predictions

#### Phase 2: SAC Training

```bash
jupyter notebook Final_Project_Phase_2.ipynb
```

- Configure environment and hyperparameters
- Train SAC agent on HalfCheetah task
- Monitor learning progress
- Evaluate trained policy

---

## 📁 Project Structure

```
AiProjectSUT4032/
├── Final_Project_Phase_1.ipynb    # U-Net implementations and training
├── Final_Project_Phase_2.ipynb    # SAC algorithm implementation
├── README.md                      # This file
└── tmp/                          # Model checkpoints and outputs
```

---

## 🔍 Key Features

### Phase 1 Highlights

- **Multiple Architecture Comparison**: Direct comparison of segmentation approaches
- **Advanced Loss Functions**: Combination of Dice, IoU, and BCE losses
- **Attention Mechanisms**: State-of-the-art attention gates implementation
- **Residual Learning**: Enhanced gradient flow for better training
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

### Phase 2 Highlights

- **Modern RL Algorithm**: SAC with entropy regularization
- **Stable Learning**: Experience replay and soft target updates
- **Continuous Control**: Complex locomotion task
- **Modular Design**: Separate Actor, Critic, and Value networks
- **Performance Monitoring**: Real-time training visualization

---

## 📊 Results Summary

### Image Segmentation Performance

The project successfully demonstrates the progressive improvement from standard U-Net to attention-enhanced and residual variants, with each architecture showing measurable improvements in segmentation quality.

### Reinforcement Learning Performance

The SAC implementation learns effective locomotion policies, demonstrating the algorithm's capability to balance exploration and exploitation through entropy regularization.

---

## 🔬 Technical Insights

### Architecture Innovations

1. **Attention Mechanisms**: Improve feature selection in segmentation
2. **Residual Connections**: Enhance gradient flow and training stability
3. **Entropy Regularization**: Balance exploration and exploitation in RL

### Training Strategies

1. **Progressive Complexity**: From basic to advanced architectures
2. **Comprehensive Evaluation**: Multiple metrics for thorough assessment
3. **Hyperparameter Tuning**: Optimized for each specific task

---

## 📚 References

### Image Segmentation

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)

### Reinforcement Learning

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning](https://arxiv.org/abs/1801.01290)
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

---

## 🤝 Contributing

This project is part of an academic assignment. For questions or discussions about the implementations, please contact the team members listed above.

---

## 📄 License

This project is developed for educational purposes as part of the Artificial Intelligence course at Sharif University of Technology.
