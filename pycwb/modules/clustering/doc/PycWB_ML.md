# PycWB ML

- [ ]  BKG as LLM with fine tuning with latest data

cWB statistics → Embedding → Clustering

What about TD → Pixel Selection → Skylocation reconstruction

### Main Stream Image Pattern Recognition Algorithm

- CNN:
    - **EfficientNetV2** — 在准确率与速度之间取得很好的折中，训练和迁移学习表现强劲。
    - **ConvNeXt** — 使用现代训练技巧改进的 CNN，性能接近 transformer。
    - **MobileNetV3 / Tiny CNN** — 适用于资源受限环境，如边缘设备和快速推理。
    - **ResNet / DenseNet / RegNet** — 稳定、成熟的基准架构，在许多视觉任务中仍被广泛使用。
- Transformer
    - **ViT（Vision Transformer）** — 将图像分块（patch）作为 token 输入，自注意力建模全局依赖。
    - **Swin Transformer** — 引入层次结构和局部窗口注意力，平衡效率和表现。
    - **Hybrid / EfficientFormer / TinyViT** — 结合 CNN + Transformer 或做轻量化改进，更适合资源受限场景。
    - **大型多模态视觉-语言模型（如 CLIP 和衍生 Vision-Language 模型）** — 图像识别可结合文本理解进行 zero-shot、跨模态任务。

### **🔹 3. 目标检测与分割等任务的先进框架**

虽然不是纯分类，但对于 *信号图谱检测* 这类任务非常有启发：

- **RF-DETR / YOLOv12** 等基于 Transformer 或 attention 的实时检测模型在定位任务中表现强劲。
- **Segment Anything Model (SAM)** 等大模型可用于像素级标注 + 下游分类任务。

# **Unsupervised Embedding for Time–Frequency Gravitational-Wave Searches**

## **A cWB-Centered Perspective**

### **Abstract**

Modern gravitational-wave burst searches, such as those based on coherentWaveBurst (cWB), rely on physically motivated time–frequency representations and coherence statistics across detector networks. While cWB is highly effective at detecting generic excess power events, downstream tasks such as glitch characterization, ranking refinement, and discovery of previously unseen signal morphologies remain challenging.

This report discusses how **unsupervised embedding learning** can be integrated *on top of* cWB, without replacing its physics-driven core. We review commonly used representation-learning algorithms, define physically consistent data augmentation strategies, highlight key pitfalls, and present a concrete implementation pathway tailored to cWB triggers.

---

## **1. Background: What cWB Already Provides**

coherentWaveBurst (cWB) performs the following essential steps:

1. Whitening of multi-detector strain data
2. Time–frequency decomposition (e.g., WDM / wavelet / Q-transform)
3. Identification of excess power tiles
4. Maximization of network coherence
5. Clustering of time–frequency tiles into events
6. Computation of coherent likelihood and auxiliary statistics

Importantly, **cWB already outputs high–information-density, physically meaningful quantities**, such as:

- Wavelet coefficients
- Coherent and incoherent energies
- Network coherence
- Null / residual energy
- Reconstructed waveforms and clusters

This makes cWB an ideal upstream source for representation learning.

---

## **2. What Is an Embedding?**

An **embedding** is a mapping

f: \text{event} \rightarrow \mathbb{R}^d

that converts a complex object (e.g., a multi-channel time–frequency cluster) into a low-dimensional vector such that:

- Similar physical events are close in embedding space
- Dissimilar events are far apart
- Novel or rare morphologies appear as outliers

Unlike classifiers, embeddings:

- Do not require predefined labels
- Support clustering, anomaly detection, and similarity search
- Are well suited for exploratory and discovery-driven analyses

---

## **3. Common Algorithms for Unsupervised Embedding**

### **3.1 Contrastive Learning (Recommended)**

**Core idea:**

Different “views” of the *same physical event* should map to similar embeddings, while different events should map to dissimilar embeddings.

Typical losses:

- InfoNCE
- NT-Xent
- Triplet loss (semi-supervised extension)

This approach is currently the **most robust and widely adopted** framework for unsupervised representation learning in vision and signal processing.

---

### **3.2 Masked / Reconstruction-Based Learning**

Examples:

- Autoencoders
- Masked spectrogram or masked patch prediction (ViT-style)

These methods enforce that the embedding retains enough information to reconstruct missing data, but:

- They often emphasize noise modeling
- They do not always produce well-separated embedding spaces

They are best used for **pretraining or warm-up**, not as a standalone solution.

---

### **3.3 Pure Clustering (Not Recommended Alone)**

Direct clustering (e.g., k-means on raw features) lacks invariance and typically produces unstable or non-generalizable representations.

---

## **4. Embedding Inputs Derived from cWB**

For each cWB trigger, construct an event-centric tensor:

X \in \mathbb{R}^{C \times T \times F}

Recommended channels include:

- log |wavelet coefficient|
- coherent energy E_c
- incoherent energy E_i
- network coherence c = E_c / (E_c + E_i)
- null or residual energy
- optional phase consistency measures

All of these quantities are **native to cWB** and retain clear physical meaning.

---

## **5. Data Augmentation: The Key to Unsupervised Learning**

### **5.1 Purpose of Augmentation**

In unsupervised embedding learning, augmentation defines **what transformations preserve event identity**.

Augmentation does **not** create new physics; it encodes *physical invariances*.

---

### **5.2 Physically Consistent Augmentations for cWB**

### **Strongly Recommended**

1. **Noise resampling / re-whitening**
    - Same signal, different noise realization
    - Reflects stochastic detector noise
2. **Detector dropout**
    - Use subsets of detectors (e.g., H1-only, L1-only)
    - Encourages robustness to network configuration
3. **Channel masking**
    - Randomly remove coherence or null-energy channels
    - Prevents over-reliance on a single statistic

---

### **Use with Caution**

1. **Small time shifts (within cluster bounds)**
2. **Mild time–frequency cropping**
3. **Multi-resolution representations (different Q or wavelet scales)**

---

### **Not Recommended**

- Time warping or frequency scaling (changes physical parameters)
- Random rotations or flips (invalid in time–frequency space)
- Strong additive noise that alters event morphology

---

### **5.3 Role in Contrastive Learning**

For each event:

- Generate two (or more) augmented views
- Encourage embeddings of the same event to be close
- Push embeddings of different events apart

---

## **6. Model Architecture (Typical Choice)**

- Encoder: CNN, ConvNeXt, Swin Transformer, or lightweight ViT
- Input: multi-channel time–frequency maps
- Output: embedding vector (e.g., 64–256 dimensions)
- Optional projection head (used only during training)

After training, **only the encoder is retained**.

---

## **7. Downstream Usage of Embeddings**

Once trained, embeddings can be used for:

- **Glitch family clustering** (e.g., HDBSCAN)
- **Anomaly / novelty detection**
- **Similarity search between events**
- **Ranking refinement**, combined with cWB likelihood

A conservative and collaboration-friendly approach is:

\text{Final score} = \alpha \cdot \text{cWB likelihood} + \beta \cdot \text{embedding anomaly} + \gamma \cdot \text{embedding similarity}

---

## **8. Key Pitfalls and Best Practices**

### **Common Failure Modes**

- Overly aggressive augmentation that changes event identity
- Relying solely on autoencoders
- Excessively high embedding dimensionality
- Attempting to replace cWB end-to-end

### **Best Practices**

- Keep augmentation conservative and physically motivated
- Let the model learn weights via attention, not manual tuning
- Validate embeddings visually (UMAP / t-SNE) and statistically
- Treat embedding as an **auxiliary representation**, not a decision-maker

---

## **9. Why This Approach Fits cWB Particularly Well**

- cWB already filters data through strong physical constraints
- Events are well localized in time–frequency space
- Detector coherence provides natural invariances
- Burst searches inherently target unknown morphologies

In this sense, **cWB is an ideal front-end for unsupervised embedding learning**.

---

## **10. Conclusion**

Unsupervised embedding learning provides a principled way to extend cWB beyond detection into characterization, clustering, and discovery, without compromising its physics-driven foundations. By carefully designing physically consistent augmentations and contrastive objectives, one can construct embedding spaces that reflect genuine signal morphology rather than noise artifacts.

This approach is particularly well suited for:

- Glitch taxonomy
- Discovery of rare or novel burst signals
- Supporting low-latency and offline ranking improvements in future observing runs.