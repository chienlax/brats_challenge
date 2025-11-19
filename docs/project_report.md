# **BraTS 2024 Post-Treatment MRI Segmentation: Comprehensive Project Report**

**Author:** Student Name

**Date:** October 6, 2025

**Course:** Deep Learning / Medical Image Processing

**Status:** Final Consolidated Report

## **1\. Executive Summary & Objectives**

### **1.1. Problem Description**

**Automated Segmentation of Post-Treatment Glioma in Multi-Modal MRI Using Deep Learning**

This study addresses the significant challenge of automating the segmentation of post-treatment glioma sub-regions. In contrast to pre-treatment scenarios, the post-treatment MRI landscape is characterized by high heterogeneity, including surgically induced cavities, resection margins, scar tissue, and radiation-induced leukoencephalopathy, all of which confound standard segmentation algorithms.

### **1.2. Research Objectives**

The primary objectives of this research are to:

1. **Develop an automated segmentation framework** utilizing the nnU-Net architecture to accurately delineate multiple tumor sub-regions in post-treatment glioma patients using multi-modal MRI scans.  
2. **Address the "Resection Cavity" (RC) segmentation challenge** introduced in the BraTS 2024 dataset, distinguishing surgical voids from residual non-enhancing tumor core (NETC) and necrosis.  
3. **Evaluate the efficacy of the 3D full-resolution U-Net** configuration in handling severe class imbalance and anatomical variability inherent in post-operative neuroimaging.  
4. **Conduct a comprehensive performance assessment** employing both voxel-wise metrics (Dice Similarity Coefficient, IoU) and lesion-wise analysis (Hausdorff Distance 95, Lesion Dice) to ensure clinical relevance and reliability.

## **2\. Medical Background & Annotation Protocol**

A fundamental understanding of MRI physics and the specific annotation protocols utilized in BraTS 2024 is prerequisite for interpreting the model's performance characteristics.

### **2.1. MRI Modalities and Principles**

* **T1-weighted (T1n):** The "Anatomical Map."  
  * *Principle:* Sensitive to longitudinal relaxation times (T1). Adipose tissue appears hyperintense, while water and Cerebrospinal Fluid (CSF) appear hypointense.  
  * *Clinical Role:* Serves as the baseline for subtraction with T1-Gd to isolate true contrast enhancement. Crucial for identifying "intrinsic T1 hyperintensity" (e.g., methemoglobin from post-operative hemorrhage), which designates a region as Non-enhancing Tumor Core (NETC) rather than active Enhancing Tissue.  
* **T2-weighted (T2w):** The "Pathology Map."  
  * *Principle:* Sensitive to transverse relaxation times (T2). Pathological tissues with high water content (tumor, edema) and CSF appear hyperintense.  
  * *Clinical Role:* The primary modality for delineating the **Resection Cavity (RC)**, which typically presents as a fluid-filled, T2-hyperintense surgical void.  
* **FLAIR (T2-FLAIR):** The "Edema Map."  
  * *Principle:* T2-weighted sequence with CSF signal suppression (Inversion Recovery).  
  * *Clinical Role:* The gold standard for defining **Surrounding Non-enhancing FLAIR Hyperintensity (SNFH)**. By suppressing the CSF signal, FLAIR ensures that peritumoral edema and infiltrative tumor are distinguishable from ventricles and sulci.  
* **T1-Gd (T1-weighted with Gadolinium):** The "Active Tumor Map."  
  * *Principle:* Paramagnetic contrast agents (Gadolinium) accumulate in regions with Blood-Brain Barrier (BBB) disruption.  
  * *Clinical Role:* The definitive sequence for identifying **Enhancing Tissue (ET)**, representing active tumor proliferation.

### **2.2. Label Definitions (BraTS 2024\)**

The segmentation targets are formally defined as follows:

| Label | Class Name | Color Code | Definition | Key MRI Features |
| :---- | :---- | :---- | :---- | :---- |
| **1** | **Enhancing Tissue (ET)** | Blue | Active, proliferating tumor tissue. | Hyperintense on T1-Gd relative to T1n. Strictly excludes intravascular contrast and intrinsic T1 hyperintensity. |
| **2** | **Non-enhancing Tumor Core (NETC)** | Red | Necrotic core, cysts, or non-enhancing tumor parenchyma. | Hypointense on T1-Gd; may encompass regions of intrinsic T1 hyperintensity (e.g., blood products). |
| **3** | **Surrounding FLAIR Hyperintensity (SNFH)** | Green | Edema, infiltration, gliosis, and radiation effects. | Hyperintense on FLAIR. Represents the "whole tumor" extent typically including vasogenic edema. |
| **4** | **Resection Cavity (RC)** | Yellow | Surgical cavity following tumor resection. | **Novel Class.** Fluid-filled cavity; hyperintense on T2, hypointense on T1/FLAIR. |

### **2.3. Common Segmentation Pitfalls**

Expert annotation review highlights recurring challenges for automated systems:

1. **Vascular Mimicry:** Intracranial vessels exhibit contrast enhancement (T1-Gd hyperintensity) similar to ET. *Distinction:* Vessels display linear/tubular morphology, whereas tumor tissue is typically nodular.  
2. **Choroid Plexus Enhancement:** The choroid plexus lacks a blood-brain barrier and naturally enhances. *Distinction:* Anatomical localization within the ventricles allows for differentiation.  
3. **Intrinsic T1 Hyperintensity:** Post-operative blood products appear bright on both pre- and post-contrast T1 images. Models relying solely on T1-Gd intensity may misclassify these as ET. *Distinction:* Comparison with T1n confirms the signal is intrinsic, classifying it as NETC or RC.  
4. **Microvascular Ischemic Changes:** Elderly patients often present with white matter lesions (leukoaraiosis) that are hyperintense on FLAIR. *Distinction:* These lesions are typically symmetric, patchy, and spatially distinct from the glioma mass.

## **3\. Dataset Characterization**

### **3.1. Dataset Snapshot**

This study utilizes the official BraTS 2024 Post-Treatment Glioma dataset, standardized to the NIfTI format.

| Cohort | Cases | Modalities | Resolution | Orientation |
| :---- | :---- | :---- | :---- | :---- |
| **Training** | 1,350 | T1, T1-Gd, T2, FLAIR | 1.0 mm³ Isotropic | LAS (Left-Anterior-Superior) |
| **Additional** | 271 | T1, T1-Gd, T2, FLAIR | 1.0 mm³ Isotropic | LAS |
| **Total** | **1,621** | \- | \- | \- |

### **3.2. Intensity and Geometric Standardization**

* **Geometry:** All volumes are rigid-registered to a common atlas space with dimensions of 182 × 218 × 182 voxels. This spatial normalization facilitates the learning of anatomical priors by the 3D CNN.  
* **Intensity:** Substantial inter-scanner variability is observed (T1c 99th percentile values range from approximately 92 to 12,000).  
  * *Implication:* Robust preprocessing, specifically percentile-based clipping (1st–99th percentile) and Z-score normalization, is mandatory to harmonize the input distribution.

### **3.3. Class Imbalance Analysis**

The dataset demonstrates severe class imbalance, a critical factor influencing model convergence and performance:

* **Background:** \>98% of total voxels.  
* **NETC (Label 2):** \~67% of the tumor mass (Dominant foreground class).  
* **RC (Label 4):** \~19% of the tumor mass.  
* **SNFH (Label 3):** \~11% of the tumor mass.  
* **ET (Label 1):** **\~2.3% of the tumor mass.** (Extremely rare, posing the greatest segmentation challenge).

## **4\. Methodology**

### **4.1. Preprocessing Pipeline**

To ensure data consistency and optimal network performance, the following preprocessing steps were applied:

1. **Reorientation:** All volumes were realigned to the standard RAS (Right-Anterior-Superior) coordinate system.  
2. **Resampling:** A strictly isotropic voxel spacing of 1.0 × 1.0 × 1.0 mm³ was enforced to maintain volumetric consistency.  
3. **Cropping:** Foreground cropping was performed to eliminate non-informative background regions, optimizing computational efficiency.  
4. **Normalization:** Channel-wise Z-score normalization was applied to non-zero voxels to standardize intensity distributions across modalities.

### **4.2. Model Architecture: nnU-Net 3D Full-Resolution**

This study employs the **nnU-Net framework** configured for **3D Full-Resolution** segmentation. The architecture is a dynamically configured PlainConvUNet derived from the dataset fingerprint (Dataset501\_BraTSPostTx).

* **Architecture Class:** dynamic\_network\_architectures.architectures.unet.PlainConvUNet  
* **Topological Structure:**  
  * **Depth:** The network consists of **6 stages** in both the encoder (contracting path) and decoder (expanding path).  
  * **Feature Maps:** The number of features per stage is hierarchically organized as [32, 64, 128, 256, 320, 320].  
  * **Convolutional Blocks:** Each stage contains 2 convolutional blocks (n\_conv\_per\_stage: [2, 2, 2, 2, 2, 2]).  
  * **Kernel Size:** Uniform 3x3x3 kernels are utilized across all stages.  
* **Downsampling Strategy (Strides):**  
  * Stage 1: [1, 1, 1] (No downsampling)  
  * Stages 2-5: [2, 2, 2] (Isotropic downsampling)  
  * Stage 6: [2, 2, 1] (Anisotropic downsampling to preserve Z-axis resolution in deep layers).  
* **Operational Components:**  
  * **Convolution:** torch.nn.modules.conv.Conv3d  
  * **Normalization:** torch.nn.modules.instancenorm.InstanceNorm3d (Affine=True, Epsilon=1e-05).  
  * **Activation:** torch.nn.LeakyReLU (In-place=True).  
* **Input Configuration:**  
  * **Patch Size:** [128, 160, 112] voxels.  
  * **Batch Size:** 2\.  
* **Training Strategy:**  
  * **Cross-Validation:** A robust 5-fold cross-validation scheme was implemented.  
  * **Loss Function:** A combination of Dice Loss and Cross-Entropy Loss was used to optimize segmentation overlap and pixel-wise classification accuracy.  
  * **Deep Supervision:** Auxiliary loss layers were active during training to improve gradient flow in deeper network layers.

## **5\. Comprehensive Performance Analysis**

This section details the quantitative results obtained from the validation of the 3D full-resolution UNet model.

### **5.1. Global Performance Statistics**

The aggregated metrics across the validation cohort indicate moderate overall performance, heavily influenced by the difficulty of specific sub-regions.

| Metric | Value | Interpretation |
| :---- | :---- | :---- |
| **Dice Similarity Coefficient (DSC)** | **0.6111** | Indicates moderate overlap between prediction and ground truth. |
| **Intersection over Union (IoU)** | 0.5302 | Reflects the area of overlap divided by the area of union. |
| **Precision** | 0.8894 | Demonstrates high positive predictive value (low false positive rate). |
| **Recall** | 0.8809 | Indicates high sensitivity in detecting tumor presence. |

### **5.2. Class-Specific Performance Breakdown**

Performance stratification by class reveals significant variance in segmentation accuracy.

| Class | Region | Dice | Precision | Recall | Analysis |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **ET** | **0.3816** | 0.6804 | 0.7764 | **Critical Bottleneck.** The low DSC is attributed to the small volumetric footprint of ET and confusion with vascular structures. |
| **2** | **NETC** | **0.8136** | 0.9051 | 0.9006 | **Optimal Performance.** The necrotic core presents distinct signal characteristics, leading to robust segmentation. |
| **3** | **SNFH** | 0.7153 | 0.9053 | 0.8659 | Strong performance; the boundaries of edema on FLAIR are well-captured. |
| **4** | **RC** | 0.5340 | 0.8213 | 0.7960 | Moderate performance. The morphological variability of the resection cavity challenges the model. |

### **5.3. Lesion-Wise Analysis**

Lesion-wise metrics provide a clinically relevant assessment by evaluating connected components rather than individual voxels.

| Class | Lesion Dice (Mean) | Lesion HD95 (Mean mm) | Clinical Insight |
| :---- | :---- | :---- | :---- |
| **ET** | 0.514 | 139.5 | The high Hausdorff Distance (HD95) suggests that the model misses small, focal lesions distant from the primary tumor mass. |
| **NETC** | 0.353 | 188.5 | The discrepancy between voxel-wise (0.81) and lesion-wise scores indicates fragmentation in the prediction of the necrotic core. |
| **SNFH** | 0.428 | 165.6 | While the main mass is detected, fuzzy boundaries and distant white matter lesions lead to distance penalties. |

### **5.4. Error Analysis and Pattern Discovery**

#### **Case-Level Performance**

* **Failures (DSC \< 0.25):** The model exhibits catastrophic failure in cases likely affected by severe artifacts or domain shifts (e.g., BraTS-GLI-02492-102, DSC 0.00). In these instances, the model often fails to detect the ET and SNFH classes entirely, only identifying the more obvious NETC.  
* **Successes (DSC \> 0.95):** In optimal cases (e.g., BraTS-GLI-02406-100, DSC 0.962), the model achieves near-human performance, accurately segmenting even the challenging ET class (DSC 0.96).

#### **Correlation Analysis**

1. **Volume Bias:** A positive correlation (\~0.37 \- 0.41) exists between lesion volume and Dice score across all classes. The model consistently underperforms on small lesions, which disproportionately affects the **Enhancing Tissue (ET)** class due to its naturally small volume.  
2. **Inter-Class Dependency:**  
   * **SNFH Context:** A positive correlation (0.447) between SNFH volume and ET performance suggests that large edema regions provide necessary spatial context for locating the active tumor.  
   * **RC Interference:** A negative correlation (-0.296) between RC volume and ET performance indicates that large resection cavities confound the segmentation of enhancing tissue. This aligns with the radiological difficulty of distinguishing post-surgical marginal enhancement (scarring) from true tumor recurrence.

## **6\. Conclusion**

The implementation of the 3D full-resolution nnU-Net demonstrates substantial efficacy in the segmentation of post-treatment glioma, achieving high accuracy for the necrotic core (**NETC, 81.36% Dice**) and peritumoral edema (**SNFH, 71.53% Dice**).

However, the analysis highlights two distinct challenges necessitating further investigation:

1. **Enhancing Tissue (ET) Precision:** With a Dice score of 38.16%, the differentiation of active tumor from post-treatment artifacts (e.g., vascular enhancement, blood products) remains a significant limitation.  
2. **Resection Cavity (RC) Delineation:** The novel BraTS 2024 class presents a moderate challenge (53.40% Dice), with error analysis suggesting confusion at the cavity boundaries where scar tissue and residual tumor interact.

**Future Work:**

* **Small Object Detection:** Implementation of hard negative mining or focal loss variants to specifically penalize the misclassification of small, enhancing structures.  
* **Boundary Refinement:** Investigation of attention-based mechanisms to improve delineation at the ambiguous interface between the resection cavity and enhancing tissue.  
* **Post-Processing:** Development of heuristic post-processing steps to filter spatially isolated false positives, such as vascular structures mimicking enhancing tumor.

## **Appendix: Computational Configuration**

**Hardware Environment:**

* **GPU:** 2 x NVIDIA H100 (160GB VRAM)
* **Compute Capability:** CUDA 11.x/12.x compatible