# ConVibNet: Needle Detection during Continuous Insertion via Frequency-Inspired Features

![pipeline](figures/fig-overview.png)

# Abstract

**Purpose:** Ultrasound-guided needle interventions are widely used in clinical practice, but their success critically depends on accurate needle placement, which is frequently hindered by the poor and intermittent visibility of needles in ultrasound images. Existing approaches remain limited by artifacts, occlusions, and low contrast, and often fail to support real-time continuous insertion. To overcome these challenges, this study introduces a robust real-time framework for continuous needle detection.

**Methods:** We present ConVibNet, an extension of VibNet for detecting needles with significantly reduced visibility, addressing real-time, continuous needle tracking during insertion. ConVibNet leverages temporal dependencies across successive ultrasound frames to enable continuous estimation of both needle tip position and shaft angle in dynamic scenarios. To strengthen temporal awareness of needle-tip motion, we introduce a novel intersection-and-difference loss that explicitly leverages motion correlations across consecutive frames. In addition, we curated a dedicated dataset for model development and evaluation.

**Results:** The performance of the proposed ConVibNet model was evaluated on our dataset, demonstrating superior accuracy compared to the baseline VibNet and UNet-LSTM models. Specifically, ConVibNet achieved a tip error of 2.80±2.42 mm and an angle error of 1.69±2.00°. These results represent a 0.75 mm improvement in tip localization accuracy over the best-performing baseline, while preserving real-time inference capability.

**Conclusion:** ConVibNet advances real-time needle detection in ultrasound-guided interventions by integrating temporal correlation modeling with a novel intersection-and-difference loss, thereby improving accuracy and robustness and demonstrating high potential for integration into autonomous insertion systems.

# Usage

This project is largely based on the official implementation of **VibNet: Vibration-Boosted Needle Detection in Ultrasound Images**.

We only provide the modified parts of the code in this repository. Specifically, we replaced the following components:

- dataset.py: replaced with **dataset_4loss.py**
- train.py: replaced with **train_4loss.py**
- test.py: replaced with **test_visual.py**

To run this project, please first clone the original repository and then substitute the corresponding files with those provided here.

Original repository:
```text
https://github.com/marslicy/VibNet
```