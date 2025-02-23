## Abstract 
Medical endoscopic image segmentation is crucial for clinical
diagnosis. However, existing methods face significant challenges due to
domain heterogeneity and prompt dependency constraints. In particular,
small target segmentation and complex background noise remain major
obstacles. Moreover, pre-trained models struggle to adapt to medical sce-
narios due to domain heterogeneity, which limits their ability to handle
diverse endoscopic images with small targets and complex backgrounds,
while their reliance on high-quality prompts further constrains perfor-
mance in medical applications. To address these, we propose MedEn-
doSAM, a novel framework for medical endoscopic segmentation. It inte-
grates a prompt enhance adapter to optimize class prompt interactions,
weighted multi-scale linear attention to suppress background noise and
enhance target features, and partial convolution and spatial attention to
capture multi-scale context for improved robustness. Furthermore, the
class-activated prompt encoder adaptively learns category-specific fea-
tures, reducing dependency on manual prompts. Extensive experiments
on the VocalFolds and EndoVis2018 datasets demonstrate that MedEn-
doSAM achieves state-of-the-art performance across multiple metrics
with a small number of parameters. It improves segmentation accuracy
for endoscopic images and significantly enhances robustness in complex
backgrounds, outperforming existing methods.
![](figure/framework.jpg)
## Clone Repository
1. Clone the repository.
``` shell
git clone https://github.com/ZHENGER001/MedEndoSAM.git
cd MedEndoSAM/
```
2. Create a virtual environment for SurgicalSAM and activate the environment.
```shell
conda create -n MedEndoSAM python=3.10
conda activate MedEndoSAM
pip install -r requirements.txt
```


