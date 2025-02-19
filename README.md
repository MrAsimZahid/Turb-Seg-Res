<div align="center">

# 【CVPR'2024🔥】Turb-Seg-Res: A Segment-then-Restore Pipeline for Dynamic Videos with Atmospheric Turbulence

</div>

## Useful Links
| Links | Description | 
|:-----: |:-----: |
| [![Website Demo](https://img.shields.io/badge/TurbSegRes-Website-blue)](https://riponcs.github.io/TurbSegRes/) | Official project page with detailed information | 
| [![GitHub](https://img.shields.io/badge/TurbSegRes-GitHub-blue)](https://github.com/Riponcs/Turb-Seg-Res) | Link to the GitHub repository |
| [![Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/2404.13605) | Link to the CVPR 2024 paper |
| [![QuickTurbSim](https://img.shields.io/badge/QuickTurbSim-GitHub-blue)](https://github.com/Riponcs/QuickTurbSim) | Repository for simulating atmospheric turbulence effects |
| [![DOST Dataset](https://img.shields.io/badge/Dataset-DOST-orange)](https://turbulence-research.github.io/) | Dataset used in the project |

## Setup and Run
```sh
git clone https://github.com/Riponcs/Turb-Seg-Res.git
cd Turb-Seg-Res
pip install -r requirements.txt
python Demo.py
```

## Contributions
- **High Focal Length Video Stabilization:** Stabilizes videos captured by high focal length cameras, which are highly sensitive to vibrations.
- **Turbulence Video Simulation:** Introduces a novel tilt-and-blur video simulator based on simplex noise for generating plausible turbulence effects with temporal coherence.
- **Unsupervised Motion Segmentation:** Efficiently segments dynamic scenes affected by atmospheric turbulence, distinguishing between static and dynamic components.

## Usage (Demo.py)
The main script `Demo.py` processes input images with stabilization and enhancement options. You can run it with various command-line arguments:

### Basic Usage
```sh
python Demo.py
```

### Command Line Arguments
```sh
python Demo.py [OPTIONS]

Options:
  --input PATH        Input path (e.g., Input/sniper/*.png)
  --output PATH       Output directory path
  --frames NUMBER     Number of frames to process
  --resize FACTOR     Resize factor for input images
  --no-stabilize     Disable image stabilization
  --model PATH       Path to pretrained model
  --max-stb NUMBER   Maximum stabilization pixels
```

### Examples
```sh
# Process specific input folder
python Demo.py --input "Input/custom/*.png" --output "Output/custom"

# Process 50 frames with resizing
python Demo.py --frames 50 --resize 0.5

# Process without stabilization
python Demo.py --no-stabilize

# Use custom model
python Demo.py --model "path/to/custom/model.pth"
```

### Default Configuration
If no arguments are provided, the script uses these default settings:
- Input Path: `Input/Single_Car/*.png`
- Output Path: `Output/Single_Car/`
- Number of Frames: 100
- Stabilization: Enabled
- Resize Factor: 1
- Model Path: `PretrainedModel/restormer_ASUSim_trained.pth`
- Max Stabilization: 50 pixels

### Hardware Requirements
- GPU with at least 24GB VRAM recommended
- Warning will be shown if available GPU memory is insufficient

## Train the Model
Prepare the dataset:
```sh
python TrainRestormer/preapareDataset.py
```

Train the model:
```sh
python TrainRestormer/main.py
```

## Citation
If you find this work useful, please cite our CVPR 2024 paper:
```bibtex
@article{saha2024turb,
    title     = {Turb-Seg-Res: A Segment-then-Restore Pipeline for Dynamic Videos with Atmospheric Turbulence},
    author    = {Saha, Ripon Kumar and Qin, Dehao and Li, Nianyi and Ye, Jinwei and Jayasuriya, Suren},
    booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    year      = {2024},
}
```

## Additional Resources
- [![QuickTurbSim](https://img.shields.io/badge/QuickTurbSim-GitHub-blue)](https://github.com/Riponcs/QuickTurbSim): A repository for simulating atmospheric turbulence effects on images using 3D simplex noise and Gaussian blur.
