# Privacy-Preserving Face Recognition and Verification with Lensless Camera
This is the official pytorch implmentation of the TCSVT 2024 paper titled [Privacy-Preserving Face Recognition and Verification With Lensless Camera](https://ieeexplore.ieee.org/document/10793399).

# Dataset Preparation

Download the [**FlatCam Face Dataset**](https://computationalimaging.rice.edu/databases/flatcam-face-dataset/) by (Rice University)

- 87 subjects
- 23,838 samples
- Real lensless sensor measurements
- Variations in pose, expression, and illumination

⚠️ **Dataset is not included** in this repository due to licensing and privacy constraints.  
Please refer to the paper for dataset access and preprocessing details.

## Citation

If you use this work in your research, please cite:

```bibtex
@ARTICLE{10793399,
  author={Henry, Chris and Salman Asif, M. and Li, Zhu},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Privacy-Preserving Face Recognition and Verification With Lensless Camera}, 
  year={2025},
  volume={7},
  number={3},
  pages={354-367},
  keywords={Face recognition;Discrete cosine transforms;Cameras;Privacy;Image recognition;Training;Noise;Image reconstruction;Lensless camera;FlatCam;face recognition;face verification;visual privacy;discrete cosine transform},
  doi={10.1109/TBIOM.2024.3515144}}

@INPROCEEDINGS{10096627,
  author={Henry, Chris and Asif, M. Salman and Li, Zhu},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Privacy Preserving Face Recognition with Lensless Camera}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Privacy;Face recognition;Speech recognition;Cameras;Time measurement;Discrete cosine transforms;Lensless camera;FlatCam;face recognition;visual privacy;DCT},
  doi={10.1109/ICASSP49357.2023.10096627}}
