# Privacy-Preserving Face Recognition and Verification with Lensless Camera
This is the official pytorch implementation of the TCSVT 2024 paper titled [Privacy-Preserving Face Recognition and Verification With Lensless Camera](https://ieeexplore.ieee.org/document/10793399).

## Dependencies
The code is tested on Linux with the following packages:

1. Python 3.8
2. PyTorch 1.11.0
3. Numpy 1.23.0
4. OpenCV Python
5. Pillow
6. Joblib

## Dataset Preparation

Download the pre-computed dataset required for training or testing the system [here](https://mailmissouri-my.sharepoint.com/:u:/g/personal/chffn_umsystem_edu/IQBgLURyOuKgSrwfl8fLn8ipAY6Ikc-va09tctmaHQaVGcY?e=1Xz4Bo).

To prepare dataset from scratch, download the [**FlatCam Face Dataset**](https://computationalimaging.rice.edu/databases/flatcam-face-dataset/) by Rice University. Particurlarly, download the "Raw captures" ("fc_captures.tar.gz") which will contain .png files of the raw Flatcam sensor measurements.

1. Split the data into train/test.
   ```bash
   cd fc_data_process/
   python prep_data_recog_complete.py --data_path "path_to_input_data/" --out_path "path_to_splitted_data/"
   ```
2. [Optional] Run the following to generate pseudo-random noise locations. Skip this step if you want to use pre-computed noise locations in folder 'data/noise_locations' consistent with the paper.
   ```bash
   python generate_noise_locations.py --loc_per_pixel 10
   ```
3. Generate the lensless dataset containing the proposed Multi-resolution DCT Subband Representation. This may consume a couple of minutes even with multi-core processing.
   ```bash
   python process_raw_fc_meas.py --data_path "path_to_splitted_data/train" --out_path "path_to_output_folder/train"
   python process_raw_fc_meas.py --data_path "path_to_splitted_data/test" --out_path "path_to_output_folder/test"
   ```
   Output folders:

   - ymdct_npy: Contains the proposed Multi-resoluion DCT Subband Representation. Save as a numpy array (.npy). This would be used for training the model.

   - ymdct_noisy_npy: Contains the proposed Multi-resoluion DCT Subband Representation with pseudo-random noise pattern. Save as a numpy array (.npy).

   - dct_vis: Contains DCT of the sensor measurement. Saved an image file.

   - meas_vis: Contains visulation of the resized sensor measurement. Saved an image file.
4. [Optional] Generate verification pairs for testing. Skip this step if you'd like to use test pairs consistent with the paper ('data/pairs_verification.txt').
   ```bash
   python generate_verification_pairs.py --data_path "path_to_output_folder/test/ymdct_npy" --output_file "pairs_new.txt" --num_of_pairs 10000
   ```

## Training

## Testing


## Citations

If you use this work in your research, please cite:

### Journal Version
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
```
### Conference Version
```bibtex
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
```

## Contact
In case of questions, feel free to contact at chffn@umsystem.edu or engr.chrishenry@gmail.com
