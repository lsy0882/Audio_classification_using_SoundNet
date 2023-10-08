# Audio Classification with Multi-head Classifier
This repository provides scripts to train and test a multi-head classifier on audio features extracted from SoundNet's pool5, conv6, and conv7 layers.


## Feature Extraction with SoundNet
Before training or testing the classifier, it's crucial to extract relevant features from your audio files. We employ the SoundNet architecture to extract features from audio data.
<br>

## About SoundNet
SoundNet is a deep convolutional neural network (CNN) architecture that has been trained on a vast amount of supervised and unsupervised data to produce meaningful audio representations.
<br>

## Environment Settings
We suggest using [conda](https://docs.conda.io/en/latest/) to manage your packages. You can quickly check or install the required packages from `environment.yaml`.
* Note: Please refer to [Pytorch](https://pytorch.org/get-started/locally/) for more detailed pytorch settings.
If you use conda, you should easilly install the packages through:
```bash
conda env create -f environment.yaml
```
Install FFMPEG by:
```bash
apt install ffmpeg
```
<br>

## Data and Labels
Please download the data from [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip) with wget. You could also download the data manually from [here](https://www.kaggle.com/competitions/cmu-11775-f23-hw1-audio-based-med/data). Then unzip it and put the videos under "$path_to_this_repo/videos", and labels under "$path_to_this_repo/labels". You can either directly download the data to this folder or in anywhere else then build a [soft link](https://linuxhint.com/create_symbolic_link_ubuntu/)

The `.zip` file should include the following:
1. `video/` folder with 8249 videos in MP4 format
2. `labels/` folder with two files:
    - `cls_map.csv`: csv with the mapping of the labels and its corresponding class ID (*Category*)
    - `train_val.csv`: csv with the Id of the video and its label
    - `test_for_students.csv`: submission template with the list of test samples

Let's create the folders to save extracted features, audios and models:
```bash
mkdir mp3/
for file in videos/*; do filename=$(basename "$file" .mp4); ffmpeg -y -i "$file" -q:a 0 -map a mp3/"${filename}".mp3; done
```
<br>

## How to Extract Features
* Note: Ensure that you use the same sampling rate and audio length as specified in the script configurations.

Run the extract_soundnet_feats.py script:
```bash
python extract_soundnet_feats.py --model_path <path_to_soundnet_weights> --input_dir <audio_files_directory> --output_dir <features_output_directory> [--file_ext .mp3] [--feat_layer ""]
```
Arguments:
- `model_path`: Path to the .npy file with the SoundNet weights.
- `input_dir`: Directory containing the audio files from which you want to extract features.
- `output_dir`: Directory where the extracted features will be stored. The script creates subdirectories for each layer (pool5, conv6, conv7).
- `file_ext`: (Optional) File extension of the audio files. Default is .mp3.
- `feat_layer`: (Optional) If you want to specify a particular layer for feature extraction. By default, the script extracts features from pool5, conv6, and conv7.
Once the feature extraction is complete, you'll find the features stored in the specified output_dir with subdirectories for each layer (pool5, conv6, conv7).
<br>

## Model
The model comprises of a Multi-head Classifier that contains three individual heads (classifiers) dedicated for each layer: pool5, conv6, and conv7. Each head processes its respective features, makes class predictions, and the combined loss from all heads is used for backpropagation.
<br>

## Training the Classifier
* Note: It's important to prepare your data correctly. The scripts expect features to be stored in specific directories (pool5, conv6, conv7) within the provided feat_dir.

1. Run the script with the necessary arguments:
```bash
python train_mlp.py <feat_dir> <feat_dim> <list_videos> <output_file> <feat_appendix> <mode>
```
Arguments:
- `feat_dir`: Directory containing the feature files.
- `feat_dim`: Dimension of the features.
- `list_videos`: Text file listing video IDs and their categories.
- `output_file`: Path to save the trained model.
- `feat_appendix`: (Optional) File extension of the feature files. Default is .csv.
- `mode`: (Optional) Mode to run the script in. Choices are train or train_val. Default is train_val.

2. Monitor the training process. After each epoch, the training loss will be printed. If <mode> is set to "train_val", the validation loss will also be displayed.

3. Once training is complete, the trained model will be saved to the specified <output_file>.
<br>

## Testing the Classifier
1. To test the trained model, run the test_mlp.py script:
```bash
python test_mlp.py <model_file> <feat_dir> <feat_dim> <list_videos> <output_file> <file_ext>
```
Arguments:
- `model_file`: Path to the saved trained model.
- `feat_dir`: Directory containing the feature files.
- `feat_dim`: Dimension of the features.
- `list_videos`: Text file listing video IDs.
- `output_file`: Path to save the predictions.
- `file_ext`: (Optional) File extension of the feature files. Default is .csv.

2. Once testing is complete, predictions will be saved in the specified output_file. The predictions file will list the video IDs and their respective predicted categories.
