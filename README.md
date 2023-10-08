# Audio Classification with Multi-head Classifier
This repository provides scripts to train and test a multi-head classifier on audio features extracted from SoundNet's pool5, conv6, and conv7 layers.
<br>

## Feature Extraction with SoundNet
Before training or testing the classifier, it's crucial to extract relevant features from your audio files. We employ the SoundNet architecture to extract features from audio data.
<br>

## About SoundNet
SoundNet is a deep convolutional neural network (CNN) architecture that has been trained on a vast amount of supervised and unsupervised data to produce meaningful audio representations.
<br>

## How to Extract Features
* Note: Ensure that you use the same sampling rate and audio length as specified in the script configurations.
Ensure that you have the necessary libraries installed:
```bash
pip install numpy torch librosa tqdm
```
Run the extract_soundnet_feats.py script:
```bash
python extract_soundnet_feats.py --model_path <path_to_soundnet_weights> --input_dir <audio_files_directory> --output_dir <features_output_directory> [--file_ext .mp3] [--feat_layer ""]
```
Arguments:
--model_path: Path to the .npy file with the SoundNet weights.
--input_dir: Directory containing the audio files from which you want to extract features.
--output_dir: Directory where the extracted features will be stored. The script creates subdirectories for each layer (pool5, conv6, conv7).
--file_ext: (Optional) File extension of the audio files. Default is .mp3.
--feat_layer: (Optional) If you want to specify a particular layer for feature extraction. By default, the script extracts features from pool5, conv6, and conv7.
Once the feature extraction is complete, you'll find the features stored in the specified output_dir with subdirectories for each layer (pool5, conv6, conv7).
<br>

## Dataset
The custom AudioDataset class reads features for each layer and respective labels, facilitating easy batching during training.
<br>

## Model
The model comprises of a Multi-head Classifier that contains three individual heads (classifiers) dedicated for each layer: pool5, conv6, and conv7. Each head processes its respective features, makes class predictions, and the combined loss from all heads is used for backpropagation.
<br>

## Training the Classifier
* Note: It's important to prepare your data correctly. The scripts expect features to be stored in specific directories (pool5, conv6, conv7) within the provided feat_dir.

1. Ensure you have the required libraries installed:

```bash
pip install numpy torch torchvision scikit-learn
```
2. Run the script with the necessary arguments:
```bash
python train_mlp.py <feat_dir> <feat_dim> <list_videos> <output_file> <feat_appendix> <mode>
```
Arguments:
feat_dir: Directory containing the feature files.
feat_dim: Dimension of the features.
list_videos: Text file listing video IDs and their categories.
output_file: Path to save the trained model.
feat_appendix: (Optional) File extension of the feature files. Default is .csv.
mode: (Optional) Mode to run the script in. Choices are train or train_val. Default is train_val.

3. Monitor the training process. After each epoch, the training loss will be printed. If <mode> is set to "train_val", the validation loss will also be displayed.

4. Once training is complete, the trained model will be saved to the specified <output_file>.
<br>

## Testing the Classifier
1. To test the trained model, run the test_mlp.py script:

```bash
python test_mlp.py <model_file> <feat_dir> <feat_dim> <list_videos> <output_file> <file_ext>
```
Arguments:
model_file: Path to the saved trained model.
feat_dir: Directory containing the feature files.
feat_dim: Dimension of the features.
list_videos: Text file listing video IDs.
output_file: Path to save the predictions.
file_ext: (Optional) File extension of the feature files. Default is .csv.

2. Once testing is complete, predictions will be saved in the specified output_file. The predictions file will list the video IDs and their respective predicted categories.