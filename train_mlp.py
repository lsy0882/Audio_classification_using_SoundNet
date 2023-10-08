import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pickle


class AudioDataset(Dataset):
    def __init__(self, feat_list_pool5, feat_list_conv6, feat_list_conv7, labels):
        self.feat_list_pool5 = feat_list_pool5
        self.feat_list_conv6 = feat_list_conv6
        self.feat_list_conv7 = feat_list_conv7
        self.labels = labels
        self.total_feats = [self.feat_list_pool5, self.feat_list_conv6, self.feat_list_conv7]

    def __len__(self):
        return len(self.feat_list_pool5)

    def __getitem__(self, idx):
        return (torch.tensor(self.feat_list_pool5[idx], dtype=torch.float),
                torch.tensor(self.feat_list_conv6[idx], dtype=torch.float),
                torch.tensor(self.feat_list_conv7[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.long))

class SingleHeadClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SingleHeadClassifier, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        self.block4 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Linear(512, num_classes)
        
        self.downsample = nn.Linear(input_dim, 512)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        out = self.block4(out)
        out = self.classifier(out)
        return out
    
class MultiHeadClassifier(nn.Module):
    def __init__(self, pool5_dim, conv6_dim, conv7_dim, num_classes):
        super(MultiHeadClassifier, self).__init__()
        self.pool5_classifier = SingleHeadClassifier(pool5_dim, num_classes)
        self.conv6_classifier = SingleHeadClassifier(conv6_dim, num_classes)
        self.conv7_classifier = SingleHeadClassifier(conv7_dim, num_classes)

    def forward(self, x_pool5, x_conv6, x_conv7):
        out_pool5 = self.pool5_classifier(x_pool5)
        out_conv6 = self.conv6_classifier(x_conv6)
        out_conv7 = self.conv7_classifier(x_conv7)
        return out_pool5, out_conv6, out_conv7

def main(args):
    with open(args.list_videos, "r") as fread:
        lines = fread.readlines()[1:]

    feat_list_pool5, feat_list_conv6, feat_list_conv7, label_list = [], [], [], []
    for line in lines:
        video_id, category = line.strip().split(",")
        feat_filepath_pool5 = os.path.join(args.feat_dir, "pool5", video_id + args.feat_appendix)
        feat_filepath_conv6 = os.path.join(args.feat_dir, "conv6", video_id + args.feat_appendix)
        feat_filepath_conv7 = os.path.join(args.feat_dir, "conv7", video_id + args.feat_appendix)
        if os.path.exists(feat_filepath_pool5) and os.path.exists(feat_filepath_conv7):
            feat_list_pool5.append(np.genfromtxt(feat_filepath_pool5, delimiter=";", dtype="float"))
            feat_list_conv6.append(np.genfromtxt(feat_filepath_conv6, delimiter=";", dtype="float"))
            feat_list_conv7.append(np.genfromtxt(feat_filepath_conv7, delimiter=";", dtype="float"))
            label_list.append(int(category))

    # Depending on the mode, adjust datasets and dataloaders
    if args.mode == "train":
        train_dataset = AudioDataset(feat_list_pool5, feat_list_conv6, feat_list_conv7, label_list)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    elif args.mode == "train_val":
        X_train_pool5, X_val_pool5, X_train_conv6, X_val_conv6, X_train_conv7, X_val_conv7, y_train, y_val = train_test_split(
            feat_list_pool5, feat_list_conv6, feat_list_conv7, label_list, test_size=0.1, random_state=42)

        train_dataset = AudioDataset(X_train_pool5, X_train_conv6, X_train_conv7, y_train)
        val_dataset = AudioDataset(X_val_pool5, X_val_conv6, X_val_conv7, y_val)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
    else:
        raise ValueError(f"Invalid mode {args.mode}. Choose 'train' or 'train_val'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool5_dim = len(train_dataset.feat_list_pool5[0])
    conv6_dim = len(train_dataset.feat_list_conv6[0])
    conv7_dim = len(train_dataset.feat_list_conv7[0])
    
    model = MultiHeadClassifier(pool5_dim, conv6_dim, conv7_dim, num_classes=len(set(label_list))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    for epoch in range(400):
        model.train()
        running_loss = 0.0
        for features_pool5, features_conv6, features_conv7, labels in train_loader:
            features_pool5, features_conv6, features_conv7, labels = features_pool5.to(device), features_conv6.to(device), features_conv7.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs_pool5, outputs_conv6, outputs_conv7 = model(features_pool5, features_conv6, features_conv7)
            loss_pool5 = criterion(outputs_pool5, labels)
            loss_conv6 = criterion(outputs_conv6, labels)
            loss_conv7 = criterion(outputs_conv7, labels)
            total_loss = (loss_pool5 + loss_conv6 + loss_conv7) / 3.0
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * features_pool5.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        if args.mode == "train_val":
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features_pool5, features_conv6, features_conv7, labels in val_loader:
                    features_pool5, features_conv6, features_conv7, labels = features_pool5.to(device), features_conv6.to(device), features_conv7.to(device), labels.to(device)
                    outputs_pool5, outputs_conv6, outputs_conv7 = model(features_pool5, features_conv6, features_conv7)
                    loss_pool5 = criterion(outputs_pool5, labels)
                    loss_conv6 = criterion(outputs_conv6, labels)
                    loss_conv7 = criterion(outputs_conv7, labels)
                    total_val_loss = (loss_pool5 + loss_conv6 + loss_conv7) / 3.0
                    val_loss += total_val_loss.item() * features_pool5.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), args.output_file)
    print('MLP model saved successfully')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("feat_dim", type=int)
    parser.add_argument("list_videos")
    parser.add_argument("output_file")
    parser.add_argument("feat_appendix", default=".csv")
    parser.add_argument("mode", default="train_val", choices=["train", "train_val"], help="Select mode: 'train' or 'train_val'")

    args = parser.parse_args()
    main(args)