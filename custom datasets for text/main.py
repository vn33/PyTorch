# Importing necessary libraries
import os
import pandas as pd
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # for padding batches
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # for loading images
import torchvision.transforms as transforms

# Comments to explain the overall goals of the code:
# We want to convert text to numerical values for model input
# 1. Create a Vocabulary class to map each word to an index.
# 2. Set up a PyTorch Dataset class to load the data.
# 3. Ensure padding of each batch so all sequences have the same length, and set up a DataLoader.

# Initialize the Spacy English tokenizer
spacy_eng = spacy.load("en_core_web_sm")

spacy_eng = spacy.load("en_core_web_sm")

# Vocabulary class to handle text tokenization and word-to-index mapping
class Vocabulary:
    def __init__(self, freq_threshold):
        # Initialize dictionaries for index-to-string (itos) and string-to-index (stoi)
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold  # Frequency threshold for adding words to vocabulary

    def __len__(self):
        return len(self.itos)  # Return the size of the vocabulary

    @staticmethod
    def tokenizer_eng(text):
        # Tokenize text using a Spacy tokenizer, converting to lowercase
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        # Build vocabulary from a list of sentences based on frequency threshold
        frequencies = {}  # Dictionary to count word frequencies
        idx = 4  # Start index for new words

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1  # Initialize word frequency
                else:
                    frequencies[word] += 1  # Increment word frequency

                if frequencies[word] == self.freq_threshold:
                    # Add word to vocabulary if it meets the frequency threshold
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        # Convert text to a list of numerical indices based on the vocabulary
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


# Dataset class to load image-caption pairs
class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir  # Directory with images
        self.df = pd.read_csv(captions_file)  # Load captions from a CSV file
        self.transform = transform  # Optional transform to apply to images

        # Get columns with image filenames and captions
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build it using the captions
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)  # Return the total number of data points

    def __getitem__(self, index):
        # Get the caption and image ID for a specific index
        caption = self.captions[index]
        img_id = self.imgs[index]
        # Load the image and convert it to RGB
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)  # Apply transform if provided

        # Numericalize the caption and add special tokens
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # Return the image and the numericalized caption as a tensor
        return img, torch.tensor(numericalized_caption)


# Custom collate function to pad sequences in each batch
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader



Here's the code with detailed comments explaining each part, including how the MyCollate class and get_loader function work together to process and load the dataset:

python
Copy code
# Import necessary libraries
import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Initialize the Spacy English tokenizer
spacy_eng = spacy.load("en_core_web_sm")

# Vocabulary class to handle text tokenization and word-to-index mapping
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# Dataset class to load image-caption pairs
class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

# Custom collate function to pad sequences in each batch
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets

# Function to create a DataLoader
def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader

# Main function to test the data loader
def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )

    dataloader = get_loader("flick8k/images/",
                            annotation_file="flickr8k/captions.txt",
                            transform=transform)

    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()