from cgi import test
import numpy as np
import os
from dataselection.utils.data.datasets.SL.custom_dataset_selcon import CustomDataset_WithId_SELCON
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split, TensorDataset
from torchvision import transforms
import PIL.Image as Image


from dataselection.utils.data.data_utils import *
import re
import pandas as pd
import torch
import pickle
from ..__utils import TinyImageNet
from dataselection.utils.data.data_utils import WeightedSubset
import pandas as pd
from datasets import load_dataset
from dataselection.utils.data.datasets.SL.color_mnist import ColoredDataset
from dataselection.utils.data.datasets.SL.celeba_loader import MyImageDataset, celeba

LABEL_MAPPINGS = {'glue_sst2':'label', 
                  'hf_trec6':'coarse_label', 
		          'imdb':'label',
                  'rotten_tomatoes': 'label',
                  'tweet_eval': 'label'}

SENTENCE_MAPPINGS = {'glue_sst2': 'sentence', 
                    'hf_trec6':'text',  
                    'imdb':'text',
                    'rotten_tomatoes': 'text',
                    'tweet_eval': 'text'}

class standard_scaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data

    def transform(self, data):
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data


def clean_data(sentence, type = 0, TREC=False):
    # From yoonkim: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    if type == 0:
        """
        Tokenization for SST
        """
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.strip().lower()
    elif type == 1:
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)     
        sentence = re.sub(r"\'s", " \'s", sentence) 
        sentence = re.sub(r"\'ve", " \'ve", sentence) 
        sentence = re.sub(r"n\'t", " n\'t", sentence) 
        sentence = re.sub(r"\'re", " \'re", sentence) 
        sentence = re.sub(r"\'d", " \'d", sentence) 
        sentence = re.sub(r"\'ll", " \'ll", sentence) 
        sentence = re.sub(r",", " , ", sentence) 
        sentence = re.sub(r"!", " ! ", sentence) 
        sentence = re.sub(r"\(", " \( ", sentence) 
        sentence = re.sub(r"\)", " \) ", sentence) 
        sentence = re.sub(r"\?", " \? ", sentence) 
        sentence = re.sub(r"\s{2,}", " ", sentence)    
        return sentence.strip() if TREC else sentence.strip().lower() 
        # if we are using glove uncased, keep TREC = False even for trec6 dataset
    else:
        return sentence

def get_class(sentiment, num_classes):
    # Return a label based on the sentiment value
    return int(sentiment * (num_classes - 0.001))


def loadGloveModel(gloveFile):
    glove = pd.read_csv(gloveFile, sep=' ', header=None, encoding='utf-8', index_col=0, na_values=None, keep_default_na=False, quoting=3)
    return glove  # (word, embedding), 400k*dim


class SSTDataset(Dataset):
    label_tmp = None

    def __init__(self, path_to_dataset, name, num_classes, wordvec_dim, wordvec, device='cpu'):
        """SST dataset
        
        Args:
            path_to_dataset (str): path_to_dataset
            name (str): train, dev or test
            num_classes (int): 2 or 5
            wordvec_dim (int): Dimension of word embedding
            wordvec (array): word embedding
            device (str, optional): torch.device. Defaults to 'cpu'.
        """
        phrase_ids = pd.read_csv(path_to_dataset + 'phrase_ids.' +
                                name + '.txt', header=None, encoding='utf-8', dtype=int)
        phrase_ids = set(np.array(phrase_ids).squeeze())  # phrase_id in this dataset
        self.num_classes = num_classes
        phrase_dict = {}  # {id->phrase} 


        if SSTDataset.label_tmp is None:
            # Read label/sentiment first
            # Share 1 array on train/dev/test set. No need to do this 3 times.
            SSTDataset.label_tmp = pd.read_csv(path_to_dataset + 'sentiment_labels.txt',
                                    sep='|', dtype={'phrase ids': int, 'sentiment values': float})
            SSTDataset.label_tmp = np.array(SSTDataset.label_tmp)[:, 1:]  # sentiment value
        
        with open(path_to_dataset + 'dictionary.txt', 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                phrase, phrase_id = line.strip().split('|')
                if int(phrase_id) in phrase_ids:  # phrase in this dataset
                    phrase = clean_data(phrase)  # preprocessing
                    phrase_dict[int(phrase_id)] = phrase
                    i += 1
        f.close()

        self.phrase_vec = []  # word index in glove
        # label of each sentence
        self.labels = torch.zeros((len(phrase_dict),), dtype=torch.long)
        missing_count = 0
        for i, (idx, p) in enumerate(phrase_dict.items()):
            tmp1 = []  
            for w in p.split(' '):
                try:
                    tmp1.append(wordvec.index.get_loc(w))  
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long)) 
            self.labels[i] = get_class(SSTDataset.label_tmp[idx], self.num_classes) 

        # print(missing_count)

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return len(self.phrase_vec)

class Trec6Dataset(Dataset):
    def __init__(self, data_path, cls_to_num, num_classes, wordvec_dim, wordvec, device='cpu'):
        self.phrase_vec = []
        self.labels = []

        missing_count = 0
        with open(data_path, 'r', encoding='latin1') as f:
            for line in f:
                label = cls_to_num[line.split()[0].split(":")[0]]
                sentence = clean_data(" ".join(line.split(":")[1:]), 1, False)
                
                tmp1 = []
                for w in sentence.split(' '):
                    try:
                        tmp1.append(wordvec.index.get_loc(w))  
                    except KeyError:
                        missing_count += 1

                self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))
                self.labels.append(label)

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return len(self.phrase_vec)

class GlueDataset(Dataset):
    def __init__(self, glue_dataset, sentence_str, label_str, clean_type, num_classes, wordvec_dim, wordvec, device='cpu'):
        self.len =  glue_dataset.__len__()       
        self.phrase_vec = []  # word index in glove
        # label of each sentence
        self.labels = torch.zeros((self.len,), dtype=torch.long)
        missing_count = 0
        for i, p in enumerate(glue_dataset):
            tmp1 = []
            for w in clean_data(p[sentence_str], clean_type, False).split(' '): #False since glove used is uncased
                try:
                    tmp1.append(wordvec.index.get_loc(w))  
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long)) 
            self.labels[i] = p[label_str]
        
    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]
    def __len__(self):
        return self.len

## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        self.isreg = isreg
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label)  # .astype('float32')
        # if self.isreg:
        #     return (sample_data, label, idx)
        # else:

class CustomCensusDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.float()
        self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        sa = np.array([sample_data[0], sample_data[8], sample_data[9]])
        return sample_data, label, sa  # .astype('float32')

class DatasetWithSensitiveAttributes(Dataset):
    def __init__(self, data, target, sensitive_attribute_indexes):
        self.data = data.float()
        self.targets = target.long()
        self.sensitive_attribute_indexes = sensitive_attribute_indexes

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        sa = sample_data[self.sensitive_attribute_indexes]
        return sample_data, label, sa  # .astype('float32')

class ARSDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.float()
        self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        sa = sample_data[7]  ### 7 est l'indice de l'attribut sensible "gender"
        return sample_data, label, sa  # .astype('float32')

class mobiactDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.float()
        self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        sa = np.array([sample_data[10], sample_data[11]])
        return sample_data, label, sa  # .astype('float32')

    
class dcDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.float()
        self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Check if idx is a tensor, and convert it to a list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        sa = np.array([sample_data[9], sample_data[10]])
        return sample_data, label, sa

class fairfaceDataset(Dataset) :
    def __init__(self, dataframe, transform=None, datadir = "../data/fairface/"):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the image paths and labels.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.features = dataframe.iloc[:, [0, 1, 3]].values
        self.target = dataframe.iloc[:, 2].values

        self.transform = transform
        self.datadir = datadir

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: A dictionary containing 'image' and 'label'.
        """
        # Get the image path and label from the dataframe


        img_path = self.features[idx][0]  # Assuming the first column is the image path
        
        gender = self.target[idx]     # Assuming the fifth column is the label

        if gender == "Female" :
            gender = 1
        else :
            gender = 0

        age = self.features[idx][1]
        if age in ["40-49", "50-59", "60-69", "0-2"] :
            age = 1
        else :
            age = 0

            
        race = self.features[idx][2]
        if race in ["Middle Eastern"] :
            race = 1
        else :
            race = 0
        
        sa = np.array([age, race])
        # Load the image
        image = Image.open(self.datadir + img_path)
        
        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
        
        return image, gender, sa


class CustomDataset_WithId(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label, idx  # .astype('float32')


## Utility function to load datasets from libsvm datasets
def csv_file_load(path, dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(",")]
            target.append(int(float(temp[-1])))  # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0] * dim
            count = 0
            for i in temp[:-1]:
                # ind, val = i.split(':')
                temp_data[count] = float(i)
                count += 1
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def libsvm_file_load(path, dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(" ")]
            target.append(int(float(temp[0])))  # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0] * dim

            for i in temp[1:]:
                ind, val = i.split(':')
                temp_data[int(ind) - 1] = float(val)
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

def clean_lawschool_full(path):
    df = pd.read_csv(path)
    df = df.dropna()
    # remove y from df
    y = df['ugpa']
    y = y / 4
    df = df.drop('ugpa', axis=1)
    # convert gender variables to 0,1
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    # add bar1 back to the feature set
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    # df['race'] = [int(race == 7.0) for race in df['race']]
    # a = df['race']
    return df.to_numpy(), y.to_numpy()


def adult_load(path, dim, save_data=True):
    enum = enumerate(
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
         'Never-worked'])
    workclass = dict((j, i) for i, j in enum)

    enum = enumerate(
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
         '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    education = dict((j, i) for i, j in enum)

    enum = enumerate(
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
         'Married-AF-spouse'])
    marital_status = dict((j, i) for i, j in enum)

    enum = enumerate(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                      'Handlers-cleaners',
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
                      'Protective-serv', 'Armed-Forces'])
    occupation = dict((j, i) for i, j in enum)

    enum = enumerate(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    relationship = dict((j, i) for i, j in enum)
        
    enum = enumerate(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    race = {j: 1 if j == 'White' else 0 for i, j in enum}

    sex = {'Female': 0, 'Male': 1}

    enum = enumerate(
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
         'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
         'Jamaica',
         'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
         'Haiti', 'Columbia',
         'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
         'Peru', 'Hong',
         'Holand-Netherlands'])
    native_country = dict((j, i) for i, j in enum)

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")]

            if '?' in temp or len(temp) == 1:
                line = fp.readline()
                continue

            if temp[-1].strip() == "<=50K" or temp[-1].strip() == "<=50K.":
                target.append(0)
            else:
                target.append(1)

            temp_data = [0] * dim
            count = 0
            # print(temp)

            for i in temp[:-1]:
                
                if count == 0:
                    if 30 <= float(i) <= 60:
                        temp_data[count] = 1
                    else:
                        temp_data[count] = 0
                elif count == 1:
                    temp_data[count] = workclass[i.strip()]
                elif count == 3:
                    temp_data[count] = education[i.strip()]
                elif count == 5:
                    temp_data[count] = marital_status[i.strip()]
                elif count == 6:
                    temp_data[count] = occupation[i.strip()]
                elif count == 7:
                    temp_data[count] = relationship[i.strip()]
                elif count == 8:
                    temp_data[count] = race[i.strip()]
                elif count == 9:
                    temp_data[count] = sex[i.strip()]
                elif count == 13:
                    temp_data[count] = native_country[i.strip()]
                else:
                    temp_data[count] = float(i)
                temp_data[count] = float(temp_data[count])
                count += 1

            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst, num_cls, ratio, seed=42):
    rng = np.random.default_rng(seed)
    samples_per_class = np.zeros(num_cls)
    val_samples_per_class = np.zeros(num_cls)
    tst_samples_per_class = np.zeros(num_cls)
    for i in range(num_cls):
        samples_per_class[i] = len(np.where(y_trn == i)[0])
        val_samples_per_class[i] = len(np.where(y_val == i)[0])
        tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
    min_samples = int(np.min(samples_per_class) * 0.1)
    selected_classes = rng.choice(np.arange(num_cls), size=int(ratio * num_cls), replace=False)
    for i in range(num_cls):
        if i == 0:
            if i in selected_classes:
                subset_idxs = rng.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = x_trn[subset_idxs]
            y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
        else:
            if i in selected_classes:
                subset_idxs = rng.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
            y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
    max_samples = int(np.max(val_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_val == i)[0]
        if i == 0:
            subset_ids = rng.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val, x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1), y_val[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = rng.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val, x_val_new, x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1), y_val_new, y_val[subset_ids].reshape(-1, 1)))
    max_samples = int(np.max(tst_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_tst == i)[0]
        if i == 0:
            subset_ids = rng.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst, x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1), y_tst[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = rng.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst, x_tst_new, x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1), y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

    return x_trn_new, y_trn_new.reshape(-1), x_val_new, y_val_new.reshape(-1), x_tst_new, y_tst_new.reshape(-1)


def create_noisy(y_trn, num_cls, noise_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    noise_size = int(len(y_trn) * noise_ratio)
    noise_indices = rng.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
    y_trn[noise_indices] = rng.choice(np.arange(num_cls), size=noise_size, replace=True)
    return y_trn


def tokenize_function(tokenizer, example, text_column):
    return tokenizer(example[text_column], padding = 'max_length', truncation=True)


def load_dataset(datadir, train_file, test_file, val_file, dset_name, isnumpy=False, **kwargs):
    num_cls = 2
    num_cls_mnist = 10
    fullset, valset, testset = None, None, None

    #### MobiAct
    ############
    if dset_name == "load-mobiact":
        # Paths to the saved CSV files
        trn_file = os.path.join(datadir, train_file)
        val_file = os.path.join(datadir, val_file)
        tst_file = os.path.join(datadir, test_file)

        # Read the CSV files
        train_df = pd.read_csv(trn_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(tst_file)

        # Extract X and y for each dataset
        X_trn = train_df.drop(columns=['activity']).values
        y_trn = train_df['activity'].values

        X_val = val_df.drop(columns=['activity']).values
        y_val = val_df['activity'].values

        X_tst = test_df.drop(columns=['activity']).values
        y_tst = test_df['activity'].values

        # Check if we want the data as numpy arrays or as a Dataset
        if isnumpy:
            fullset = (X_trn, y_trn)
            valset = (X_val, y_val)
            testset = (X_tst, y_tst)
        else:
            fullset = mobiactDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
            valset = mobiactDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
            testset = mobiactDataset(torch.from_numpy(X_tst).float(), torch.from_numpy(y_tst).long())

        return fullset, valset, testset, num_cls

    #### KDD
    ############
    elif dset_name == "load-kdd":
        # Paths to the saved CSV files
        trn_file = os.path.join(datadir, train_file)
        val_file = os.path.join(datadir, val_file)
        tst_file = os.path.join(datadir, test_file)

        # Read the CSV files
        train_df = pd.read_csv(trn_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(tst_file)

        # Extract X and y for each dataset
        X_trn = train_df.drop(columns=['income']).values
        y_trn = train_df['income'].values

        X_val = val_df.drop(columns=['income']).values
        y_val = val_df['income'].values

        X_tst = test_df.drop(columns=['income']).values
        y_tst = test_df['income'].values

        # Check if we want the data as numpy arrays or as a Dataset
        if isnumpy:
            fullset = (X_trn, y_trn)
            valset = (X_val, y_val)
            testset = (X_tst, y_tst)
        else:
            unchanged_column_indices = [train_df.columns.get_loc(col) for col in ['race', 'sex', 'age']]
            fullset = DatasetWithSensitiveAttributes(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long(), unchanged_column_indices)
            valset = DatasetWithSensitiveAttributes(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long(), unchanged_column_indices)
            testset = DatasetWithSensitiveAttributes(torch.from_numpy(X_tst).float(), torch.from_numpy(y_tst).long(), unchanged_column_indices)

        return fullset, valset, testset, num_cls

    #### ARS
    ############
    elif dset_name == "load-ars":
        # Paths to the saved CSV files
        trn_file = os.path.join(datadir, train_file)
        val_file = os.path.join(datadir, val_file)
        tst_file = os.path.join(datadir, test_file)

        # Read the CSV files
        train_df = pd.read_csv(trn_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(tst_file)

        # Extract X and y for each dataset
        X_trn = train_df.drop(columns=['activity']).values
        y_trn = train_df['activity'].values

        X_val = val_df.drop(columns=['activity']).values
        y_val = val_df['activity'].values

        X_tst = test_df.drop(columns=['activity']).values
        y_tst = test_df['activity'].values

        # Check if we want the data as numpy arrays or as a Dataset
        if isnumpy:
            fullset = (X_trn, y_trn)
            valset = (X_val, y_val)
            testset = (X_tst, y_tst)
        else:
            fullset = ARSDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
            valset = ARSDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
            testset = ARSDataset(torch.from_numpy(X_tst).float(), torch.from_numpy(y_tst).long())

        return fullset, valset, testset, num_cls

    #### Adult
    ############
    elif dset_name == "load-adult":
        # Paths to the saved CSV files
        trn_file = os.path.join(datadir, train_file)
        val_file = os.path.join(datadir, val_file)
        tst_file = os.path.join(datadir, test_file)

        # Read the CSV files
        train_df = pd.read_csv(trn_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(tst_file)

        # Extract X and y for each dataset
        X_trn = train_df.drop(columns=['income']).values
        y_trn = train_df['income'].values

        X_val = val_df.drop(columns=['income']).values
        y_val = val_df['income'].values

        X_tst = test_df.drop(columns=['income']).values
        y_tst = test_df['income'].values

        # Check if we want the data as numpy arrays or as a Dataset
        if isnumpy:
            fullset = (X_trn, y_trn)
            valset = (X_val, y_val)
            testset = (X_tst, y_tst)
        else:
            unchanged_column_indices = [train_df.columns.get_loc(col) for col in ['age', 'race', 'sex']]
            fullset = CustomCensusDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
            valset = CustomCensusDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
            testset = CustomCensusDataset(torch.from_numpy(X_tst).float(), torch.from_numpy(y_tst).long())

        return fullset, valset, testset, num_cls

    #### DC
    ############
    elif dset_name == "load-dc":
        # Paths to the saved CSV files
        trn_file = os.path.join(datadir, train_file)
        val_file = os.path.join(datadir, val_file)
        tst_file = os.path.join(datadir, test_file)

        # Read the CSV files
        train_df = pd.read_csv(trn_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(tst_file)

        # Extract X and y for each dataset
        X_trn = train_df.drop(columns=['occupation']).values
        y_trn = train_df['occupation'].values

        X_val = val_df.drop(columns=['occupation']).values
        y_val = val_df['occupation'].values

        X_tst = test_df.drop(columns=['occupation']).values
        y_tst = test_df['occupation'].values

        # Check if we want the data as numpy arrays or as a Dataset
        if isnumpy:
            fullset = (X_trn, y_trn)
            valset = (X_val, y_val)
            testset = (X_tst, y_tst)
        else:
            unchanged_column_indices = [train_df.columns.get_loc(col) for col in ['age', 'sex']]
            fullset = dcDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
            valset = dcDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
            testset = dcDataset(torch.from_numpy(X_tst).float(), torch.from_numpy(y_tst).long())

        return fullset, valset, testset, num_cls

    #### MNIST
    ############
    elif dset_name == "load-mnist":
         # Paths to the directories where images are stored
         train_dir = os.path.join(datadir, 'MNIST', train_file)
         val_dir = os.path.join(datadir, 'MNIST', val_file)
         test_dir = os.path.join(datadir, 'MNIST', test_file)

         # Load the metadata (CSV files containing labels and image paths)
         train_metadata = pd.read_csv(os.path.join(train_dir, 'labels.csv'))
         val_metadata = pd.read_csv(os.path.join(val_dir, 'labels.csv'))
         test_metadata = pd.read_csv(os.path.join(test_dir, 'labels.csv'))

         # Create datasets using the metadata, including the sensitive attribute (flag)
         def create_dataset_from_metadata(metadata, image_dir):
             images = []
             labels = metadata['label'].values
             sensitive_attributes = metadata['flag'].values  # Sensitive attribute column is named 'flag'

             for idx, row in metadata.iterrows():
                 img_path = os.path.join(image_dir, row['filename'])
                 # Check the flag value to determine if the image is colored or grayscale
                 if row['flag'] == 0:
                     img = Image.open(img_path).convert('RGB')  # Load as RGB if flag is 0
                 else:
                     img = Image.open(img_path).convert('L')    # Load as grayscale if flag is 1
                     img = img.convert('RGB')  # Convert grayscale image to RGB for consistency

                 img = torchvision.transforms.ToTensor()(img)

                 # Normalize the image
                 img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

                 images.append(img)

             # Convert images, labels, and sensitive attributes into tensors
             images_tensor = torch.stack(images)
             labels_tensor = torch.tensor(labels)
             sensitive_tensor = torch.tensor(sensitive_attributes)

             # Return as a dataset
             return torch.utils.data.TensorDataset(images_tensor, labels_tensor, sensitive_tensor)

         # Load the fullset, valset, and testset
         fullset = create_dataset_from_metadata(train_metadata, train_dir)
         valset = create_dataset_from_metadata(val_metadata, val_dir)
         testset = create_dataset_from_metadata(test_metadata, test_dir)

         # If class imbalance is specified, handle it here based on metadata
         if kwargs.get('feature') == 'classimb':
             rng = np.random.default_rng(kwargs.get('seed', None))
             samples_per_class = torch.zeros(num_cls_mnist)
             for i in range(num_cls_mnist):
                 samples_per_class[i] = len(torch.where(torch.tensor([label for _, label in train_metadata.iterrows()]) == i)[0])
             min_samples = int(torch.min(samples_per_class) * 0.1)
             selected_classes = rng.choice(np.arange(num_cls_mnist), size=int(kwargs['classimb_ratio'] * num_cls_mnist), replace=False)
             subset_idxs = []
             for i in range(num_cls_mnist):
                 if i in selected_classes:
                     batch_subset_idxs = list(rng.choice(torch.where(torch.tensor([label for _, label in train_metadata.iterrows()]) == i)[0].cpu().numpy(), size=min_samples, replace=False))
                 else:
                     batch_subset_idxs = list(torch.where(torch.tensor([label for _, label in train_metadata.iterrows()]) == i)[0].cpu().numpy())
                 subset_idxs.extend(batch_subset_idxs)
             fullset = torch.utils.data.Subset(fullset, subset_idxs)

         return fullset, valset, testset, num_cls_mnist





#### Saving images (MNIST)

def save_colored_mnist(dataset, folder_name, datadir):
    # Create the directory if it doesn't exist
    folder_path = os.path.join(datadir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    labels = []

    # Iterate through the dataset and save each image
    for idx, (img, label, flag) in enumerate(dataset):
        # Convert the tensor image to a PIL image
        img = transforms.ToPILImage()(img)
        img_filename = f"image_{idx}.png"
        img.save(os.path.join(folder_path, img_filename))

        # Save label and color flag information
        labels.append([img_filename, label, flag])

    # Save the labels and color flags information to a CSV file
    labels_csv_path = os.path.join(folder_path, 'labels.csv')
    np.savetxt(labels_csv_path, labels, delimiter=',', fmt='%s', header="filename,label,flag", comments='')


#### Saving images (CelebA)

def save_celeba_images(dataset, folder_name, datadir):
    # Create the directory if it doesn't exist
    folder_path = os.path.join(datadir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    labels = []

    # Iterate through the dataset and save each image
    for idx, (img, label, sa) in enumerate(dataset):
        # Convert the tensor image to a PIL image
        img = transforms.ToPILImage()(img)
        img_filename = f"image_{idx}.png"
        img.save(os.path.join(folder_path, img_filename))

        # Save label and sensitive attribute information
        labels.append([img_filename, label] + sa.tolist())

    # Save the labels and sensitive attributes information to a CSV file
    labels_csv_path = os.path.join(folder_path, 'labels.csv')
    np.savetxt(labels_csv_path, labels, delimiter=',', fmt='%s', header="filename,label,sa1,sa2", comments='')


def gen_dataset(datadir, dset_name, feature, seed=42, isnumpy=False, **kwargs):
    if feature == 'classimb':
        if 'classimb_ratio' in kwargs:
            pass
        else:
            raise KeyError("Specify a classimbratio value in the config file")

    if dset_name == "dna":
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        y_trn -= 1  # First Class should be zero
        y_val -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val,
                                                                        x_tst, y_tst, num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))
        return fullset, valset, testset, num_cls


    elif dset_name in ["Community_Crime", "LawSchool_selcon"]:
        if dset_name == "Community_Crime":
            x_trn, y_trn = clean_communities_full(os.path.join(datadir, 'communities.scv'))
        elif dset_name == "LawSchool_selcon":
            x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, 'lawschool.csv'))
        else:
            raise NotImplementedError

        fullset = (x_trn, y_trn)
        data_dims = x_trn.shape[1]
        device = 'cpu'

        x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
            = get_slices(dset_name, fullset[0], fullset[1], device, 3)

        assert(val_classes == tst_classes)

        trainset = CustomDataset_WithId_SELCON(torch.from_numpy(x_trn).float().to(device),torch.from_numpy(y_trn).float().to(device))
        valset = CustomDataset_WithId_SELCON(torch.cat(x_val_list,dim=0), torch.cat(y_val_list,dim=0))
        testset = CustomDataset_WithId_SELCON(torch.cat(x_tst_list,dim=0), torch.cat(y_tst_list,dim=0))

        return trainset, valset, testset, val_classes

    elif dset_name in ["cadata","abalone","cpusmall",'LawSchool']:
        if dset_name == "cadata":
            trn_file = os.path.join(datadir, 'cadata.txt')
            x_trn, y_trn = libsvm_file_load(trn_file, dim=8)
        elif dset_name == "abalone":
            trn_file = os.path.join(datadir, 'abalone_scale.txt')
            x_trn, y_trn = libsvm_file_load(trn_file, 8)
        elif dset_name == "cpusmall":
            trn_file = os.path.join(datadir, 'cpusmall_scale.txt')
            x_trn, y_trn = libsvm_file_load(trn_file, 12)
        elif dset_name == 'LawSchool':
            x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, 'lawschool.csv'))

        # create train and test indices
        #train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.2, random_state=seed)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        sc_l = StandardScaler()
        y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn, (-1, 1))), (-1))
        y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val, (-1, 1))), (-1))
        y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst, (-1, 1))), (-1))

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn),if_reg=True)
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val),if_reg=True)
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst),if_reg=True)

        return fullset, valset, testset, 1

    elif dset_name == 'MSD':

        trn_file = os.path.join(datadir, 'YearPredictionMSD')
        x_trn, y_trn = libsvm_file_load(trn_file, 90)

        tst_file = os.path.join(datadir, 'YearPredictionMSD.t')
        x_tst, y_tst = libsvm_file_load(tst_file, 90)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.005, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        sc_l = StandardScaler()
        y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn, (-1, 1))), (-1))
        y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val, (-1, 1))), (-1))
        y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst, (-1, 1))), (-1))

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn),if_reg=True)
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val),if_reg=True)
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst),if_reg=True)

        return fullset, valset, testset, 1
        
    elif dset_name == "adult":
        trn_file = os.path.join(datadir, 'a9a.trn')
        tst_file = os.path.join(datadir, 'a9a.tst')
        data_dims = 123
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        y_trn[y_trn < 0] = 0
        y_tst[y_tst < 0] = 0

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val,
                                                                        x_tst, y_tst, num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "connect_4":
        trn_file = os.path.join(datadir, 'connect_4.trn')

        data_dims = 126
        num_cls = 3

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        # The class labels are (-1,0,1). Make them to (0,1,2)
        y_trn[y_trn < 0] = 2

        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val,
                                                                        x_tst, y_tst, num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "letter":
        trn_file = os.path.join(datadir, 'letter.scale.trn')
        val_file = os.path.join(datadir, 'letter.scale.val')
        tst_file = os.path.join(datadir, 'letter.scale.tst')
        data_dims = 16
        num_cls = 26
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "satimage":
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "svmguide1":
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_val[y_val < 0] = 0
        y_tst[y_tst < 0] = 0

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "sklearn-digits":
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=seed)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name in ['prior_shift_large_linsep_4', 'conv_shift_large_linsep_4', 'red_large_linsep_4',
                       'expand_large_linsep_4',
                       'shrink_large_linsep_4', 'red_conv_shift_large_linsep_4', "linsep_4", "large_linsep_4"]:
        trn_file = os.path.join(datadir, dset_name + '.trn')
        val_file = os.path.join(datadir, dset_name + '.val')
        tst_file = os.path.join(datadir, dset_name + '.tst')
        data_dims = 2
        num_cls = 4
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name in ['prior_shift_clf_2', 'prior_shift_gauss_2', 'conv_shift_clf_2', 'conv_shift_gauss_2', "gauss_2",
                       "clf_2", "linsep"]:
        trn_file = os.path.join(datadir, dset_name + '.trn')
        val_file = os.path.join(datadir, dset_name + '.val')
        tst_file = os.path.join(datadir, dset_name + '.tst')
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name == "covertype":
        trn_file = os.path.join(datadir, 'covtype.data')
        data_dims = 54
        num_cls = 7
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)

        y_trn -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.2, random_state=seed)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name == "adult":
        # trn_file = os.path.join(datadir, 'adult.data')
        data_dims = 14
        num_cls = 2
        trn_file = os.path.join(datadir, 'adult.data')
        tst_file = os.path.join(datadir, 'adult.test')
        x_trn, y_trn = adult_load(trn_file, dim=data_dims)
        x_tst, y_tst = adult_load(tst_file, dim=data_dims)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        
        #X, y = census_load(trn_file, dim=data_dims)
        #x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
        sc = StandardScaler()
        unchanged_column_indices = [0, 8, 9] 
    
        # Extract the columns to keep unchanged
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]

        # Extract the columns to be transformed (all columns except the unchanged ones)
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        # Fit and transform the columns to be transformed
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        
        # Concatenate the transformed columns with the unchanged columns in the correct order
        x_trn = np.insert(x_trn_transformed, [0,7,7], x_trn_unchanged, axis=1)
        x_val = np.insert(x_val_transformed, [0,7,7], x_val_unchanged, axis=1)
        x_tst = np.insert(x_tst_transformed, [0,7,7], x_tst_unchanged, axis=1)
        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
    
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = CustomCensusDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomCensusDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomCensusDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))
        
        return fullset, valset, testset, num_cls

    #######################################################################
    ########## Adult 

    elif dset_name == "save-adult":
         data_dims = 14
         num_cls = 2
         trn_file = os.path.join(datadir, 'adult.data')
         tst_file = os.path.join(datadir, 'adult.test')
         # Load the training and test data
         x_trn, y_trn = adult_load(trn_file, dim=data_dims)
         x_tst, y_tst = adult_load(tst_file, dim=data_dims)
         x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
         sc = StandardScaler()
         unchanged_column_indices = [0, 8, 9]
         # Extract the columns to keep unchanged
         x_trn_unchanged = x_trn[:, unchanged_column_indices]
         x_val_unchanged = x_val[:, unchanged_column_indices]
         x_tst_unchanged = x_tst[:, unchanged_column_indices]
         # Extract the columns to be transformed (all columns except the unchanged ones)
         x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
         x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
         x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
         # Fit and transform the columns to be transformed
         x_trn_transformed = sc.fit_transform(x_trn_to_transform)
         x_val_transformed = sc.transform(x_val_to_transform)
         x_tst_transformed = sc.transform(x_tst_to_transform)
         # Concatenate the transformed columns with the unchanged columns in the correct order
         x_trn = np.insert(x_trn_transformed, [0,7,7], x_trn_unchanged, axis=1)
         x_val = np.insert(x_val_transformed, [0,7,7], x_val_unchanged, axis=1)
         x_tst = np.insert(x_tst_transformed, [0,7,7], x_tst_unchanged, axis=1)
         if feature == 'classimb':
             x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                    num_cls, kwargs['classimb_ratio'], seed=seed)
         elif feature == 'noise':
             y_trn = create_noisy(y_trn, num_cls, seed=seed)
         columns =  [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
         columns.append('income')
         # Save the datasets as CSV files
         train_df = pd.DataFrame(np.column_stack((x_trn, y_trn)), columns=columns)
         train_df.to_csv(os.path.join(datadir, 'train_adult.csv'), index=False)
         val_df = pd.DataFrame(np.column_stack((x_val, y_val)), columns=columns)
         val_df.to_csv(os.path.join(datadir, 'val_adult.csv'), index=False)
         test_df = pd.DataFrame(np.column_stack((x_tst, y_tst)), columns=columns)
         test_df.to_csv(os.path.join(datadir, 'test_adult.csv'), index=False)
         if isnumpy:
             fullset = (x_trn, y_trn)
             valset = (x_val, y_val)
             testset = (x_tst, y_tst)
         else:
             fullset = CustomCensusDataset(torch.from_numpy(x_trn).float(), torch.from_numpy(y_trn).long())
             valset = CustomCensusDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
             testset = CustomCensusDataset(torch.from_numpy(x_tst).float(), torch.from_numpy(y_tst).long())
         return fullset, valset, testset, num_cls
    
    
    elif dset_name == "kdd":
        data_file = os.path.join(datadir, 'KDD_preprocessed.csv')
        df = pd.read_csv(data_file)
        x_trn = df.drop(columns=['income']).values
        y_trn = df['income'].values
        data_dims = len(df.columns) - 1
        num_cls = 2
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        sc = StandardScaler()
        
        # Define the list of sensitive attributes to keep unchanged
        columns_to_find = ['race', 'sex', 'age']
        unchanged_column_indices = [df.columns.get_loc(col) for col in columns_to_find]
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]

        # Extract the columns to be transformed (all columns except the unchanged ones)
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        # Fit and transform the columns to be transformed
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        
        # Concatenate the transformed columns with the unchanged columns
        x_trn = np.concatenate((x_trn_transformed, x_trn_unchanged), axis=1)
        x_val = np.concatenate((x_val_transformed, x_val_unchanged), axis=1)
        x_tst = np.concatenate((x_tst_transformed, x_tst_unchanged), axis=1)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
    
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = DatasetWithSensitiveAttributes(torch.from_numpy(x_trn), 
                                                     torch.from_numpy(y_trn), 
                                                     unchanged_column_indices)
            valset = DatasetWithSensitiveAttributes(torch.from_numpy(x_val), 
                                                    torch.from_numpy(y_val), 
                                                    unchanged_column_indices)
            testset = DatasetWithSensitiveAttributes(torch.from_numpy(x_tst), 
                                                     torch.from_numpy(y_tst),
                                                     unchanged_column_indices)
        return fullset, valset, testset, num_cls
    
    ###############################################################################################################
    ################ Sauvegarde KDD 
    elif dset_name == "save-kdd":
         data_file = os.path.join(datadir, 'KDD_preprocessed.csv')
         df = pd.read_csv(data_file)
         original_columns = df.columns.tolist()
         x = df.drop(columns=['income']).values
         y = df['income'].values
         data_dims = len(df.columns) - 1
         num_cls = 2

         x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.1, random_state=seed)
         x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)

         sc = StandardScaler()
         columns_to_find = ['race', 'sex', 'age']
         unchanged_column_indices = [df.columns.get_loc(col) for col in columns_to_find]

         x_trn_unchanged = x_trn[:, unchanged_column_indices]
         x_val_unchanged = x_val[:, unchanged_column_indices]
         x_tst_unchanged = x_tst[:, unchanged_column_indices]

         x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
         x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
         x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)

         x_trn_transformed = sc.fit_transform(x_trn_to_transform)
         x_val_transformed = sc.transform(x_val_to_transform)
         x_tst_transformed = sc.transform(x_tst_to_transform)

         x_trn = np.concatenate((x_trn_transformed, x_trn_unchanged), axis=1)
         x_val = np.concatenate((x_val_transformed, x_val_unchanged), axis=1)
         x_tst = np.concatenate((x_tst_transformed, x_tst_unchanged), axis=1)

         if feature == 'classimb':
             x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                    num_cls, kwargs['classimb_ratio'], seed=seed)
         elif feature == 'noise':
             y_trn = create_noisy(y_trn, num_cls, seed=seed)

         columns = original_columns[:-1]
    
         # Save train set
         train_df = pd.DataFrame(x_trn, columns=columns)
         train_df['income'] = y_trn
         train_df.to_csv(os.path.join(datadir, 'train_kdd.csv'), index=False)
    
         # Save validation set
         val_df = pd.DataFrame(x_val, columns=columns)
         val_df['income'] = y_val
         val_df.to_csv(os.path.join(datadir, 'val_kdd.csv'), index=False)
    
         # Save test set
         test_df = pd.DataFrame(x_tst, columns=columns)
         test_df['income'] = y_tst
         test_df.to_csv(os.path.join(datadir, 'test_kdd.csv'), index=False)

         if isnumpy:
             fullset = (x_trn, y_trn)
             valset = (x_val, y_val)
             testset = (x_tst, y_tst)
         else:
             fullset = DatasetWithSensitiveAttributes(torch.from_numpy(x_trn), 
                                                 torch.from_numpy(y_trn), 
                                                 unchanged_column_indices)
             valset = DatasetWithSensitiveAttributes(torch.from_numpy(x_val), 
                                                torch.from_numpy(y_val), 
                                                unchanged_column_indices)
             testset = DatasetWithSensitiveAttributes(torch.from_numpy(x_tst), 
                                                 torch.from_numpy(y_tst),
                                                 unchanged_column_indices)

         return fullset, valset, testset, num_cls

    ################################################################################################################
    ################################################################################################################


    elif dset_name == "ars":
        trn_file = os.path.join(datadir, 'ARS.csv')
        data_dims = 9
        num_cls = 2
        df = pd.read_csv(trn_file)
        X = df.drop(columns=['activity']).values
        y = df['activity'].values

        x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
        x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
        sc = StandardScaler()
        unchanged_column_indices = [7] 

        # Extract the columns to keep unchanged
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]

        # Extract the columns to be transformed (all columns except the unchanged ones)
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        # Fit and transform the columns to be transformed
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        
        # Concatenate the transformed columns with the unchanged columns in the correct order
        x_trn = np.insert(x_trn_transformed, unchanged_column_indices, x_trn_unchanged, axis=1)
        x_val = np.insert(x_val_transformed, unchanged_column_indices, x_val_unchanged, axis=1)
        x_tst = np.insert(x_tst_transformed, unchanged_column_indices, x_tst_unchanged, axis=1)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
    
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = ARSDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = ARSDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = ARSDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls
    
    
    ############################################################################
    ### Saving ARS
    ############################################################################

    elif dset_name == "save-ars":
         trn_file = os.path.join(datadir, 'ARS.csv')
         data_dims = 9
         num_cls = 2
         df = pd.read_csv(trn_file)
         original_columns = df.columns.tolist()
         X = df.drop(columns=['activity']).values
         y = df['activity'].values

         x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
         x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

         sc = StandardScaler()
         unchanged_column_indices = [7]
    
         # Extract the columns to keep unchanged
         x_trn_unchanged = x_trn[:, unchanged_column_indices]
         x_val_unchanged = x_val[:, unchanged_column_indices]
         x_tst_unchanged = x_tst[:, unchanged_column_indices]

         # Extract the columns to be transformed (all columns except the unchanged ones)
         x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
         x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
         x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)

         # Fit and transform the columns to be transformed
         x_trn_transformed = sc.fit_transform(x_trn_to_transform)
         x_val_transformed = sc.transform(x_val_to_transform)
         x_tst_transformed = sc.transform(x_tst_to_transform)

         # Concatenate the transformed columns with the unchanged columns in the correct order
         x_trn = np.insert(x_trn_transformed, unchanged_column_indices, x_trn_unchanged, axis=1)
         x_val = np.insert(x_val_transformed, unchanged_column_indices, x_val_unchanged, axis=1)
         x_tst = np.insert(x_tst_transformed, unchanged_column_indices, x_tst_unchanged, axis=1)

         if feature == 'classimb':
             x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                    num_cls, kwargs['classimb_ratio'], seed=seed)
         elif feature == 'noise':
             y_trn = create_noisy(y_trn, num_cls, seed=seed)

         columns = original_columns[:-1]
    
         # Save the datasets as CSV files
         train_df = pd.DataFrame(x_trn, columns=columns)
         train_df['activity'] = y_trn
         train_df.to_csv(os.path.join(datadir, 'train_ars.csv'), index=False)

         val_df = pd.DataFrame(x_val, columns=columns)
         val_df['activity'] = y_val
         val_df.to_csv(os.path.join(datadir, 'val_ars.csv'), index=False)

         test_df = pd.DataFrame(x_tst, columns=columns)
         test_df['activity'] = y_tst
         test_df.to_csv(os.path.join(datadir, 'test_ars.csv'), index=False)

         if isnumpy:
             fullset = (x_trn, y_trn)
             valset = (x_val, y_val)
             testset = (x_tst, y_tst)
         else:
             fullset = ARSDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
             valset = ARSDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
             testset = ARSDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

         return fullset, valset, testset, num_cls

    ############################################################################
    ############################################################################

    elif dset_name == "mobiact":
        trn_file = os.path.join(datadir, 'mobiact.csv')
        data_dims = 12
        num_cls = 2
        df = pd.read_csv(trn_file)
        X = df.drop(columns=['activity']).values
        y = df['activity'].values

        x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
        sc = StandardScaler()
        unchanged_column_indices = [10,11] 

        # Extract the columns to keep unchanged
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]

        # Extract the columns to be transformed (all columns except the unchanged ones)
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        # Fit and transform the columns to be transformed
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        
        # Concatenate the transformed columns with the unchanged columns in the correct order
        x_trn = np.insert(x_trn_transformed, [10,10], x_trn_unchanged, axis=1)
        x_val = np.insert(x_val_transformed, [10,10], x_val_unchanged, axis=1)
        x_tst = np.insert(x_tst_transformed, [10,10], x_tst_unchanged, axis=1)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
    
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = mobiactDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = mobiactDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = mobiactDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    ##################################################################################################################
    ####### Saving training, test and validation sets
    ##################################################################################################################
    elif dset_name == "save-mobiact":
        trn_file = os.path.join(datadir, 'mobiact.csv')
        data_dims = 12
        num_cls = 2
        df = pd.read_csv(trn_file)
        original_columns = df.columns.tolist()
        X = df.drop(columns=['activity']).values
        y = df['activity'].values
        x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
        sc = StandardScaler()
        unchanged_column_indices = [10, 11]
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        x_trn = np.insert(x_trn_transformed, [10, 10], x_trn_unchanged, axis=1)
        x_val = np.insert(x_val_transformed, [10, 10], x_val_unchanged, axis=1)
        x_tst = np.insert(x_tst_transformed, [10, 10], x_tst_unchanged, axis=1)
        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst, num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
        columns = original_columns[:-1]
        train_df = pd.DataFrame(x_trn, columns=columns)
        train_df['activity'] = y_trn
        train_df.to_csv(os.path.join(datadir, 'train_mobiact.csv'), index=False)
        val_df = pd.DataFrame(x_val, columns=columns)
        val_df['activity'] = y_val
        val_df.to_csv(os.path.join(datadir, 'val_mobiact.csv'), index=False)
        test_df = pd.DataFrame(x_tst, columns=columns)
        test_df['activity'] = y_tst
        test_df.to_csv(os.path.join(datadir, 'test_mobiact.csv'), index=False)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = mobiactDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = mobiactDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = mobiactDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls


    ##################################################################################################################


    elif dset_name == "dc":
        trn_file = os.path.join(datadir, 'dc.csv')
        data_dims = 11
        num_cls = 2
        df = pd.read_csv(trn_file)
        X = df.drop(columns=['occupation']).values
        y = df['occupation'].values

        x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
        x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
        sc = StandardScaler()
        unchanged_column_indices = [9,10] 

        # Extract the columns to keep unchanged
        x_trn_unchanged = x_trn[:, unchanged_column_indices]
        x_val_unchanged = x_val[:, unchanged_column_indices]
        x_tst_unchanged = x_tst[:, unchanged_column_indices]

        # Extract the columns to be transformed (all columns except the unchanged ones)
        x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
        x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
        x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)
        # Fit and transform the columns to be transformed
        x_trn_transformed = sc.fit_transform(x_trn_to_transform)
        x_val_transformed = sc.transform(x_val_to_transform)
        x_tst_transformed = sc.transform(x_tst_to_transform)
        
        # Concatenate the transformed columns with the unchanged columns in the correct order
        x_trn = np.insert(x_trn_transformed, [9,9], x_trn_unchanged, axis=1)
        x_val = np.insert(x_val_transformed, [9,9], x_val_unchanged, axis=1)
        x_tst = np.insert(x_tst_transformed, [9,9], x_tst_unchanged, axis=1)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                        num_cls, kwargs['classimb_ratio'], seed=seed)
        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls, seed=seed)
    
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = dcDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = dcDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = dcDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls
    
    ##########################################################################
    ##### Saving DC
    ##########################################################################
    elif dset_name == "save-dc":
         trn_file = os.path.join(datadir, 'dc.csv')
         data_dims = 11
         num_cls = 2
         df = pd.read_csv(trn_file)
         original_columns = df.columns.tolist()

         X = df.drop(columns=['occupation']).values
         y = df['occupation'].values

         x_trn, X_val_test, y_trn, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
         x_val, x_tst, y_val, y_tst = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)

         sc = StandardScaler()
         unchanged_column_indices = [9, 10]

         # Extract the columns to keep unchanged
         x_trn_unchanged = x_trn[:, unchanged_column_indices]
         x_val_unchanged = x_val[:, unchanged_column_indices]
         x_tst_unchanged = x_tst[:, unchanged_column_indices]

         # Extract the columns to be transformed (all columns except the unchanged ones)
         x_trn_to_transform = np.delete(x_trn, unchanged_column_indices, axis=1)
         x_val_to_transform = np.delete(x_val, unchanged_column_indices, axis=1)
         x_tst_to_transform = np.delete(x_tst, unchanged_column_indices, axis=1)

         # Fit and transform the columns to be transformed
         x_trn_transformed = sc.fit_transform(x_trn_to_transform)
         x_val_transformed = sc.transform(x_val_to_transform)
         x_tst_transformed = sc.transform(x_tst_to_transform)

         # Concatenate the transformed columns with the unchanged columns in the correct order
         x_trn = np.insert(x_trn_transformed, [9, 9], x_trn_unchanged, axis=1)
         x_val = np.insert(x_val_transformed, [9, 9], x_val_unchanged, axis=1)
         x_tst = np.insert(x_tst_transformed, [9, 9], x_tst_unchanged, axis=1)

         if feature == 'classimb':
             x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst,
                                                                    num_cls, kwargs['classimb_ratio'], seed=seed)
         elif feature == 'noise':
             y_trn = create_noisy(y_trn, num_cls, seed=seed)

         # Use the original columns for the transformed dataset, excluding the target column
         column_names = original_columns[:-1]
         column_names.append('occupation')  # Add the target column name

         # Save the datasets as CSV files
         train_df = pd.DataFrame(np.column_stack((x_trn, y_trn)), columns=column_names)
         train_df.to_csv(os.path.join(datadir, 'train_dc.csv'), index=False)

         val_df = pd.DataFrame(np.column_stack((x_val, y_val)), columns=column_names)
         val_df.to_csv(os.path.join(datadir, 'val_dc.csv'), index=False)

         test_df = pd.DataFrame(np.column_stack((x_tst, y_tst)), columns=column_names)
         test_df.to_csv(os.path.join(datadir, 'test_dc.csv'), index=False)

         if isnumpy:
             fullset = (x_trn, y_trn)
             valset = (x_val, y_val)
             testset = (x_tst, y_tst)
         else:
             fullset = dcDataset(torch.from_numpy(x_trn).float(), torch.from_numpy(y_trn).long())
             valset = dcDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
             testset = dcDataset(torch.from_numpy(x_tst).float(), torch.from_numpy(y_tst).long())

         return fullset, valset, testset, num_cls

    
    elif dset_name == "mnist":
        mnist_transform = transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_tst_transform = transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = ColoredDataset(torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform), classes=10, colors=[0, 1], std=0.3)
        testset = ColoredDataset(torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_tst_transform), classes=10, colors=[0, 1], std=0.3)
        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls
    
    ###################################################################################################
    ############# Saving MNIST
    ###################################################################################################

    elif dset_name == "save-mnist":
         # Define the transformations
         mnist_transform = transforms.Compose([
             torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))
         ])

         mnist_tst_transform = transforms.Compose([
             torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))
         ])
    
         num_cls = 10
         # Load the full training and test sets with transformations
         fullset = ColoredDataset(torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform), classes=10, colors=[0, 1], std=0.3)
         testset = ColoredDataset(torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_tst_transform), classes=10, colors=[0, 1], std=0.3)

         if feature == 'classimb':
             rng = np.random.default_rng(seed)
             samples_per_class = torch.zeros(num_cls)
             for i in range(num_cls):
                 samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
             min_samples = int(torch.min(samples_per_class) * 0.1)
             selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
             subset_idxs = []
             for i in range(num_cls):
                 if i in selected_classes:
                     batch_subset_idxs = list(rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                 else:
                     batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                 subset_idxs.extend(batch_subset_idxs)
             fullset = torch.utils.data.Subset(fullset, subset_idxs)

         # Split the full training set into training and validation subsets
         validation_set_fraction = 0.1
         num_fulltrn = len(fullset)
         num_val = int(num_fulltrn * validation_set_fraction)
         num_trn = num_fulltrn - num_val
         trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

         # Save the datasets to their respective folders
         save_colored_mnist(trainset, 'MNIST/train_mnist', datadir)
         save_colored_mnist(valset, 'MNIST/valid_mnist', datadir)
         save_colored_mnist(testset, 'MNIST/test_mnist', datadir)

         print("MNIST dataset images and labels saved successfully.")

         return trainset, valset, testset, num_cls
    



    elif dset_name == "fashion-mnist":
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_tst_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10

        fullset = torchvision.datasets.FashionMNIST(root=datadir, train=True, download=True,
                                                    transform=mnist_transform)
        testset = torchvision.datasets.FashionMNIST(root=datadir, train=False, download=True,
                                                    transform=mnist_tst_transform)

        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls

    elif dset_name == "cifar10":
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        num_cls = 10

        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True,
                                               transform=cifar_tst_transform)

        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls


    elif dset_name == "tinyimagenet":
        tiny_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(90),
            # transforms.RandomGrayscale(),
            # transforms.RandomPerspective(),
            transforms.ToTensor(),
            # transforms.RandomErasing(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # tiny_transform = transforms.Compose([
        #     transforms.Resize(256), # Resize images to 256 x 256
        #     transforms.CenterCrop(224), # Center crop image
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        size = int(64 * 1.15)
        tiny_tst_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # tiny_tst_transform = transforms.Compose([
        #     transforms.Resize(256), # Resize images to 256 x 256
        #     transforms.CenterCrop(224), # Center crop image
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        num_cls = 200
        fullset = TinyImageNet(root=datadir, split='train', download=True, transform=tiny_transform)
        testset = TinyImageNet(root=datadir, split='val', download=True, transform=tiny_tst_transform)

        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls

    elif dset_name == "cifar100":
        cifar100_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        cifar100_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        num_cls = 100

        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        testset = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True,
                                                transform=cifar100_tst_transform)

        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls


    elif dset_name == "svhn":
        svhn_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        svhn_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        num_cls = 10
        fullset = torchvision.datasets.SVHN(root=datadir, split='train', download=True, transform=svhn_transform)
        testset = torchvision.datasets.SVHN(root=datadir, split='test', download=True, transform=svhn_tst_transform)

        if feature == 'classimb':
            rng = np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls


    elif dset_name == "kmnist":
        kmnist_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
        ])

        kmnist_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
        ])

        num_cls = 10

        fullset = torchvision.datasets.KMNIST(root=datadir, train=True, download=True, transform=kmnist_transform)
        testset = torchvision.datasets.KMNIST(root=datadir, train=False, download=True,
                                              transform=kmnist_tst_transform)

        if feature == 'classimb':
            rng=np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls


    elif dset_name == "stl10":
        stl10_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        stl10_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        num_cls = 10

        fullset = torchvision.datasets.STL10(root=datadir, split='train', download=True, transform=stl10_transform)
        testset = torchvision.datasets.STL10(root=datadir, split='test', download=True, transform=stl10_tst_transform)

        if feature == 'classimb':
            rng=np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls


    elif dset_name == "emnist":
        emnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        emnist_tst_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        num_cls = 10

        fullset = torchvision.datasets.EMNIST(root=datadir, split='digits', train=True, download=True,
                                              transform=emnist_transform)
        testset = torchvision.datasets.EMNIST(root=datadir, split='digits', train=False, download=True,
                                              transform=emnist_tst_transform)

        if feature == 'classimb':
            rng=np.random.default_rng(seed)
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            rng.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls

    elif dset_name == "celeba":
         crop_size = 108
         re_size = 32
         offset_height = (218 - crop_size) // 2
         offset_width = (178 - crop_size) // 2
         crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
         transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Lambda(crop),
             transforms.ToPILImage(),
             transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
         ])

         num_cls = 2
         img_dir = os.path.join(datadir, 'img_align_celeba')
         attr_file = os.path.join(datadir, 'list_attr_celeba.txt')
    
    
         # Load the attribute file into a DataFrame
         attr_df = pd.read_csv(attr_file, delim_whitespace=True, header=1)
         attr_df.reset_index(inplace=True)
         attr_df.rename(columns={'index': 'image_id'}, inplace=True)

         # Split the DataFrame into train, validation, and test sets based on the partitions
         partition_file = os.path.join(datadir, 'list_eval_partition.txt')
         partition_df = pd.read_csv(partition_file, delim_whitespace=True, header=None, names=['image_id', 'partition'])

         # Merge the partition info with the attributes DataFrame
         attr_df = pd.merge(attr_df, partition_df, on='image_id')

         # Create datasets based on the partition values
         train_df = attr_df[attr_df['partition'] == 0].reset_index(drop=True)
         val_df = attr_df[attr_df['partition'] == 1].reset_index(drop=True)
         test_df = attr_df[attr_df['partition'] == 2].reset_index(drop=True)

         # Instantiate the custom dataset using the transformed DataFrame and image paths
         trainset = celeba(img_dir=img_dir, attr_df=train_df, transform=transform)
         valset = celeba(img_dir=img_dir, attr_df=val_df, transform=transform)
         testset = celeba(img_dir=img_dir, attr_df=test_df, transform=transform)

         return trainset, valset, testset, num_cls


    ########### Saving CelebA
    #####################################################

    elif dset_name == "save-celeba":
         crop_size = 108
         re_size = 32
         offset_height = (218 - crop_size) // 2
         offset_width = (178 - crop_size) // 2
         crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
         transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Lambda(crop),
             transforms.ToPILImage(),
             transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
         ])
    
         num_cls = 2

         # Load the full training, validation, and test sets with transformations
         trainset = MyImageDataset(torchvision.datasets.CelebA(datadir, download=False, split='train', transform=transform, target_type="attr"))
         valset = MyImageDataset(torchvision.datasets.CelebA(datadir, download=False, split='valid', transform=transform, target_type="attr"))
         testset = MyImageDataset(torchvision.datasets.CelebA(datadir, download=False, split='test', transform=transform, target_type="attr"))

         # Apply class imbalance if specified
         if feature == 'classimb':
             rng = np.random.default_rng(seed)
             samples_per_class = torch.zeros(num_cls)
             for i in range(num_cls):
                 samples_per_class[i] = len(torch.where(trainset.dataset.attr[:, 31] == i)[0])
             min_samples = int(torch.min(samples_per_class) * 0.1)
             selected_classes = rng.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls), replace=False)
             subset_idxs = []
             for i in range(num_cls):
                 if i in selected_classes:
                     batch_subset_idxs = list(
                         rng.choice(torch.where(trainset.dataset.attr[:, 31] == i)[0].cpu().numpy(), size=min_samples, replace=False))
                 else:
                     batch_subset_idxs = list(torch.where(trainset.dataset.attr[:, 31] == i)[0].cpu().numpy())
                 subset_idxs.extend(batch_subset_idxs)
             trainset = torch.utils.data.Subset(trainset, subset_idxs)

         # Save the datasets to their respective folders
         save_celeba_images(trainset, 'celeba/train_celeba', datadir)
         save_celeba_images(valset, 'celeba/valid_celeba', datadir)
         save_celeba_images(testset, 'celeba/test_celeba', datadir)

         print("CelebA dataset images and labels saved successfully.")


    elif dset_name == "fairface":
        datadir = datadir + '/fairface/'
        label_train = pd.read_csv(datadir + 'fairface_label_train.csv')
        label_val = pd.read_csv(datadir + 'fairface_label_val.csv')
        label_val, label_test = train_test_split(label_val, test_size=0.5, random_state=42, stratify=label_val['service_test'])

        crop_size = 168
        re_size = 48
        offset_height = (224 - crop_size) // 2
        offset_width = (224 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        num_cls = 2
        
        trainset = fairfaceDataset(dataframe = label_train, transform = transform, datadir = datadir)
        valset = fairfaceDataset(dataframe = label_val, transform = transform, datadir = datadir)
        testset = fairfaceDataset(dataframe = label_test, transform = transform, datadir = datadir)


        return trainset, valset, testset, num_cls
        
    elif dset_name == "sst2" or dset_name == "sst2_facloc":
        '''
        download data/SST from https://drive.google.com/file/d/14KU6RQJpP6HKKqVGm0OF3MVxtI0NlEcr/view?usp=sharing
        or get the stanford sst data and make phrase_ids.<dev/test/train>.txt files
        pass datadir arg in dataset in config appropiriately(should look like ......../SST)
        '''
        num_cls = 2
        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        trainset = SSTDataset(datadir, 'train', num_cls, wordvec_dim, wordvec)
        testset = SSTDataset(datadir, 'test', num_cls, wordvec_dim, wordvec)
        valset = SSTDataset(datadir, 'dev', num_cls, wordvec_dim, wordvec)
        return trainset, valset, testset, num_cls

    elif dset_name == "glue_sst2":
        num_cls = 2
        raw = load_dataset("glue", "sst2")

        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)

        clean_type = 0
        fullset = GlueDataset(raw['train'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        # testset = GlueDataset(raw['test'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec) # doesn't have gold labels
        valset = GlueDataset(raw['validation'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec)

        test_set_fraction = 0.05
        seed = 42
        num_fulltrn = len(fullset)
        num_test = int(num_fulltrn * test_set_fraction)
        num_trn = num_fulltrn - num_test
        trainset, testset = random_split(fullset, [num_trn, num_test], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls

    elif  dset_name == "sst5":
        '''
        download data/SST from https://drive.google.com/file/d/14KU6RQJpP6HKKqVGm0OF3MVxtI0NlEcr/view?usp=sharing
        or get the stanford sst data and make phrase_ids.<dev/test/train>.txt files
        pass datadir arg in dataset in config appropiriately(should look like ......../SST)
        '''
        num_cls = 5
        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)
        trainset = SSTDataset(datadir, 'train', num_cls, wordvec_dim, wordvec)
        testset = SSTDataset(datadir, 'test', num_cls, wordvec_dim, wordvec)
        valset = SSTDataset(datadir, 'dev', num_cls, wordvec_dim, wordvec)
        return trainset, valset, testset, num_cls

    elif dset_name == 'trec6':
        num_cls = 6
        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)

        cls_to_num = {"DESC": 0, "ENTY": 1, "HUM": 2, "ABBR": 3, "LOC": 4, "NUM": 5}

        trainset = Trec6Dataset(datadir+'train.txt', cls_to_num, num_cls, wordvec_dim, wordvec)
        testset = Trec6Dataset(datadir+'test.txt', cls_to_num, num_cls, wordvec_dim, wordvec)
        valset = Trec6Dataset(datadir+'valid.txt', cls_to_num, num_cls, wordvec_dim, wordvec)
        return trainset, valset, testset, num_cls

    elif dset_name == "hf_trec6": # hugging face trec6
        num_cls = 6
        raw = load_dataset("trec")

        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        data_dir = kwargs['dataset'].datadir
        if not os.path.exists(os.path.abspath(data_dir)):
            os.makedirs(os.path.abspath(data_dir), exist_ok=True)
        if (os.path.exists(os.path.join(os.path.abspath(data_dir), 'trainset.pkl'))) and (os.path.exists(os.path.join(os.path.abspath(data_dir), 'valset.pkl'))) and (os.path.exists(os.path.join(os.path.abspath(data_dir), 'testset.pkl'))):
            trainset = torch.load(os.path.join(os.path.abspath(data_dir), 'trainset.pkl'))
            valset = torch.load(os.path.join(os.path.abspath(data_dir), 'valset.pkl'))
            testset = torch.load(os.path.join(os.path.abspath(data_dir), 'testset.pkl'))
        else:
            weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
            wordvec = loadGloveModel(weight_full_path)
            clean_type = 1
            fullset = GlueDataset(raw['train'], 'text', 'coarse_label', clean_type, num_cls, wordvec_dim, wordvec)
            testset = GlueDataset(raw['test'], 'text', 'coarse_label', clean_type, num_cls, wordvec_dim, wordvec)
            validation_set_fraction = 0.1
            seed = 42
            num_fulltrn = len(fullset)
            num_val = int(num_fulltrn * validation_set_fraction)
            num_trn = num_fulltrn - num_val    
            trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
            torch.save(trainset, os.path.join(os.path.abspath(data_dir), 'trainset.pkl'))
            torch.save(valset, os.path.join(os.path.abspath(data_dir), 'valset.pkl'))
            torch.save(testset, os.path.join(os.path.abspath(data_dir), 'testset.pkl'))
        return trainset, valset, testset, num_cls
    
    elif dset_name == "imdb": # hugging face trec6
        num_cls = 2
        raw = load_dataset("imdb")

        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)

        clean_type = 1
        fullset = GlueDataset(raw['train'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        testset = GlueDataset(raw['test'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        # valset = GlueDataset(raw['validation'], num_cls, wordvec_dim, wordvec)
        
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

        return trainset, valset, testset, num_cls

    elif dset_name == 'rotten_tomatoes':
        num_cls = 2
        raw = load_dataset("rotten_tomatoes")

        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)

        clean_type = 1
        trainset = GlueDataset(raw['train'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        valset = GlueDataset(raw['validation'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        testset = GlueDataset(raw['test'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        
        return trainset, valset, testset, num_cls

    elif dset_name == 'tweet_eval':
        num_cls = 20
        raw = load_dataset("tweet_eval", "emoji")

        wordvec_dim = kwargs['dataset'].wordvec_dim
        weight_path = kwargs['dataset'].weight_path
        weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
        wordvec = loadGloveModel(weight_full_path)

        clean_type = 1
        trainset = GlueDataset(raw['train'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        valset = GlueDataset(raw['validation'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        testset = GlueDataset(raw['test'], 'text', 'label', clean_type, num_cls, wordvec_dim, wordvec)
        
        return trainset, valset, testset, num_cls

    elif dset_name == "glue_sst2_transformer":
        """
        Load GLUE SST2 dataset. We are only using train and validation splits since the test split doesn't come with gold labels. For testing purposes, we use 5% of train
        dataset as test dataset.
        """
        num_cls = 2
        glue_dataset = load_dataset("glue", "sst2")
        tokenizer_mapping = lambda example: tokenize_function(kwargs['tokenizer'], example, SENTENCE_MAPPINGS['glue_sst2'])
        glue_dataset = glue_dataset.map(tokenizer_mapping, batched=True) 
        glue_dataset = glue_dataset.remove_columns([SENTENCE_MAPPINGS['glue_sst2'], "idx"])
        glue_dataset = glue_dataset.rename_column(LABEL_MAPPINGS['glue_sst2'], "labels")
        glue_dataset.set_format("torch")

        fullset = glue_dataset['train']
        valset = glue_dataset['validation']
        test_set_fraction = 0.05
        seed = 42
        num_fulltrn = len(fullset)
        num_test = int(num_fulltrn * test_set_fraction)
        num_trn = num_fulltrn - num_test
        trainset, testset = random_split(fullset, [num_trn, num_test], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls

    elif dset_name == 'hf_trec6_transformer':
        num_cls = 6
        trec6_dataset = load_dataset("trec")
        
        tokenizer_mapping = lambda example: tokenize_function(kwargs['tokenizer'], example, SENTENCE_MAPPINGS['hf_trec6'])
        trec6_dataset = trec6_dataset.map(tokenizer_mapping, batched=True) 
        trec6_dataset = trec6_dataset.remove_columns([SENTENCE_MAPPINGS['hf_trec6'], 'fine_label'])
        trec6_dataset = trec6_dataset.rename_column(LABEL_MAPPINGS['hf_trec6'], "labels")
        trec6_dataset.set_format("torch")

        fullset = trec6_dataset["train"]
        testset = trec6_dataset['test']
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls
        
    elif dset_name == 'imdb_transformer':
        num_cls = 2
        imdb_dataset = load_dataset("imdb")

        tokenizer_mapping = lambda example: tokenize_function(kwargs['tokenizer'], example, SENTENCE_MAPPINGS['imdb'])
        imdb_dataset = imdb_dataset.map(tokenizer_mapping, batched=True) 
        imdb_dataset = imdb_dataset.remove_columns([SENTENCE_MAPPINGS['imdb']])
        imdb_dataset = imdb_dataset.rename_column(LABEL_MAPPINGS['imdb'], "labels")
        imdb_dataset.set_format("torch")

        fullset = imdb_dataset["train"]
        testset = imdb_dataset['test']
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
        return trainset, valset, testset, num_cls
    
    elif dset_name == 'rotten_tomatoes_transformer':
        num_cls = 2
        dataset = load_dataset("rotten_tomatoes")
        tokenizer_mapping = lambda example: tokenize_function(kwargs['tokenizer'], example, SENTENCE_MAPPINGS['rotten_tomatoes'])
        dataset = dataset.map(tokenizer_mapping, batched=True) 
        dataset = dataset.remove_columns([SENTENCE_MAPPINGS['rotten_tomatoes']])
        dataset = dataset.rename_column(LABEL_MAPPINGS['rotten_tomatoes'], "labels")
        dataset.set_format("torch")

        trainset = dataset["train"]
        valset = dataset["validation"]
        testset = dataset['test']
        return trainset, valset, testset, num_cls

    elif dset_name == 'tweet_eval_transformer':
        num_cls = 20
        tweet_dataset = load_dataset("tweet_eval", "emoji")

        tokenizer_mapping = lambda example: tokenize_function(kwargs['tokenizer'], example, SENTENCE_MAPPINGS['tweet_eval'])
        tweet_dataset = tweet_dataset.map(tokenizer_mapping, batched=True) 
        tweet_dataset = tweet_dataset.remove_columns([SENTENCE_MAPPINGS['tweet_eval']])
        tweet_dataset = tweet_dataset.rename_column(LABEL_MAPPINGS['tweet_eval'], "labels")
        tweet_dataset.set_format("torch")

        trainset = tweet_dataset["train"]
        valset = tweet_dataset["validation"]
        testset = tweet_dataset['test']
        return trainset, valset, testset, num_cls

    else:
        raise NotImplementedError
