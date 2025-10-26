import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from os.path import join, exists
from os import mkdir 

from dataselection.utils.data.datasets.SL.builder import load_dataset, gen_dataset
from dataselection.utils.models import LogisticRegNet, SVMNet, MLPModel
from dataselection.utils.data.dataloader.SL.adaptive import (
    GLISTERDataLoader,
    CRAIGDataLoader,
    GradMatchDataLoader,
    RandomDataLoader,
)
from dotmap import DotMap
import pandas as pd
import logging

from bias_metrics import get_fair_metrics, get_fair_metrics_ars, get_fair_metrics_dc, get_fair_metrics_mobiact, get_fair_metrics_adult, get_fair_metrics_fairface, get_fair_metrics_voxceleb, get_fair_metrics_audioMNIST
from dataselection.utils.models import resnet, VGG, VGGM, LSTM_MFCC, AudioClassifier, AudioLSTM, AudioCNN

from tqdm import tqdm



def load_config(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return SimpleNamespace(**config_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    return load_config(args.config)


def evaluation(data_iter, model, args):
    model.eval()
    with torch.no_grad():
        corrects = 0
        for data, label, sensitive_attributes in data_iter:
            sentences = data.to(args.device, non_blocking=True)
            labels = label.to(args.device, non_blocking=True)
            logit = model(sentences)
            corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum().item()
        size = len(data_iter.dataset)
        model.train()
        return 100.0 * corrects / size


def evaluate_model (data_iter, model, args) : 
    model.eval()
    data_target = []
    data_pred = []

    with torch.no_grad():
        corrects = 0
        for data, labels, sensitive_attributes in data_iter:
            sentences = data.to(args.device, non_blocking=True)

            outputs = model(sentences)

            probabilities  = nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probabilities, dim=1).cpu().numpy() 

            if args.dataset_name =='ars':
                gender = sensitive_attributes.numpy()
            elif args.dataset_name =='dc' or args.dataset_name =='mobiact' or args.dataset_name == 'celeba':
                age = sensitive_attributes[:, 0].numpy()
                gender = sensitive_attributes[:, 1].numpy()
            elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                race = sensitive_attributes[:, 0].numpy()
                gender = sensitive_attributes[:, 1].numpy()
                age = sensitive_attributes[:, 2].numpy()
            elif args.dataset_name == 'fairface' : 
                age = sensitive_attributes[:, 0].numpy()
                race = sensitive_attributes[:, 1].numpy()
            elif args.dataset_name == 'voxceleb' :
                race = sensitive_attributes[:, 0].numpy()
            elif args.dataset_name == 'audiomnist':
                age = sensitive_attributes[:, 0].numpy()
                gender = sensitive_attributes[:, 1].numpy()


            if args.dataset_name =='ars':
                batch_data = zip(gender, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[0],  sample[2]))
                    data_target.append(sample[:2])
            elif args.dataset_name in ['dc', 'mobiact', 'audiomnist']:
                batch_data = zip(age, gender, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[0], sample[1], sample[3]))
                    data_target.append(sample[:3])
            elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                batch_data = zip(race, gender, age, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[0], sample[1], sample[2],sample[4]))
                    data_target.append(sample[:4])
            elif args.dataset_name == 'fairface' : 
                batch_data = zip(age, race, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[0], sample[1], sample[3]))
                    data_target.append((sample[0], sample[1], sample[2]))
            elif args.dataset_name == 'celeba' : 
                batch_data = zip(age, gender, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[3], sample[0], sample[1]))
                    data_target.append((sample[2], sample[0], sample[1]))
            elif args.dataset_name == 'voxceleb' :
                batch_data = zip(race, labels, pred)
                for sample in batch_data:
                    data_pred.append((sample[2], sample[0]))
                    data_target.append((sample[1], sample[0]))
            
        model.train()
    
    return data_target, data_pred 

def num_features(args):
    if args.dataset_name == 'ars':
        num_features = 9
    elif args.dataset_name == 'adult':
        num_features = 14
    elif args.dataset_name == 'dc':
        num_features = 11
    elif args.dataset_name == 'mobiact':
        num_features = 12
    elif args.dataset_name == 'kdd':
        num_features = 395
    elif args.dataset_name == 'celeba':
        num_features = 2
    elif args.dataset_name == 'fairface' : 
        num_features = 2
    elif args.dataset_name == 'voxceleb' :
        num_features = 64                                                                                                                                                                                                                         
    elif args.dataset_name == 'audiomnist':
        num_features = 13
    else:
        raise ValueError("Unknown dataset name")
    
    return num_features

def fair_metrics_func(args):
    if args.dataset_name == 'ars' :
        return get_fair_metrics_ars
    elif args.dataset_name == 'adult' or args.dataset_name == 'kdd':
        return get_fair_metrics_adult
    elif args.dataset_name == 'dc':
        return get_fair_metrics_dc
    elif args.dataset_name == 'mobiact':
        return get_fair_metrics_mobiact
    elif args.dataset_name == 'celeba':
        return get_fair_metrics
    elif args.dataset_name == 'fairface' : 
        return get_fair_metrics_fairface 
    elif args.dataset_name == 'voxceleb' :
        return get_fair_metrics_voxceleb 
    elif args.dataset_name == 'audiomnist':
        return get_fair_metrics_audioMNIST
    else:
        raise ValueError("Unknown dataset name")

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    args.device = device
    num_f = num_features(args)
    function = fair_metrics_func(args)
    
    if not exists(args.result_path) : 
        mkdir(args.result_path) 

    for model_name in args.models:
        print("*****************")
        print(model_name)
        print("*****************")
        fullset, valset, testset, num_cls = load_dataset(args.dataset_path, args.train_file, args.test_file, args.val_file,  args.data_load, isnumpy=False)
                
        training_iter = DataLoader(dataset=fullset,
                                    batch_size=args.batch_size,
                                    num_workers=0, shuffle=True, pin_memory=True)
    
        testing_iter = DataLoader(dataset=testset,
                                    batch_size=args.batch_size,
                                    num_workers=0, pin_memory=True)
        validation_iter = DataLoader(dataset=valset,
                                        batch_size=args.batch_size,
                                        num_workers=0, pin_memory=True)
        for ratio in args.ratios:
            args.fraction = ratio

            values_final = args.values
            for i in values_final:
                args.ss = i
                for x in range(args.runs): 
                    start = time.time()
                
                    # Datasets
            
                    if model_name == 'MLP' :
                        model = MLPModel(num_f, args.label_num).to(device)
                    elif model_name == 'SVM' :
                        model = SVMNet(num_f, args.label_num).to(device)
                    elif model_name == 'ResNet18':
                        model = resnet.ResNet18(num_f)
                        model.to(device) 
                    elif model_name == "VGG" :
                        model = VGG('VGG11') 
                        model.to(device)
                    elif model_name == "VGGM" :
                        model = VGGM() 
                        model.to(device) 
                    elif model_name == "lstm" : 
                        model = LSTM_MFCC() 
                        model.to(device)
                    elif model_name == "AudioClassifier" : 
                        model = AudioClassifier() 
                        model.to(device)
                    elif model_name == "AudioLSTM":
                        if args.dataset_name == "audiomnist":
                            f = 40  
                        else:
                            f = num_f
                        model = AudioLSTM(in_dim=f, num_classes=args.label_num)
                        model.to(device)

                    elif model_name == "AudioCNN":
                        # CNN 2D style VGG sur log-mels 
                        model = AudioCNN(num_classes=args.label_num)
                        model.to(device)
                    else :
                        model = LogisticRegNet(num_f, args.label_num).to(device)
                
                
                    criterion = nn.CrossEntropyLoss()
                    criterion_nored = nn.CrossEntropyLoss(reduction='none')
                    if args.optimizer == "SGD" :
                        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    elif args.optimizer == "Adam" : 
                        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 5e-4)

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.000001)


                    if args.ss == 1:
                        dss_args=dict(type="GradMatch",
                                            fraction=args.fraction,
                                            select_every=args.select_every,
                                            lam=0.5,
                                            selection_type='PerClassPerGradient',
                                            v1=True,
                                            valid=False,
                                            kappa=0,
                                            eps=1e-100,
                                            linear_layer=True,
                                            model=model,
                                            loss=criterion_nored,
                                            eta = args.lr,
                                            num_classes = args.label_num,
                                            device = args.device
                                            )
                    elif args.ss == 2:
                        dss_args=dict(type="GradMatchPB",
                                            fraction=args.fraction,
                                            select_every=args.select_every,
                                            lam=0,
                                            selection_type='PerBatch',
                                            v1=True,
                                            valid=False,
                                            eps=1e-100,
                                            linear_layer=True,
                                            kappa=0,
                                            model=model,
                                            loss=criterion_nored,
                                            eta = args.lr,
                                            num_classes = args.label_num,
                                            device = args.device
                                            )
                    elif args.ss == 3:
                        print("craig") 
                        dss_args=dict(type="CRAIGPB",
                                            fraction=args.fraction,
                                            select_every=args.select_every,
                                            lam=0,
                                            selection_type='PerBatch',
                                            v1=True,
                                            valid=False,
                                            eps=1e-100,
                                            linear_layer=False,
                                            kappa=0,
                                            model=model,
                                            if_convex = False,
                                            loss=criterion_nored,
                                            eta = args.lr,
                                            num_classes = args.label_num,
                                            device = args.device,
                                            optimizer = 'stochastic'
                                            )
                    elif args.ss == 4:
                        dss_args=dict(type="GLISTERPB",
                                            fraction=args.fraction,
                                            select_every=args.select_every,
                                            lam=0,
                                            selection_type='PerBatch',
                                            v1=True,
                                            valid=False,
                                            eps=1e-100,
                                            linear_layer=False,
                                            kappa=0,
                                            model=model,
                                            if_convex = False,
                                            loss=criterion_nored,
                                            eta = args.lr,
                                            num_classes = args.label_num,
                                            device = args.device,
                                            greedy='Stochastic',
                                            )
                    elif args.ss == 5:
                        dss_args=dict(type="Random",
                                            fraction=args.fraction,
                                            select_every=1,
                                            device = args.device,
                                            kappa = 0
                                            )
                
                    str_sys = "Full"
                
                    if args.ss == 1 or args.ss == 2:
                        logger = logging.getLogger(__name__)
                        dss_args = DotMap(dss_args)
                        dataloader = GradMatchDataLoader(training_iter, validation_iter, dss_args, logger,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                pin_memory=True,
                                                                collate_fn=None)
                        str_sys = "GradMatchPB"
                            
                    elif args.ss == 3:
                        logger = logging.getLogger(__name__)
                        dss_args = DotMap(dss_args)
                        dataloader = CRAIGDataLoader(training_iter, validation_iter, dss_args, logger,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                pin_memory=True,
                                                                collate_fn=None, 
                                                                model_name = model_name, 
                                                                ratio= args.fraction)
                        str_sys = "CRAIGPB"
                    elif args.ss == 4:
                        logger = logging.getLogger(__name__)
                        dss_args = DotMap(dss_args)
                        dataloader = GLISTERDataLoader(training_iter, validation_iter, dss_args, logger,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                pin_memory=True,
                                                                collate_fn=None)
                        str_sys = "GLISTERPB"
                    elif args.ss == 5:
                        logger = logging.getLogger(__name__)
                        dss_args = DotMap(dss_args)
                        dataloader = RandomDataLoader(training_iter, dss_args, logger,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                pin_memory=True,
                                                                collate_fn=None)
                        str_sys = "Random"
                        
                    step = 0
                    loss_sum = 0
                    best_acc = 0
                    best_epoch = 0

                    folder = join(args.result_path, str_sys) 
                    if not exists(folder) : 
                        mkdir(folder)
                        
                    if args.ss == 0 : 
                        res = join(folder, args.dataset_name + "_1")
                        print(res)
                        if not exists(res) :
                            mkdir(res) 
                    else :
                        res = join(folder, args.dataset_name + "_" + str(args.fraction))
                        print(res) 
                        if not exists(res) :
                            mkdir(res) 
                
                    run_start_time = time.time()
                    
                    fair_metrics = pd.DataFrame(columns=args.cols)
                    fair_metrics.index.name = "epoch"
                
                    cost_metric = pd.DataFrame(columns=['Model_training_time', 'Full_training_time', 'Loss'])
                    cost_metric.index.name = "epoch"
                
                    resample_metric = pd.DataFrame(columns=['Data_selection_execution_time'])
                    resample_metric.index.name = "epoch"
                
                    if args.ss > 0:
                        #for epoch in range(1, args.epoch+1):
                        for epoch in tqdm(range(1, args.epoch + 1), desc=f"Training {model_name}, run {x}"):
                            subtrn_loss = 0
                            subtrn_correct = 0.0
                            subtrn_total = 0.0
                            model.train()
                            start_time = time.time()
                            resample_time = 0.0
                            resample = False
                            data_target = []
                            data_pred = []
                            for _, data in enumerate(dataloader):
                                epoch_start_time = time.time()
                                if not(resample) and (epoch % args.select_every) == 1:
                                    resample_time = epoch_start_time - start_time
                                    resample = True
                
                                inputs, targets, sensitive_attributes, weights = data
                                inputs = inputs.float()  # Convert input tensor to Float
                                inputs = inputs.to(args.device)
                                targets = targets.to(args.device, non_blocking=True)
                                weights = weights.to(args.device)
                                optimizer.zero_grad()
                                outputs = model(inputs)
                                losses = criterion_nored(outputs, targets)
                                loss = torch.dot(losses, weights / (weights.sum()))
                                loss.backward()
                                subtrn_loss += loss.item()
                                loss_value = loss.item() 
                
                                loss_sum += subtrn_loss
                                if step % args.log_interval == 0:
                                    loss_sum = 0
                                    step = 0
                                step += 1
                
                                optimizer.step()
                                if args.scheduler : 
                                    scheduler.step() 

                                if args.scheduler : 
                                    scheduler.step() 
                                _, predicted = outputs.max(1)
                                subtrn_total += targets.size(0)
                                subtrn_correct += predicted.eq(targets).sum().item()
                                if args.dataset_name =='ars':
                                    gender = sensitive_attributes.numpy()
                                elif args.dataset_name =='dc' or args.dataset_name =='mobiact' or args.dataset_name == 'celeba':
                                        age = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        race = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                        age = sensitive_attributes[:, 2].numpy()
                                elif args.dataset_name == 'fairface' : 
                                        age = sensitive_attributes[:, 0].numpy()
                                        race = sensitive_attributes[:, 1].numpy()
                                elif args.dataset_name == 'voxceleb' :
                                        race = sensitive_attributes[:, 0].numpy()
                                        
                                targets = targets.cpu().numpy()
                                probabilities  = nn.functional.softmax(outputs, dim=1)
                
                                labels = torch.argmax(probabilities, dim=1).cpu().numpy() 
                
                                if args.dataset_name =='ars':
                                    batch_data = zip(gender, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[0], sample[2]))
                                        data_target.append(sample[:2])
                                elif args.dataset_name =='dc' or args.dataset_name =='mobiact' :
                                    batch_data = zip(age, gender, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[0], sample[1], sample[3]))
                                        data_target.append(sample[:3])
                                elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                    batch_data = zip(race, gender, age, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[0], sample[1], sample[2],sample[4]))
                                        data_target.append(sample[:4])
                                elif args.dataset_name == 'fairface' : 
                                    batch_data = zip(age, race, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[0], sample[1], sample[3]))
                                        data_target.append((sample[0], sample[1], sample[2]))
                                elif args.dataset_name == 'celeba' : 
                                    batch_data = zip(age, gender, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[3], sample[0], sample[1]))
                                        data_target.append((sample[2], sample[0], sample[1]))
                                elif args.dataset_name == 'voxceleb' :
                                    batch_data = zip(race, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[2], sample[0]))
                                        data_target.append((sample[1], sample[0]))
                
                            epoch_time = time.time() - start_time - resample_time
                            test_target, test_pred = evaluate_model(testing_iter, model, args)
                            
                            if (epoch > args.warmup_epochs):
                                df_target = pd.DataFrame(test_target, columns=args.columns)
                                df_pred = pd.DataFrame(test_pred, columns=args.columns)
                                
                                fair_metrics = function(df_target, df_pred, args.sensitive_attributes, fair_metrics, epoch)
                                fair_metrics.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_{args.fraction}/fair_metrics_{model_name}_{str_sys}_{x}.csv')
                                cost_metric.loc[epoch] = [epoch_time, 0, loss_value]
                                cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_{args.fraction}/cost_metrics_{model_name}_{str_sys}_{x}.csv')
                
                            if(args.ss != 5 and args.ss != 0 and resample_time > 0.0 and epoch > 0):
                                resample_metric.loc[epoch] = [resample_time]
                                resample_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_{args.fraction}/resample_{model_name}_{str_sys}_{x}.csv')
    
                        run_end_time = time.time()
                        run_time = run_end_time - run_start_time
                        cost_metric['Full_training_time'] = run_time
                        cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_{args.fraction}/cost_metrics_{model_name}_{str_sys}_{x}.csv')
                        
                        print(x)
                    elif args.ss == 0:
                        if x >= 0: 
                            for epoch in tqdm(range(1, args.epoch + 1), desc=f"Training {model_name}, run {x}"):
                                model.train()
                                start_time = time.time()
                                data_target = []
                                data_pred = []
                                for data, label, sensitive_attributes in training_iter:
                                    sentences = data.to(device, non_blocking=True)  # Asynchronous loading
                                    labels = label.to(device, non_blocking=True)
                    
                                    optimizer.zero_grad()
                                    logits = model(sentences)
                                    loss = criterion(logits, labels)
                                    loss_sum += loss.data
                                    loss_value = loss.item() 
                    
                                    if step % args.log_interval == 0:
                                        loss_sum = 0
                                        step = 0
                                    step += 1
                    
                                    loss.backward()
                                    optimizer.step()
                                    if args.scheduler : 
                                        scheduler.step()


                                    if args.scheduler : 
                                        scheduler.step() 
                                    if args.dataset_name =='ars':
                                        gender = sensitive_attributes.numpy()
                                    elif args.dataset_name =='dc' or args.dataset_name =='mobiact' or args.dataset_name == 'celeba':
                                        age = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                    elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        race = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                        age = sensitive_attributes[:, 2].numpy()
                                    elif args.dataset_name == 'fairface' : 
                                        age = sensitive_attributes[:, 0].numpy()
                                        race = sensitive_attributes[:, 1].numpy()
                                    elif args.dataset_name == 'voxceleb' :
                                        race = sensitive_attributes[:, 0].numpy()


                                    labels = labels.cpu().numpy()
                                    probabilities  = nn.functional.softmax(logits, dim=1)
                                    logits = torch.argmax(probabilities, dim=1).cpu().numpy() 
                    
                                    if args.dataset_name =='ars':
                                        batch_data = zip(gender, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[0],  sample[2]))
                                            data_target.append(sample[:2])
                                    elif args.dataset_name =='dc' or args.dataset_name =='mobiact':
                                        batch_data = zip(age, gender, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[0], sample[1], sample[3]))
                                            data_target.append(sample[:3])
                                    elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        batch_data = zip(race, gender, age, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[0], sample[1], sample[2],sample[4]))
                                            data_target.append(sample[:4])
                                    elif args.dataset_name == 'fairface' : 
                                        batch_data = zip(age, race, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[0], sample[1], sample[3]))
                                            data_target.append((sample[0], sample[1], sample[2]))
                                    elif args.dataset_name == 'celeba' : 
                                        batch_data = zip(age, gender, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[3], sample[0], sample[1]))
                                            data_target.append((sample[2], sample[0], sample[1]))
                                    elif args.dataset_name == 'voxceleb' :
                                        batch_data = zip(race, labels, logits)
                                        for sample in batch_data:
                                            data_pred.append((sample[2], sample[0]))
                                            data_target.append((sample[1], sample[0]))

                                
                                
                                epoch_time = time.time() - start_time

                                # Evaluation 

                                test_target, test_pred = evaluate_model(testing_iter, model, args)
                                
                                if (epoch > args.warmup_epochs):
                                    df_target = pd.DataFrame(test_target, columns=args.columns)
                                    df_pred = pd.DataFrame(test_pred, columns=args.columns)
                                    fair_metrics = function(df_target,df_pred, args.sensitive_attributes, fair_metrics, epoch)
                                    fair_metrics.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/fair_metrics_{model_name}_{str_sys}_{x}.csv')
                                    cost_metric.loc[epoch] = [epoch_time, 0, loss_value]
                                    cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/cost_metrics_{model_name}_{str_sys}_{x}.csv')


                        run_end_time = time.time()
                        run_time = run_end_time - run_start_time
                        cost_metric['Full_training_time'] = run_time
                        cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/cost_metrics_{model_name}_{str_sys}_{x}.csv')            
                                

if __name__ == "__main__":
    main()
