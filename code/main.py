import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

from dataselection.utils.data.datasets.SL.builder import load_dataset
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

from bias_metrics import get_fair_metrics_ars, get_fair_metrics_dc, get_fair_metrics_mobiact, get_fair_metrics_adult


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

def num_features(args):
    if args.dataset_name == 'ars':
        num_features = 9
    elif args.dataset_name == 'adult':
        num_features = 14
    elif args.dataset_name == 'dc':
        num_features = 11
    elif args.dataset_name == 'mobiact':
        num_features = 14
    elif args.dataset_name == 'kdd':
        num_features = 14
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
    else:
        raise ValueError("Unknown dataset name")

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    args.device = device
    num_f = num_features(args)
    function = fair_metrics_func(args)

    for model_name in args.models:
        print("*****************")
        print(model_name)
        print("*****************")
        for ratio in args.ratios:
            args.fraction = ratio
            if(ratio == 0.05):
                values_final = args.values
            else:
                values_final = args.values
            for i in values_final:
                args.ss = i
                print(i)
                for x in range(args.runs): 
                    start = time.time()
                
                    # Datasets
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
            
                    if model_name == 'MLP' :
                        model = MLPModel(num_f, args.label_num).to(device)
                    elif model_name == 'SVM' :
                        model = SVMNet(num_f, args.label_num).to(device)
                    else :
                        model = LogisticRegNet(num_f, args.label_num).to(device)
                
                
                    criterion = nn.CrossEntropyLoss()
                    criterion_nored = nn.CrossEntropyLoss(reduction='none')
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                
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
                
                    run_start_time = time.time()
                    
                    fair_metrics = pd.DataFrame(columns=args.cols)
                    fair_metrics.index.name = "epoch"
                
                    cost_metric = pd.DataFrame(columns=['accuracy', 'Model_training_time', 'Full_training_time'])
                    cost_metric.index.name = "epoch"
                
                    resample_metric = pd.DataFrame(columns=['Data_selection_execution_time'])
                    resample_metric.index.name = "epoch"
                
                    if args.ss > 0:
                        for epoch in range(1, args.epoch+1):
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
                
                                loss_sum += subtrn_loss
                                if step % args.log_interval == 0:
                                    loss_sum = 0
                                    step = 0
                                step += 1
                
                                optimizer.step()
                                _, predicted = outputs.max(1)
                                subtrn_total += targets.size(0)
                                subtrn_correct += predicted.eq(targets).sum().item()
                                if args.dataset_name =='ars':
                                    gender = sensitive_attributes.numpy()
                                elif args.dataset_name =='dc' or args.dataset_name =='mobiact':
                                        age = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        race = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                        age = sensitive_attributes[:, 2].numpy()
                                        
                                targets = targets.cpu().numpy()
                                probabilities  = nn.functional.softmax(outputs, dim=1)
                
                                labels = torch.argmax(probabilities, dim=1).cpu().numpy() 
                
                                if args.dataset_name =='ars':
                                    batch_data = zip(gender, targets, labels)
                                    for sample in batch_data:
                                        data_pred.append((sample[0], sample[2]))
                                        data_target.append(sample[:2])
                                elif args.dataset_name =='dc' or args.dataset_name =='mobiact':
                                        batch_data = zip(age, gender, targets, labels)
                                        for sample in batch_data:
                                            data_pred.append((sample[0], sample[1], sample[3]))
                                            data_target.append(sample[:3])
                                elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        batch_data = zip(race, gender, age, targets, labels)
                                        for sample in batch_data:
                                            data_pred.append((sample[0], sample[1], sample[2],sample[4]))
                                            data_target.append(sample[:4])
                
                            epoch_time = time.time() - start_time - resample_time
                            acc = evaluation(testing_iter, model, args)
                            if (epoch > args.warmup_epochs):
                                df_target = pd.DataFrame(data_target, columns=args.columns)
                                df_pred = pd.DataFrame(data_pred, columns=args.columns)
                                
                                fair_metrics = function(df_target, df_pred, args.sensitive_attributes, fair_metrics, epoch)
                                fair_metrics.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_{args.fraction}/fair_metrics_{model_name}_{str_sys}_{x}.csv')
                                cost_metric.loc[epoch] = [acc, epoch_time, 0]
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
                            for epoch in range(1, args.epoch + 1):
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
                    
                                    if step % args.log_interval == 0:
                                        loss_sum = 0
                                        step = 0
                                    step += 1
                    
                                    loss.backward()
                                    optimizer.step()
                                    if args.dataset_name =='ars':
                                        gender = sensitive_attributes.numpy()
                                    elif args.dataset_name =='dc' or args.dataset_name =='mobiact':
                                        age = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                    elif args.dataset_name =='adult' or args.dataset_name =='kdd':
                                        race = sensitive_attributes[:, 0].numpy()
                                        gender = sensitive_attributes[:, 1].numpy()
                                        age = sensitive_attributes[:, 2].numpy()


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

                                    
                    
                                epoch_time = time.time() - start_time
                                acc = evaluation(testing_iter, model, args)
                                if (epoch > args.warmup_epochs):
                                    df_target = pd.DataFrame(data_target, columns=args.columns)
                                    df_pred = pd.DataFrame(data_pred, columns=args.columns)
                                    fair_metrics = function(df_target,df_pred, args.sensitive_attributes, fair_metrics, epoch)
                                    fair_metrics.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/fair_metrics_{model_name}_{str_sys}_{x}.csv')
                                    cost_metric.loc[epoch] = [acc, epoch_time, 0]
                                    cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/cost_metrics_{model_name}_{str_sys}_{x}.csv')


                        run_end_time = time.time()
                        run_time = run_end_time - run_start_time
                        cost_metric['Full_training_time'] = run_time
                        cost_metric.to_csv(f'{args.result_path}/{str_sys}/{args.dataset_name}_1/cost_metrics_{model_name}_{str_sys}_{x}.csv')            
                                

if __name__ == "__main__":
    main()
