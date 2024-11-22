from src.data import *
from src.model import *
import logging
import yaml
import argparse
import time
import random
import torch.multiprocessing as mp
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class SimplePredictor(nn.Module):
    def __init__(self, in_dim, n_class, args, pd_models=None, bias=False):
        super().__init__()
        self.args = args
        self.use_pd = args.use_pd
        self.use_f = args.use_f
        self.pd_models = pd_models
        self.n_pd = 0 if pd_models is None else len(pd_models)
        self.fc1 = nn.LazyLinear(n_class, bias=True)
        self.bn1 = nn.BatchNorm1d(n_class, affine=False)
        self.activation = nn.Sigmoid()

    def forward(self, PDs, PD_lens, PD_counts, X, mult_in_equiv=False):
        pd_rep = None
        if self.use_pd:   
            for i in range(self.n_pd):
                PD = PDs[i].to(X.device).float()
                PD_len = torch.tensor(PD_lens[i], dtype=torch.float32).to(X.device)
                if self.args.use_mult:
                    PD_count = PD_counts[i].to(X.device).float()
                else:
                    PD_count = None
                pd_output = self.pd_models[i](PD, PD_len, PD_count, mult_in_equiv)
                if pd_rep is None:
                    pd_rep = pd_output
                else:
                    pd_rep = torch.cat([pd_rep, pd_output], dim=1)
            if pd_rep is not None:
                if self.use_f:
                    X = torch.cat([pd_rep, X], dim=1)
                else:
                    X = pd_rep                
        elif self.use_f:
            X = X
        else:
            X = torch.rand_like(X).to(X.device)
        logging.debug(f"X.shape={X.shape}")
        X = self.fc1(X)
        X = self.bn1(X)
        return self.activation(X)

def evaluate_accuracy(model, dataloaders, device, args):
    model.eval()
    accuracies = {}

    with torch.no_grad():
        for phase in ['train', 'test']: 
            correct = 0
            total = 0
            for PDs, PD_counts, PD_lens, X, Y in dataloaders[phase]:
                if X.size(0) == 1:
                    continue
                PDs, PD_counts, PD_lens, X, Y = PDs, PD_counts, PD_lens, X.to(device).float(), Y.to(device).float()
                one_hot_labels = Y.to(device)
                outputs = model(PDs, PD_lens, PD_counts, X, mult_in_equiv=args.mult_in_equiv)
                # Convert predictions and one-hot labels to class indices
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(one_hot_labels, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracies[phase] = 100 * correct / total
    return accuracies

def train(param, queue):
    ds, data_idx, args = param
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    dl_indices = [(train_indices, test_indices) for train_indices, test_indices in kf.split(range(len(ds)))]
    train_indices, test_indices = dl_indices[data_idx]
    train_dataset = Subset(ds, train_indices)
    test_dataset = Subset(ds, test_indices)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    
    device = args.device
    n_class = args.n_class
    use_pd = args.use_pd
    x_dim = ds.x_dim

    if not use_pd:
        model = SimplePredictor(x_dim, n_class, args).to(device)
    else:
        pd_models = []
        total_pd_out_dim = 0
        for i in range(ds.n_pds):
            if args.pd_model.lower() == "transformer":
                pd_models.append(Transformer(args.num_out, args.num_hiddens, args.ffn_num_hiddens, 
                                             args.num_heads, args.num_inds, args.dropout, args.use_bias, 
                                             args.pre_ln, args.equiv, args.num_equiv, 
                                             args.num_queries, args.pooling).to(device))
                if args.equiv.lower() in ['sab', 'isab']:
                    total_pd_out_dim += args.num_out * args.num_queries
                else:
                    total_pd_out_dim += args.num_out
            else:
                raise ValueError(f"pd_model={args.pd_model} not exists.") 
        in_dim = x_dim + total_pd_out_dim if args.use_f else total_pd_out_dim
        logging.debug(f"x_dim={x_dim}")
        if args.pd_model.lower() == "transformer" and args.pooling in ['sum']:
            in_dim += 1
        logging.debug(f"in_dim={in_dim}")
        model = SimplePredictor(in_dim, n_class, args, pd_models=pd_models).to(device)
    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta, 0.999))

    _dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    batches_per_epoch = len(train_loader)
    num_training_steps = args.n_epochs * batches_per_epoch
    num_warmup_steps = args.warmup_epochs * batches_per_epoch
    num_cycles = args.num_cycles if hasattr(args, 'num_cycles') else 1
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                                   num_warmup_steps=num_warmup_steps, 
                                                                   num_training_steps=num_training_steps,
                                                                   num_cycles=num_cycles)
    start_time = time.time()
    epoch_test_accs = []
    epoch_train_accs = []
    epoch_times = []
    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss_train = 0.0
        for PDs, PD_counts, PD_lens, X, Y in train_loader:
            if X.size(0) == 1:
                continue
            X, Y = X.to(device).float(), Y.to(device).float()
            optimizer.zero_grad()
            Y_ = model(PDs, PD_lens, PD_counts, X, mult_in_equiv=args.mult_in_equiv)
            loss = criterion(Y_, Y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss_train += loss.item() * X.size(0)
        acc = evaluate_accuracy(model, _dataloaders, device, args)
        epoch_test_accs.append(acc['test'])
        epoch_train_accs.append(acc['train'])
        epoch_times.append(time.time() - epoch_start_time)
        logging.info(f"Epoch {epoch+1}/{args.n_epochs}, Idx:{data_idx}, Train Accuracy: {acc['train']:.2f}%, Test Accuracy: {acc['test']:.2f}%")
    print(f"Epoch taining time of idx-{data_idx}: {np.mean(epoch_times):.2f} ± {np.std(epoch_times):.2f} s")
    execution_time = time.time() - start_time
    print(f"Accuracy of idx-{data_idx}: {np.max(epoch_test_accs):.2f}%")
    queue.put((np.max(epoch_test_accs), execution_time))
    return np.max(epoch_test_accs)

def paralleled_main(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    args.device = device
    n_classes = args.n_classes if hasattr(args, 'n_classes') else None
    diags_dict, X, Y = load_data(args.dataset, n_classes=n_classes)
    ds = MSDataset(diags_dict, X, Y, args)
    ds.print_stats()
    args.n_class = ds.n_class
    params = [[ds, data_idx, args] for data_idx in range(args.n_splits)]
    processes = []
    queue = mp.Queue()
    print(args)
    for param in params:
        p = mp.Process(target=train, args=(param, queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        if p.is_alive():
            print(f"Process {p.pid} is still running after timeout.")
    q_ = [queue.get() for _ in processes]
    accs = [t[0] for t in q_]
    times_ = [t[1] for t in q_]
    accs_mean = np.mean(accs)
    accs_std = np.std(accs)
    print(f">>>>>>>>>> Final Test Acc: {accs_mean:.2f} ± {accs_std:.2f}")
    times_mean = np.mean(times_)
    times_std = np.std(times_)
    print()
    return accs_mean

def final_results(args):
    accs = []
    for i in range(args.runs):
        print(f">>>>>>>>>> Start {i+1}-th RUNS")
        acc = paralleled_main(args)
        accs.append(acc)
        args.random_state = random.randint(1, 10000)
    accs_mean = np.mean(accs)
    accs_std = np.std(accs)

    res_ = f"Final Test Acc with {args.runs} RUNS: {accs_mean:.2f} ± {accs_std:.2f}"
    box_width = len(res_) + 4
    print("#" * box_width)
    print("# " + res_ + " #")
    print("#" * box_width)
    print()
    return f"{accs_mean:.2f} ± {accs_std:.2f}" 

if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=Warning)
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Load params file')
    parser.add_argument('config_file', type=str, help='Path to the YAML config file')
    cmd_args = parser.parse_args()

    rand_seed = 42
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    
    with open(cmd_args.config_file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    args = argparse.Namespace(**params)
    final_results(args)
