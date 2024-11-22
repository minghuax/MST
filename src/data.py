import os
import copy
import h5py
import pickle
import itertools
import numpy as np
import pandas as pd
import gudhi as gd
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.linalg import eigh
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import gudhi.representations as tda
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import torch_geometric.datasets as datasets
from torch_geometric.utils import to_dense_adj
from scipy.io import savemat
import logging
import time

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds to run.")
        return result
    return wrapper

### BEGIN https://github.com/MathieuCarriere/perslay ###

def get_parameters(dataset):
    if dataset == "MUTAG" or dataset == "PROTEINS":
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "REDDIT5K" or dataset == "REDDIT12K":
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_1.0-hks", "Rel1_1.0-hks", "Ext0_1.0-hks", "Ext1_1.0-hks"]}
    elif dataset == "COX2" or dataset == "DHFR" or dataset == "NCI1" or dataset == "NCI109" or dataset == "IMDB-BINARY" or dataset == "IMDB-MULTI" or dataset == "COLLAB":
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_0.1-hks", "Rel1_0.1-hks", "Ext0_0.1-hks", "Ext1_0.1-hks",
                                             "Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "ORBIT5K" or dataset == "ORBIT100K":
        dataset_parameters = {"data_type": "orbit", "filt_names": ["Alpha0", "Alpha1"]}
    elif dataset == "SYNTHETIC":
        dataset_parameters = {"data_type": "synthetic"}
    return dataset_parameters

def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)

def generate_orbit(num_pts_per_orbit, param):
    X = np.zeros([num_pts_per_orbit, 2])
    xcur, ycur = np.random.rand(), np.random.rand()
    for idx in range(num_pts_per_orbit):
        xcur = (xcur + param * ycur * (1. - ycur)) % 1
        ycur = (ycur + param * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]
    return X

@timing
def generate_diagrams_and_features(dataset, path="data/"):
    dataset_parameters = get_parameters(dataset)
    dataset_type = dataset_parameters["data_type"]

    # if "REDDIT" in dataset:
    #     print("Unfortunately, REDDIT data are not available yet for memory issues.\n")
    #     print("Moreover, the link we used to download the data,")
    #     print("http://www.mit.edu/~pinary/kdd/datasets.tar.gz")
    #     print("is down at the commit time (May 23rd).")
    #     print("We will update this repository when we figure out a workaround.")
    #     return

    path_dataset = path + dataset + "/"
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5", "w")
    list_filtrations = dataset_parameters["filt_names"]
    [diag_file.create_group(str(filtration)) for filtration in dataset_parameters["filt_names"]]

    if dataset_type == "graph":

        list_hks_times = np.unique([filtration.split("_")[1] for filtration in list_filtrations])

        # preprocessing
        pad_size = 1
        for graph_name in os.listdir(path_dataset + "mat/"):
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            pad_size = np.max((A.shape[0], pad_size))

        feature_names = ["eval" + str(i) for i in range(pad_size)] + [name + "-percent" + str(i) for name, i in
                                                                      itertools.product(
                                                                          [f for f in list_hks_times if "hks" in f],
                                                                          10 * np.arange(11))]
        features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))), columns=["label"] + feature_names)

        for idx, graph_name in enumerate((os.listdir(path_dataset + "mat/"))):

            name = graph_name.split("_")
            gid = int(name[name.index("gid") + 1]) - 1
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            num_vertices = A.shape[0]
            label = int(name[name.index("lb") + 1])

            L = csgraph.laplacian(A, normed=True)
            egvals, egvectors = eigh(L)
            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for fhks in list_hks_times:
                hks_time = float(fhks.split("-")[0])
                filtration_val = hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val)
                diag_file["Ord0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmOrd0)
                diag_file["Ext0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt0)
                diag_file["Rel1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmRel1)
                diag_file["Ext1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt1)
                graph_features.append(
                    np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time), 10 * np.arange(11)))
            features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)
        features["label"] = features["label"].astype(int)

    elif dataset_type == "orbit":

        labs = []
        count = 0
        num_diag_per_param = 1000 if "5K" in dataset else 20000
        for lab, r in enumerate([2.5, 3.5, 4.0, 4.1, 4.3]):
            print("Generating", num_diag_per_param, "orbits and diagrams for r = ", r, "...")
            for dg in range(num_diag_per_param):
                X = generate_orbit(num_pts_per_orbit=1000, param=r)
                alpha_complex = gd.AlphaComplex(points=X)
                st = alpha_complex.create_simplex_tree(max_alpha_square=1e50)
                st.persistence()
                diag_file["Alpha0"].create_dataset(name=str(count),
                                                   data=np.array(st.persistence_intervals_in_dimension(0)))
                diag_file["Alpha1"].create_dataset(name=str(count),
                                                   data=np.array(st.persistence_intervals_in_dimension(1)))
                orbit_label = {"label": lab, "pcid": count}
                labs.append(orbit_label)
                count += 1
        labels = pd.DataFrame(labs)
        labels.set_index("pcid")
        features = labels[["label"]]

    elif dataset_type == "synthetic":
        pass
    features.to_csv(path_dataset + dataset + ".csv")

    return diag_file.close()


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmOrd0 if p[0] == 0]) if len(
        dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmRel1 if p[0] == 1]) if len(
        dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt0 if p[0] == 0]) if len(
        dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt1 if p[0] == 1]) if len(
        dgmExt1) else np.empty([0, 2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1


def load_data(dataset, path ="data/", filtrations=[], verbose=False, n_classes=None):
    path_dataset = path + dataset + "/"
    if dataset.lower() == "synthetic":
        np.random.seed(42)
        n_samples = 1000
        max_label =  n_classes - 1 # n_classes - 2,3,5,11
        print(f"Synthetic: n_classes = {max_label + 1}")
        data_samples = []
        labels = []
        for _ in range(n_samples):
            label = np.random.randint(0, max_label + 1)
            sample = None
            for i_ in range(max_label+1):
                if i_ == label:
                    sample_ = np.ones((np.random.randint(32, 64), 2)) * i_ / max_label
                else:
                    sample_ = np.ones((np.random.randint(0, 32), 2)) * i_ / max_label
                sample = sample_ if sample is None else np.concatenate((sample, sample_))
            data_samples.append(sample)
            labels.append(label)
        diags_dict = {"Ext": data_samples}
        F = np.zeros((n_samples, 1))
        L = np.eye(max_label+1)[labels]
        return diags_dict, F, L

    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys()) if len(filtrations) == 0 else filtrations

    diags_dict = dict()
    if len(filts) == 0:
        filts = diagfile.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diagfile[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diagfile[filtration][str(diag)]))
        diags_dict[filtration] = list_dgm

    # Extract features and encode labels with integers
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    F = np.array(feat)[:, 1:]  # 1: removes the labels
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diags_dict, F, L

### END https://github.com/MathieuCarriere/perslay ###

def replace_with_representative(pd, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(pd)
    representative_points = kmeans.cluster_centers_
    replaced_points = [representative_points[label] for label in kmeans.labels_]
    return np.array(replaced_points)

@timing
def cluster_by_kmeans(diags_dict, n_clusters=100):
    def cluster_(pd):
        unique_pts_, _ = count_pd(pd)
        if len(unique_pts_) <= n_clusters:
            return pd
        pd_ = replace_with_representative(pd, n_clusters=n_clusters)
        assert len(pd) == len(pd_)
        return pd_
    clustered_dict = {}
    for k, pds in diags_dict.items():
        clustered_dict[k] = list(map(cluster_, pds))
    return clustered_dict

@timing
def cluster_by_dbscan(diags_dict, eps=1e-2, min_samples=1):
    from pathos.multiprocessing import ProcessPool
    _eps = eps
    clustered_dict = {}
    for k, pds in diags_dict.items():
        def cluster_(pd):
            if len(pd) == 0:
                return pd
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(pd)
            labels = db.labels_
            return np.array([pd[labels == i].mean(axis=0) if i != -1 else point for point, i in zip(pd, labels)])
        
        def parallel_map(pds):
            with ProcessPool(16) as pool:
                return pool.map(cluster_, pds)

        clustered_dict[k] = parallel_map(pds)
    return clustered_dict

def append_counts(PD):
    PD_sorted = np.array(sorted(PD.tolist(), key=lambda x: (x[0], x[1])))
    unique_points, counts = np.unique(PD_sorted, axis=0, return_counts=True)
    result = np.column_stack((unique_points, counts))
    return result

def count_pd(PD):
    if PD.shape[1] == 3:
        PD_sorted = np.array(sorted(PD.tolist(), key=lambda x: (x[2], x[0], x[1])))
    else:
        PD_sorted = np.array(sorted(PD.tolist(), key=lambda x: (x[0], x[1])))
    unique_points, counts = np.unique(PD_sorted, axis=0, return_counts=True)
    counts = counts[:, np.newaxis]
    return unique_points, counts

def pad_sets(batched_data, expected_set_size=2, expected_set_len=None):
    batched_data = [torch.tensor(s, dtype=torch.float32) for s in batched_data]
    # print(expected_set_len)
    if expected_set_len is None:
        expected_set_len = max([s.size(0) for s in batched_data])
    expected_set_len = max([1, expected_set_len])
    padded_data = []
    for s in batched_data:
        pad_len = expected_set_len - s.size(0)
        if s.numel() > 0:
            pad_set_size = expected_set_size - s.size(1)
            padded_data.append(nn.functional.pad(s, (0, pad_set_size, 0, pad_len)))
        else:
            padded_data.append(torch.zeros(expected_set_len, expected_set_size, dtype=torch.float32))
    return torch.stack(padded_data)


class MSDataset(Dataset):
    def __init__(self, pd_dict, X, Y, args):
        self.args=args
        self.pd_mode = args.pd_mode
        self.pd_dict = pd_dict
        self.clustered_pd_dict = None
        self.X = X
        self.Y = Y
        
        self.keys = []
        _pd_names = []
        if args.rel_pd:
            _pd_names.append('Rel')
        if args.ext_pd:
            _pd_names.append('Ext')
        if args.ord_pd:
            _pd_names.append('Ord')
        for k in list(pd_dict.keys()):
            if k[:3] in _pd_names:
                self.keys.append(k)
        print(f"pd_keys = {self.keys}")

        self.original_PDs = [self.pd_dict[k] for k in self.keys]
        self.n_pds = len(self.keys)
        self.PDs = None
        self.PD_lens = [[] for _ in self.keys]
        self.PD_counts = [[] for _ in self.keys]
        self.ms_PDs = [[] for _ in self.keys]
        self.ms_PD_lens = [[] for _ in self.keys]
        self.ms_PD_counts = [[] for _ in self.keys]
        self.x_dim = X.shape[1]
        self.set_max_lens = []
        self.n_class = Y.shape[1]
        for k in self.keys:
            pds = self.pd_dict[k]
            pd_lens = [len(pd) for pd in pds]
            max_len_ = max(pd_lens)
            self.set_max_lens.append(max_len_)
        print(f"set_max_lens = {self.set_max_lens}")
        self.pd_dict = self.tda_preprocess(pd_dict)
        self.preprocess()

    def get_ms_max_lens(self, pd_dict):
        max_lens = []
        for k in self.keys:
            pds = pd_dict[k]
            pd_lens = [len(pts_) for pts_, _ in [count_pd(pd) for pd in pds]]
            max_len_ = max(pd_lens)
            max_lens.append(max_len_)
        print(f"ms_max_lens = {max_lens}")
        return max_lens
    
    def tda_preprocess(self, diags_dict, thresh=500):
        if self.args.use_cluster:
            thresh = 1000
        tmp = Pipeline([
            ("Selector",      tda.DiagramSelector(use=True, point_type="finite")),
            ("ProminentPts",  tda.ProminentPoints(use=True, num_pts=thresh)),
            ("Scaler",        tda.DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())])),
            ])
        
        prm = {filt: {"ProminentPts__num_pts": min(thresh, max([len(dgm) for dgm in diags_dict[filt]]))} 
            for filt in diags_dict.keys() if max([len(dgm) for dgm in diags_dict[filt]]) > 0}

        diags_dict_ = {} 
        for dt in prm.keys():
            param = prm[dt]
            tmp.set_params(**param)
            diags_dict_[dt] = tmp.fit_transform(diags_dict[dt])
        return diags_dict_

    def preprocess(self):
        path =f"data/{self.args.dataset}/"
        pd_dict_ = self.pd_dict
        if self.args.use_cluster:
            if self.args.cluster_method.lower() == "kmeans":
                file_ = path + f"KMeans-{self.args.kmeans_n_clusters}.pkl"
                if os.path.exists(file_):
                    with open(file_, 'rb') as f:
                        self.clustered_pd_dict = pickle.load(f)
                else:
                    self.clustered_pd_dict = cluster_by_kmeans(self.pd_dict, n_clusters=self.args.kmeans_n_clusters)
                    with open(file_, 'wb') as f:
                        pickle.dump(self.clustered_pd_dict, f)
            elif self.args.cluster_method.lower() == "dbscan":
                file_ = path + f"DBSCAN-{self.args.dbscan_eps}.pkl"
                if os.path.exists(file_):
                    with open(file_, 'rb') as f:
                        self.clustered_pd_dict = pickle.load(f)
                else:
                    self.clustered_pd_dict = cluster_by_dbscan(self.pd_dict, eps=self.args.dbscan_eps)
                    with open(file_, 'wb') as f:
                        pickle.dump(self.clustered_pd_dict, f)
            pd_dict_ = self.clustered_pd_dict
        PDs_ = [pd_dict_[k] for k in self.keys]
        self.PDs = copy.deepcopy(PDs_)
        for p_ in range(self.n_pds):
            for i_ in range(len(self)):
                self.PD_lens[p_].append(len(self.PDs[p_][i_]))
        for i in range(self.n_pds):
            self.PDs[i] = pad_sets(self.PDs[i], 2, self.set_max_lens[i])
            self.PD_counts[i] = torch.zeros((self.PDs[i].shape[0], self.PDs[i].shape[1], 1))
        ms_max_lens_ = self.get_ms_max_lens(pd_dict_)
        for p_ in range(self.n_pds):
            PD = PDs_[p_]
            PD_, count_ = [list(x) for x in zip(*[count_pd(pd) for pd in PD])]
            for i_ in range(len(self.Y)):
                self.ms_PDs[p_].append(PD_[i_])
                self.ms_PD_counts[p_].append(count_[i_])
                self.ms_PD_lens[p_].append(len(PD_[i_]))
        for i in range(self.n_pds):
            self.ms_PDs[i] = pad_sets(self.ms_PDs[i], 2, ms_max_lens_[i])
            self.ms_PD_counts[i] = pad_sets(self.ms_PD_counts[i], 1, ms_max_lens_[i])
            self.ms_PD_counts[i][self.ms_PD_counts[i]>0] -= 1

    def print_stats(self):
        label_counts = self.Y.sum(axis=0)
        print(f"Label counts: {label_counts}")
        __total_element = 0
        for k in self.keys:
            pds = self.pd_dict[k]
            pd_lens_ = [len(pd) for pd in pds]
            __total_element += sum(pd_lens_)
            print(f"""{k}-pd-statistics (original): {sum(pd_lens_)} | {np.mean(pd_lens_):.2f}±{np.std(pd_lens_):.2f}""")
        print(f"Total element count in dataset: {__total_element}")

        __total_element = 0
        for k in self.keys:
            pds = self.pd_dict[k]
            pd_lens = [len(pts_) for pts_, _ in [count_pd(pd) for pd in pds]]
            __total_element += sum(pd_lens)
            print(f"""{k}-pd-statistics (MS w/o cluster): {sum(pd_lens)} | {np.mean(pd_lens):.2f}±{np.std(pd_lens):.2f}""")
        print(f"Total element count in dataset: {__total_element}")
        
        __total_element = 0
        if self.clustered_pd_dict is not None:
            for k in self.keys:
                pds = self.clustered_pd_dict[k]
                pd_lens = [len(pts_) for pts_, _ in [count_pd(pd) for pd in pds]]
                __total_element += sum(pd_lens)
                print(f"""{k}-pd-statistics (MS w/ cluster): {sum(pd_lens)} | {np.mean(pd_lens):.2f}±{np.std(pd_lens):.2f}""")
            print(f"Total element count in dataset: {__total_element}")
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if self.pd_mode.lower() in ['set', 's']:
            return [p[idx] for p in self.PDs], [m[idx] for m in self.PD_counts], [l[idx] for l in self.PD_lens], self.X[idx], self.Y[idx]
        elif self.pd_mode.lower() in ['multiset','ms']:
            return [p[idx] for p in self.ms_PDs], [m[idx] for m in self.ms_PD_counts], [l[idx] for l in self.ms_PD_lens], self.X[idx], self.Y[idx]

if __name__ == '__main__':
    generate_diagrams_and_features("MUTAG")
