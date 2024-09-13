import torch
import gudhi
import numpy as np
import matplotlib.pyplot as plt
from gph import ripser_parallel

class SFTDLossGudhi:
    def __init__(self, dims = [1], card = 100, use_max = False, p = 1, min_barcode = 0):
        self.dims = dims
        self.card = card
        self.use_max = use_max
        self.p = p
        self.min_barcode = 0.
        
    def __call__(self, F1, G1):
        D_size = torch.Size([3] + list(F1.shape))
        D = torch.zeros(D_size, dtype = F1.dtype)
        
        if not self.use_max:
            D[0] = torch.min(F1, G1)
            D[1] = F1
            D[2] = torch.min(torch.min(F1, G1)).expand(F1.shape)
        else:
            D[0] = F1
            D[1] = torch.max(F1, G1)
            D[2] = torch.min(torch.min(F1, G1)).expand(F1.shape)
            
        cubical_complex = gudhi.CubicalComplex(vertices = D.detach().numpy())
        cubical_complex.compute_persistence(homology_coeff_field = 2, min_persistence = 0.0)
        
        v = cubical_complex.vertices_of_persistence_pairs()
        
        D_fortran = D.permute(*torch.arange(D.ndim - 1, -1, -1))
        D_flat = D_fortran.reshape(D.numel())
        
        loss = 0.
        
        self.v = v
        self.D = D
        self.cubical_complex = cubical_complex

        for dim in self.dims:
            
            if v[0] and len(v[0]) >= dim + 1:
                
                r = []
                
                for elem in v[0][dim]:
                    i, j = elem
                    if D_flat[j] - D_flat[i] > self.min_barcode:
                        r.append((D_flat[i], D_flat[j]))
            
                r_sorted = sorted(r, key = lambda x : x[1].item() - x[0].item(), reverse = True)
                part_loss = sum(map(lambda x : (x[1] - x[0]) ** self.p, r_sorted[:self.card]))
                loss += part_loss
                
        return loss

def sftd_graph(A, V0, V1, maxdim = 1):
    F = graph2matrix(A, V0)
    G = graph2matrix(A, V1)

    return sftd_matrix(F, G, maxdim = maxdim)

def graph2matrix(A, V):

    F = torch.full((A.shape[0], A.shape[1]), torch.inf)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                F[i, j] = V[i]
            elif A[i, j]:
                F[i, j] = max(V[i], V[j])
    return F

def sftd_matrix(F, G, maxdim = 1):
    n = F.shape[0]
    col_inf = torch.full((n, 1), torch.inf)
    F_nodes = torch.diag(F).unsqueeze(1)

    F_low_inf = F.clone()
    i_list, j_list = np.tril_indices(n, -1)
    F_low_inf[i_list, j_list] = torch.inf

    fake_vertex = torch.min(torch.min(F), torch.min(G))

    #
    #  min(F, G)   F'T    inf
    #     F'       F     F_diag
    #    inf     F_diag   min(F)
    #
    top = torch.cat((torch.min(F, G), F_low_inf.T, col_inf), 1)
    middle = torch.cat((F_low_inf, F, F_nodes), 1)
    bottom = torch.cat((col_inf.T, F_nodes.T, torch.full((1, 1), fake_vertex)), 1)

    D = torch.cat((top, middle, bottom))

    r = ripser_parallel(D.numpy(), metric = 'precomputed', maxdim = maxdim, n_threads = -1)

    return r

def plot_barcodes(arr, color_list = ['deepskyblue', 'limegreen', 'darkkhaki'], dark_color_list = None, title = '', hom = None):

    if dark_color_list is None:
        dark_color_list = color_list
        #dark_color_list = ['b', 'g', 'orange']

    sh = len(arr)
    step = 0
    if (len(color_list) < sh):
        color_list *= sh

    for i in range(sh):

        if not (hom is None):
            if i not in hom:
                continue

        barc = arr[i].copy()
        arrayForSort = np.subtract(barc[:,1],barc[:,0])

        bsorted = np.sort(arrayForSort)
        nbarc = bsorted.shape[0]

        if nbarc:
            plt.plot(barc[0], np.ones(2)*step, color = color_list[i], label = 'H{}'.format(i))
            for b in barc:
                plt.plot(b, np.ones(2)*step, color = color_list[i])
                step += 1

    plt.xlabel('$\epsilon$ (time)')
    plt.ylabel('segment')
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.rcParams["figure.figsize"] = [6, 4]
