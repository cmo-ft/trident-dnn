
import glob
import torch_geometric
import pandas as pd
import torch
import time

t_start = time.time()
def process_gnn_data():
    """
     Modify this region to control input and output
    """
    target_pdg = 11 # 11 for e, 13 for u, 15 for tau
    in_dir = f'data/'
    out_dir = 'data/'


    in_data_list = glob.glob(f'{in_dir}/xyz_*.csv')

    for ifile in range(0, len(in_data_list)):
        print(f"processing {ifile}...", flush=True)
        in_data = pd.read_csv(f"{in_dir}/xyz_{ifile}.csv").set_index('id')
        in_prim = pd.read_csv(f"{in_dir}/primary_{ifile}.csv").set_index('id')
        in_neu = pd.read_csv(f"{in_dir}/neutrino_{ifile}.csv").set_index('id')

        in_prim = in_prim[(abs(in_prim.PdgId)==target_pdg)]
        nHits = in_data['nhits'].groupby('id').sum()
        nHitDoms = in_data.index.value_counts().sort_index()

        mask =  (nHits>10) 
        in_data = in_data.loc[mask]
        indexes = in_data.index.unique()

        out_data_list = []
        # out_neutrino_list = []
        for idx in indexes:
            data = in_data.loc[[idx]]
            try:
                primary = in_prim.loc[[idx]]
                neutrino = in_neu.loc[[idx]]
            except:
                print(f'no primay in {idx}. skip.')
                continue
            
            data = data.sort_values('t1st')
            tmin = data.iloc[0]['t1st']
            fstNode = data.iloc[[0]]
            main_muon = primary[primary.e0==primary.e0.max()].iloc[[0]]
            
            data['t1st'] -= tmin
            main_muon['t0'] -= tmin

            features = torch.from_numpy(data[['nhits','t1st']].to_numpy()).type(torch.float)
            position = torch.from_numpy(data[['x0', 'y0', 'z0']].to_numpy()-fstNode[['x0', 'y0', 'z0']].to_numpy()).type(torch.float)
            vertex =  torch.from_numpy(main_muon[['x0', 'y0', 'z0']].to_numpy()-fstNode[['x0', 'y0', 'z0']].to_numpy()).type(torch.float).view(1,3)
            energy = torch.tensor(main_muon['e0'].iloc[0]).type(torch.float).reshape(1,1)
            t0 = torch.tensor([main_muon.t0.iloc[0] - tmin]).view(1,1)
            fstNode = torch.from_numpy(fstNode[['x0', 'y0', 'z0']].to_numpy()).type(torch.float).view(1,3)
            label =  torch.from_numpy(main_muon[['px', 'py', 'pz']].to_numpy()).type(torch.float).view(1,3)
            inject = torch.from_numpy(data[['xInject','yInject','zInject']].to_numpy()).type(torch.float)
            nu_P = torch.from_numpy(neutrino[['px', 'py', 'pz']].to_numpy()).type(torch.float).view(1,3)

            out_data_list.append(torch_geometric.data.Data(
                    x=features, pos=position, y=label, vertex=vertex, energy=energy, t0=t0, inject=inject, fstNode=fstNode,
                    nu_P = nu_P
                    ))

        data, slices, _ = torch_geometric.data.collate.collate(out_data_list[0].__class__,data_list=out_data_list,increment=False,add_batch=False)
        torch.save((data, slices), f"{out_dir}/xyz_{ifile}.pt")
        # torch.save(torch_geometric.data.Batch.from_data_list(out_data_list), f"{out_dir}/batch_{ifile}.pt")

process_gnn_data()
print(time.time()-t_start)
