''' Read in csv data file of 1M calibration events
'''

#########################
### imports ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot as ur
import os
########################
### MPL Settings ###

params = {
    'font.size': 14
}
plt.rcParams.update(params)

#########################

def make_all_plots(df, output_path):

    # for making plots
    with PdfPages(output_path) as pdf:
        # loop over field names
        for idx, key in enumerate(df):

            print(f"Accessing variable with name = {key} ({idx+1} / {df.shape[1]})")
            data = df[key].to_numpy()

            ##############
            # make plots #
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[5.5*3, 5.])

            # linear scale plot
            bins = 50
            _, bin_edges, _ = ax1.hist(data, bins=bins, histtype="step", density=False, label="linear")
            ax1.set_xlabel(key)
            ax1.set_ylabel("Frequency")

            if data.min() <= 0:
                data_upshifted = data + np.abs(data.min()) + 1e-30
                label = "log (upshifted)"
            else:
                data_upshifted = data
                label = "log"

            # log scale for y-axis
            ax2.hist(data_upshifted, bins=bin_edges, histtype="step", density=False, label=label)
            ax2.set_yscale("log")
            ax2.set_xlabel(key)
            ax2.legend(frameon=False, loc="upper right")
            ax2.set_ylabel("Frequency")

            # log scale for both axis
            bins_log = np.logspace(np.log10(data_upshifted.min()), np.log10(data_upshifted.max()), len(bin_edges)-1)
            ax3.hist(data_upshifted, bins=bins_log, histtype="step", density=False, label=label)
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            ax3.set_xlabel(key)
            ax3.legend(frameon=False, loc="upper right")
            ax3.set_ylabel("Frequency")
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saving all figures into {output_path}")

def normalize(x, pre_derived_scale=None):
    if pre_derived_scale:
        mean, std = pre_derived_scale[-2], pre_derived_scale[-1]
    else:
        mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    return out, mean, std

def apply_log(x, pre_derived_scale=None):
    if pre_derived_scale:
        minimum = pre_derived_scale[-4]
        epsilon = pre_derived_scale[-3]
    else:
        epsilon = 1e-10
        minimum = x.min()
    if minimum <= 0:
        x = x - minimum + epsilon
    else:
        minimum = 0
        epsilon = 0
    return np.log(x), minimum, epsilon

def apply_cuts(df):
    df = df[df["cluster_ENG_CALIB_TOT"]>0.3]
    df = df[df["clusterE"]>0.]
    df = df[df["cluster_CENTER_LAMBDA"]>0.]
    df = df[df["cluster_FIRST_ENG_DENS"]>0.]
    df = df[df["cluster_SECOND_TIME"]>0.]
    df = df[df["cluster_SIGNIFICANCE"]>0.]
    df = calculate_response(df)
    return df

def calculate_response(df):
    resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
    df["r_e_calculated"] = resp
    df = df[df["r_e_calculated"]>0.1]
    return df

def apply_scale(df, field_name, mode, pre_derived_scale=None):
    if pre_derived_scale:
        old_scale = pre_derived_scale
    else:
        old_scale = None
    if mode=='lognormalise':
        x, minimum, epsilon = apply_log(df[field_name], old_scale)
        x, mean, std = normalize(x, old_scale)
        scale = ("SaveLog / Normalize", minimum, epsilon, mean, std)
    elif mode=='normalise':
        x = df[field_name]
        x, mean, std = normalize(x, old_scale)
        scale = ("Normalize", mean, std)
    elif mode=='special':
        x = df[field_name]
        x = np.abs(x)**(1./3.) * np.sign(x)
        x, mean, std = normalize(x, old_scale)
        scale = ("Sqrt3", mean, std)
    else:
        raise ValueError('Scaling mode need for ', field_name)
    df[field_name] = x
    return df, scale

#########################

def main():

    ################
    ### params ###
    file_path = "all_info_df"
    file = ur.open('data/skimmed_full.root') #Akt4EMTopo.topo_cluster.root')
    print('Found file, reading dataset... ')
    # file = ur.open("/data1/atlng02/loch/Summer2022/MLTopoCluster/data/Akt4EMTopo.clusterfiltered.topo-cluster.root")
    # file = ur.open("/home/jmsardain/JetCalib/NewFile.root")
    tree = file["ClusterTree"]
    df = tree.arrays(library="pd")
    
    # dividing dataset for training and test
    df_pos = df[df["clusterE"]>0.]
    n = len(df_pos)
    ntrain = int(n * 0.8)
    ntest = int(n * 0.2)

    df = df_pos[:ntrain]
    df_test_forjet  = df_pos[ntrain:ntrain+ntest]
    df_test = df_test_forjet
    print('\tReading dataset completed \n')

    print('\tApplying cuts...Takes a while...\n')
    df = apply_cuts(df)
    df_test = apply_cuts(df_test)
    df_test_forjet = calculate_response(df_test_forjet)

    print('Training features: \n', df.columns)

    response = ['r_e_calculated']
    column_names = [    'clusterE', 'clusterEta',
                        'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                        'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_time', 'cluster_ISOLATION',
                        'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu',
                    ] # 'weight_response', 'weight_response_wider', 'weight_energy', 'weight_logenergy',

    ref_column_names = ['jetCnt', 'jetNConst', 'nCluster', 'clusterIndex', 'jetCalE', 'jetRawE', 
                        'truthJetE', 'truthJetPt', 'truthJetRap', 'clusterECalib']

    #before = df
    df = df[response+column_names]
    df_test = df_test[response+column_names+ref_column_names]
    df_test_forjet = df_test_forjet[response+column_names+ref_column_names]
    df_pos = df_pos[column_names+ref_column_names]

    #output_path_figures_before_preprocessing = "fig.pdf"
    output_path_figures_after_preprocessing = "fig2.pdf"
    output_path_data = "data/" + file_path + ".npy"

    save = True #True
    scales_txt_file =  output_path_data[:-4] + "_scales.txt"

    ####################
    ### read in data ###

    # df = pd.read_csv(file_path, sep=" ")

    # remove first index
    # df = df.iloc[:, 1:]

    #idx_min = df["cluster_time"].argmin()
    #print(df.shape)
    #df = df.drop(idx_min)
    #print(df.shape)

    print('Training dataset: \n', df)

    #####################
    ### preprocessing ###
    print("-"*100)
    print("Make preprocessing...")

    scales = {}
    # log-preprocessing
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS", "cluster_SECOND_TIME", "cluster_SIGNIFICANCE"]
    for field_name in field_names:
        df_test_forjet, scales[field_name] = apply_scale(df_test_forjet, field_name, 'lognormalise')
        df = apply_scale(df, field_name, 'lognormalise', scales[field_name])[0]
        df_test = apply_scale(df_test, field_name, 'lognormalise', scales[field_name])[0]

    # just normalizing
    field_names = ["clusterEta", "cluster_CENTER_MAG", "nPrimVtx", "avgMu", "cluster_ENG_FRAC_EM", "cluster_LATERAL", 
                "cluster_LONGITUDINAL", "cluster_PTD", "cluster_ISOLATION"]
    for field_name in field_names:
        df_test_forjet, scales[field_name] = apply_scale(df_test_forjet, field_name, 'normalise')
        df = apply_scale(df, field_name, 'normalise', scales[field_name])[0]
        df_test = apply_scale(df_test, field_name, 'normalise', scales[field_name])[0]

    # special preprocessing
    field_name = "cluster_time"
    df_test_forjet, scales[field_name] = apply_scale(df_test_forjet, field_name, 'special')
    df = apply_scale(df, field_name, 'special', scales[field_name])[0]
    df_test = apply_scale(df_test, field_name, 'special', scales[field_name])[0]

    # no preprocessing
    #files_names = ["r_e_calculated"]
    ######################

    print("-"*100)
    print("Make plots after preprocessing...,")

    # plots after preprocessing
    make_all_plots(df, output_path_figures_after_preprocessing)

    if save:
        #brr = before.to_numpy()
        arr = df.to_numpy()
        arr_test = df_test.to_numpy()
        #arr_test_forjet = df_test_forjet.to_numpy()
        print('Training array shape: ', arr.shape)
        print('Feature scale:', scales)
        #print('Saving all unscaled data...')
        #np.save(output_path_data, df_pos.to_numpy())
        #print("Saved as {output_path_data}")
        with open(scales_txt_file, "w") as f:
            f.write(str(scales))

        #n = len(arr)
        #ntrain = int(n * 0.8)
        #ntest = int(n * 0.2)
        #train = arr[:ntrain]
        #test  = arr_w_ref[ntrain:ntrain+ntest]
        ##val  = arr[ntrain+ntest:]
        ### clusterE / response = true energy should be higher than 0.3
        ##test_before= brr[ntrain:ntrain+ntest]

        print('Saving training data...')
        np.save("data/all_info_df_train.npy", arr)
        print('Saving test data after selection...')
        np.save("data/all_info_df_test.npy", arr_test)
        #print('Saving test data without selection...')
        #np.save("data/all_info_df_test_forjet.npy", arr_test_forjet)
        #np.save("data/all_info_df_val.npy", val)
        #np.save("data/all_info_df_test_before.npy", test_before)

if __name__ == "__main__":
    main()
