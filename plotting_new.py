import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import ast

"""
To make plots for the calibration ML performance at the cluster level and also at jet level

cluster level plots:
    1. R_EM vs R_EM(Peter) vs R_DNN,   1D histo, log-x, log-y, x-range: [e-2:e2]
    2. E_EM vs E_EM(Peter) vs E_dep,   1D histo, log-x, log-y, x-range: [e-1:e3]
    3. E_DNN vs E_dep vs E_dep(Peter), 1D histo, log-x, log-y, x-range: [e-1:e3]

    function:
    1. Histo1D_Nvars(var_arr_list, minx, maxx, xlabel='', ylabel='', var_name_list=[], logxaxis=False, extra_legend=False)
"""

def Histo1D_Nvars(
    var_arr_list, 
    minx, 
    maxx, 
    xlabel='', 
    ylabel='', 
    var_name_list=[], 
    filepath='',
    normalize=True,
    logxaxis=False, 
    add_logy=False, 
    ATLAS_lable=' Simulation Internal',
    extra_legend=False):
    
    if not add_logy:
        fig, ax1 = plt.subplots(figsize=[5.5*2, 5.])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[5.5*2, 5.])
    mycolorcycle = ['b', 'c', 'r', 'g', 'k', 'y', 'm', 'w']

    ax1.set_xlabel(r"%s"%xlabel)
    if normalize:
        ylabel += ' (normalized)'
    ax1.set_ylabel(r"%s"%ylabel)
    if logxaxis:
        ax1.set_xscale("log")

    if add_logy:
        ax2.set_title("log scale (y-axis)")
        ax2.set_xlabel(r"%s"%xlabel)
        ax2.set_ylabel(r"%s"%ylabel)
        ax2.set_yscale("log")
        if logxaxis:
            ax2.set_xscale("log")

    rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)
    if logxaxis:
        xaxis = np.logspace(minx, maxx, 100 + 1, endpoint=True)

    for v,var in enumerate(var_arr_list):
        ax1.hist(var, bins=xaxis, histtype="step", density=normalize, color=mycolorcycle[v], label=var_name_list[v])
        if add_logy:
            ax2.hist(var, bins=xaxis, histtype="step", density=normalize, color=mycolorcycle[v], label=var_name_list[v])
    ax1.legend(loc='upper right')
    # atlas_legend = ['ATLAS'+ATLAS_lable, r"$sqrt{s} = 13 TeV$"]
    # if extra_legend:
    #     atlas_legend += [r"$%s$" % (rangeEnergy)]
    # plt.legend(labels=atlas_legend, loc='upper left')

    plt.savefig(filepath)
    print('\tsaved plot in ', filepath)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make plots')
    parser.add_argument('--phase_space', default='selectedClusters', help='selectedClusters or AllClusters')
    parser.add_argument('--filepath', default='out', help='file path for the input and output')
    parser.add_argument('--scale_filepath', default='', help='file path for the input scales')
    parser.add_argument('--model', default='redRelu', help='model name for inference: redRelu or original')
    args = parser.parse_args()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    path = args.filepath+'/' 
    phase_space = args.phase_space #'AllClusters/' # 'selectedClusters/'
    model_name = args.model # "redRelu" #"original"
    outdir_clus = path+'plots/'+phase_space+'/cluster/'+model_name+'/'
    outdir_jet  = path+'plots/'+phase_space+'/jet/'    +model_name+'/'
    scale_filepath = args.scale_filepath

    AllCluster_flag = False
    if phase_space == 'AllClusters':
        AllCluster_flag = True

    try:
        os.system("mkdir {} {}".format(outdir_clus, outdir_jet))
    except ImportError:
        print("{}, {} already exists".format(outdir_clus, outdir_jet))
        pass

    ## Get information
    print('Reading test datasets...\n')

    # dataset with all selections applied
    if phase_space == 'selectedClusters':
        resp_test = np.load(path+'/trueResponse.npy') # y_test or R_EM
        x_test    = np.load(path+'/x_test.npy')
        # resp_test_peter = np.load(path+'Peter/trueResponse.npy')
        # x_test_peter    = np.load(path+'Peter/x_test.npy')
        resp_pred = np.load(path+'/predResponse_'+model_name+'.npy') # y_pred or R_DNN

    # dataset without any selection other than cluster response > 0.1
    elif phase_space == 'AllClusters':
        resp_test = np.load(path+'/trueResponse_forjet.npy') # y_test or R_EM
        x_test    = np.load(path+'/x_test_forjet.npy')
        resp_pred = np.load(path+'/predResponse_forjet_'+model_name+'.npy') # y_pred or R_DNN

    with open(scale_filepath+'/all_info_df_scales.txt') as f:
        lines = f.readlines()
    mean_scale_clusterE = ast.literal_eval(lines[0])['clusterE'][3]
    std_scale_clusterE = ast.literal_eval(lines[0])['clusterE'][4]
    energy_log = x_test[:, 0] * std_scale_clusterE + mean_scale_clusterE
    energy = np.exp(energy_log)

    # with open('data/Peter/all_info_df_scales.txt') as f:
    #     lines_peter = f.readlines()
    # mean_scale_clusterE_peter = ast.literal_eval(lines_peter[0])['clusterE'][3]
    # std_scale_clusterE_peter = ast.literal_eval(lines_peter[0])['clusterE'][4]
    # energy_log_peter = x_test_peter[:, 0] * std_scale_clusterE_peter + mean_scale_clusterE_peter
    # energy_peter = np.exp(energy_log_peter)
    # trueE_peter = energy_peter * 1. / resp_test_peter

    trueE = energy * 1. / resp_test # E_dep = E_EM/R_EM
    ################# NOT SAFE ########################## TO BE FIXED
    predE = energy * 1. / resp_pred # E_DNN = E_EM/R_DNN
    #####################################################
    recoE = energy                  # E_EM   # same as clusterE

    x_test = np.hstack((x_test, energy.reshape(-1,1)))
    x_test = np.hstack((x_test, trueE.reshape(-1,1)))
    x_test = np.hstack((x_test, predE.reshape(-1,1)))

    """
    clusterE                = x_test[:, 0]
    clusterEta              = x_test[:, 1]
    cluster_CENTER_LAMBDA   = x_test[:, 2]
    cluster_CENTER_MAG      = x_test[:, 3]
    cluster_ENG_FRAC_EM     = x_test[:, 4]
    cluster_FIRST_ENG_DENS  = x_test[:, 5]
    cluster_PTD             = x_test[:, 6]
    cluster_time            = x_test[:, 7]
    cluster_ISOLATION       = x_test[:, 8]
    cluster_SIGNIFICANCE    = x_test[:, 9]
    nPrimVtx        = x_test[:, 10]
    avgMu           = x_test[:, 11]
    jetCnt          = x_test[:, 12]
    jetNConst       = x_test[:, 13]
    nCluster        = x_test[:, 14]
    clusterIndex    = x_test[:, 15]
    jetCalE         = x_test[:, 16]
    jetRawE         = x_test[:, 17] # sum of clusterE or E_EM
    truthJetE       = x_test[:, 18]
    truthJetPt      = x_test[:, 19]
    truthJetRap     = x_test[:, 20]
    clusterECalib   = x_test[:, 21] # same as clusterE ?
    energy          = x_test[:, 22]
    trueE           = x_test[:, 23]
    predE           = x_test[:, 24]
    """
    
    # mean_scale_clusterEta = ast.literal_eval(lines[0])['clusterEta'][1]
    # std_scale_clusterEta = ast.literal_eval(lines[0])['clusterEta'][2]
    # mean_scale_nPrimVtx = ast.literal_eval(lines[0])['nPrimVtx'][1]
    # std_scale_nPrimVtx = ast.literal_eval(lines[0])['nPrimVtx'][2]
    # mean_scale_avgMu = ast.literal_eval(lines[0])['avgMu'][1]
    # std_scale_avgMu = ast.literal_eval(lines[0])['avgMu'][2]

    # clusterEta = x_test[:, 1] * std_scale_clusterEta + mean_scale_clusterEta
    # nPrimVtx = x_test[:, 10] * std_scale_nPrimVtx + mean_scale_nPrimVtx
    # avgMu = x_test[:, 11] * std_scale_avgMu + mean_scale_avgMu

    # jet_i = 0
    # jetCnt_save = 0
    # sum_E_EM = {}
    # sum_E_EM_low = {}
    # sum_E_LCW = {}
    # sum_E_DNN = {}
    # sum_E_DNN_low = {}
    # sum_E_dep = {}
    # pileup_nPV = {}
    # pileup_mu = {}
    # truthJetE = {}
    # truthJetPt = {}
    # truthJetRap = {}
    # jetCalE = {}

    # print('Calculating jet variables from test sample cluster list...\n')
    # for i in range(x_test.shape[0]):
    #     if i%10000000==0:
    #         print('\t cluster entry ', i, ' ...')
    #     if x_test[i,15] == 0:
    #         jetCnt_save = x_test[i,12]
    #         jet_i += 1
    #         pileup_nPV[jet_i] = nPrimVtx[i]
    #         pileup_mu[jet_i] = avgMu[i]
    #         truthJetE[jet_i] = x_test[i,18]
    #         truthJetPt[jet_i] = x_test[i,19]
    #         truthJetRap[jet_i] = x_test[i,20]
    #         jetCalE[jet_i] = x_test[i,16]
    #         sum_E_EM[jet_i] = 0
    #         sum_E_EM_low[jet_i] = 0
    #         sum_E_LCW[jet_i] = 0
    #         sum_E_DNN[jet_i] = 0
    #         sum_E_DNN_low[jet_i] = 0
    #         sum_E_dep[jet_i] = 0
    #     if x_test[i,12] == jetCnt_save:
    #         sum_E_EM[jet_i] += x_test[i,22]  # sum of E_EM, same as jetRawE
    #         sum_E_LCW[jet_i] += x_test[i,21] # sum of clusterECalib
    #         sum_E_DNN[jet_i] += x_test[i,24] # sum of E_DNN
    #         sum_E_dep[jet_i] += x_test[i,23] # sum of E_dep
    #         if x_test[i, 23] < 0.3:
    #             sum_E_EM_low[jet_i] += x_test[i,22]
    #             sum_E_DNN_low[jet_i] += x_test[i,24]

    # pileup_nPV = np.array(list(pileup_nPV.values()))
    # pileup_mu = np.array(list(pileup_mu.values()))
    # truthJetE = np.array(list(truthJetE.values()))
    # truthJetPt = np.array(list(truthJetPt.values()))
    # truthJetRap = np.array(list(truthJetRap.values()))
    # jetCalE = np.array(list(jetCalE.values()))
    # sum_E_EM = np.array(list(sum_E_EM.values()))
    # sum_E_EM_low = np.array(list(sum_E_EM_low.values()))
    # sum_E_LCW = np.array(list(sum_E_LCW.values()))
    # sum_E_DNN = np.array(list(sum_E_DNN.values()))
    # sum_E_DNN_low = np.array(list(sum_E_DNN_low.values()))
    # sum_E_dep = np.array(list(sum_E_dep.values()))

    # E_response_EM = sum_E_EM/truthJetE
    # E_response_LCW = sum_E_LCW/truthJetE
    # E_response_DNN = sum_E_DNN/truthJetE
    # E_response_dep = sum_E_dep/truthJetE
    # E_response_jet = jetCalE/truthJetE
    # fraction_lowE_EM = sum_E_EM_low/sum_E_EM
    # fraction_lowE_DNN = sum_E_DNN_low/sum_E_DNN
    # E_clusTruthResponse_EM = sum_E_EM/sum_E_dep
    # E_clusTruthResponse_DNN = sum_E_DNN/sum_E_dep
    # E_clusTruthResponse_truthjet = truthJetE/sum_E_dep

    # print('Checking lengths of jet arrays: ', len(truthJetE), len(truthJetPt), len(truthJetRap), len(jetCalE), len(sum_E_EM), len(sum_E_LCW), len(sum_E_DNN), '\n')

    response_list = [resp_test, resp_pred]
    response_name_list = [r'R_{EM}', r'R_{DNN}']
    Histo1D_Nvars(
        response_list, 
        -2, 3, 
        xlabel='Cluster Response', ylabel='No. of clusters', 
        var_name_list=response_name_list, 
        filepath=outdir_clus+"/response_1d.png", 
        normalize=False,
        logxaxis=True, add_logy=True, 
        extra_legend=AllCluster_flag)

    E_EM_list = [recoE, trueE, predE]
    E_EM_name_list = [r'E_{EM}', r'E_{dep}', r'E_{DNN}']
    Histo1D_Nvars(
        E_EM_list, 
        -1, 3, 
        xlabel='Energy [GeV]', ylabel='No. of clusters', 
        var_name_list=E_EM_name_list, 
        filepath=outdir_clus+"/energy_1d.png", 
        normalize=False,
        logxaxis=True, add_logy=True, 
        extra_legend=AllCluster_flag)
