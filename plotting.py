import ROOT
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mode, iqr
import math
import argparse
import rootplotting as ap
from rootplotting.tools import *
from root_numpy import fill_hist
import ast


def BinLogX(h):
    axis = h.GetXaxis()
    bins = axis.GetNbins()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins[i] = pow(10, a + i * width)

    axis.Set(bins, newbins)
    del newbins
    pass

def Median_and_Relative_Uncertainty(x, y1, y2, minx, maxx, logxaxis, xlabel, ylabel, label1='', label2='', miny=-0.5, maxy=5, AllCluster=False, printbin=False):

    if not AllCluster:
        rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"
    else:
        rangeEnergy = ""

    #plot Median & Relative Uncertainty (IQR/2*median) of y1,y2 in bins of x
    c0 = ap.canvas(batch=True, size=(600,600))
    c0.pads()[0]._bare().SetRightMargin(0.2)
    c0.pads()[0]._bare().SetLogz()
    c1 = ap.canvas(batch=True, size=(600,600))
    c1.pads()[0]._bare().SetRightMargin(0.2)
    c1.pads()[0]._bare().SetLogz()
    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)
    yaxis = np.linspace(miny, maxy, 100 + 1, endpoint=True)

    h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    h1CalcResponse = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h1PredResponse = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    if logxaxis:
        BinLogX(h1a)
        BinLogX(h1b)
        BinLogX(h1CalcResponse)
        BinLogX(h1PredResponse)

    #h2a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    #h2b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    h1CalcResolution = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h1PredResolution = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    if logxaxis:
        #BinLogX(h2a)
        #BinLogX(h2b)
        BinLogX(h1CalcResolution)
        BinLogX(h1PredResolution)

    mesh1a = np.vstack((x, y1)).T
    mesh1b = np.vstack((x, y2)).T
    #mesh2a = np.vstack((x, y1)).T
    #mesh2b = np.vstack((x, y2)).T

    fill_hist(h1a, mesh1a)
    fill_hist(h1b, mesh1b)
    #fill_hist(h2a, mesh2a)
    #fill_hist(h2b, mesh2b)

    for ibinx in range(1, h1a.GetNbinsX()+1):
        median_binX = []
        for ibiny in range(1, h1a.GetNbinsY()+1):
            n = int(h1a.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                median_binX.append(h1a.GetYaxis().GetBinCenter(ibiny))
                pass
        if not median_binX:
            continue
        #print(mode(mode_binX)[0].flatten())
        calcMedian =  np.median(median_binX)
        calcRelUnc =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
        h1CalcResponse.SetBinContent(ibinx, calcMedian)
        h1CalcResolution.SetBinContent(ibinx, calcRelUnc)
        h1CalcResponse.SetBinError(ibinx, calcRelUnc)
        h1CalcResolution.SetBinError(ibinx, 0)
        if printbin:
            print('Histo 1: bin ',ibinx,' error: ',h1CalcResponse.GetBinError(ibinx))

    for ibinx in range(1, h1b.GetNbinsX()+1):
        median_binX = []
        for ibiny in range(1, h1b.GetNbinsY()+1):
            n = int(h1b.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                median_binX.append(h1b.GetYaxis().GetBinCenter(ibiny))
                pass
        if not median_binX:
            continue
        predMedian =  np.median(median_binX)
        predRelUnc =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
        h1PredResponse.SetBinContent(ibinx, predMedian)
        h1PredResolution.SetBinContent(ibinx, predRelUnc)
        h1PredResponse.SetBinError(ibinx, predRelUnc)
        h1PredResolution.SetBinError(ibinx, 0)
        if printbin:
            print('Histo 2: bin ',ibinx,' error: ',h1PredResponse.GetBinError(ibinx))

    c0.hist(h1CalcResponse, option='PLE', markersize=0.5, markercolor=2, linecolor=2, label=label1)
    c0.hist(h1PredResponse, option='PLE', markersize=0.5, markercolor=4, linecolor=4, label=label2)
    if logxaxis:
        c0.logx()
    c0.legend()
    c0.xlabel(xlabel)
    c0.ylabel("Median [%s]" %ylabel)
    c0.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
    #c0.save("./plots/{}/Median_Reponse.png".format(args.rangeE))

    c1.hist(h1CalcResolution, option='PL', markersize=0.5, markercolor=2, linecolor=2, label=label1)
    c1.hist(h1PredResolution, option='PL', markersize=0.5, markercolor=4, linecolor=4, label=label2)
    if logxaxis:
        c1.logx()
    c1.legend()
    c1.xlabel(xlabel)
    c1.ylabel(r'Relative Uncertainty, Q^{w}_{f=68%} / (2 #times Median)'+' [%s]' %ylabel)
    c1.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
    #c1.save("./plots/{}/IQR_Reponse.png".format(args.rangeE))

    return c0,c1

def RespVsEnergy_Peter(energy, resp, minx, maxx, logxaxis, xlabel, ylabel, miny=0, maxy=10, AllCluster=False, printbin=False):

    if not AllCluster:
        rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"
    else:
        rangeEnergy = ""

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)
    yaxis = np.linspace(miny, maxy, 100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1] ])) # 0.75* 
    h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h1prof      = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    if logxaxis:
        BinLogX(h1_backdrop)
        BinLogX(h1)
        BinLogX(h1prof)

    mesh = np.vstack((energy, resp)).T
    fill_hist(h1, mesh)

    for ibinx in range(1, h1.GetNbinsX()+1):
        c1 = ap.canvas(batch=True, size=(600,600))
        h1slice = h1.ProjectionY('h1slice'+str(ibinx), ibinx, ibinx)
        c1.hist(h1slice,        option='PE', markercolor=ROOT.kRed)
        #c1.save("out/plots/h1slice"+str(ibinx)+".png")
        if h1slice.GetEntries() > 0:
            binmax = h1slice.GetMaximumBin()
            x = h1slice.GetXaxis().GetBinCenter(binmax)
            #xmin = h1slice.GetXaxis().GetBinCenter(1)
            #xmax = h1slice.GetXaxis().GetBinCenter(h1slice.GetNbinsX())
            h1slice.Fit('gaus', '', '', x-0.5, x+0.5)
            #h1slice.Fit('gaus', '', '', xmin, xmax)
            g = h1slice.GetListOfFunctions().FindObject("gaus")
            h1prof.SetBinContent(ibinx, g.GetParameter(1))
            h1prof.SetBinError(ibinx, g.GetParameter(2))
        else:
            h1prof.SetBinContent(ibinx, 0) # originally -1
            h1prof.SetBinError(ibinx, 0)
        if printbin:
            print(ibinx, h1prof.GetBinContent(ibinx), h1prof.GetBinError(ibinx))

    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)

    #c.ylim(0, 2)
    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,         option='COLZ')
    c.hist(h1prof,        option='PE', markercolor=ROOT.kBlack) #ROOT.kRed
    c.hist2d(h1_backdrop, option='AXIS')

    if logxaxis:
        c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal') #[, "E^{dep}_{clus} > 0.3 GeV" ], 
    return c

def RespVsEnergy(energy, resp):

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
    yaxis = np.linspace(0, 10,  100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ]))
    h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h1prof      = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    BinLogX(h1_backdrop)
    BinLogX(h1)
    BinLogX(h1prof)

    mesh = np.vstack((energy, resp)).T
    fill_hist(h1, mesh)
    for ibinx in range(1, h1.GetNbinsX()+1):
        mode_binX = []
        for ibiny in range(0, h1.GetNbinsY()+2):
            n = int(h1.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                mode_binX.append(h1.GetYaxis().GetBinCenter(ibiny))
                pass
        if not mode_binX:
            continue
        h1prof.SetBinContent(ibinx, mode(mode_binX)[0].flatten())
        h1prof.SetBinError(ibinx, 0)


    # h1.GetZaxis().SetRangeUser(1, 1e3)
    # h1.GetZaxis().SetRangeUser(1, 1e3)
    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,         option='COLZ')
    c.hist(h1prof,        option='P', markercolor=ROOT.kRed)
    c.hist2d(h1_backdrop, option='AXIS')

    c.logx()
    c.xlabel(r'E^{dep}_{clus}')
    c.ylabel(r'R^{DNN training} / R^{EM scale}')
    c.text(["#sqrt{s} = 13 TeV", "E^{dep}_{clus} > 0.3 GeV" ], qualifier='Simulation Internal')
    return c

def Histo1D(varx, minx, maxx, label='', logxaxis=False, AllCluster=False):
    c = ap.canvas(num_pads=1, batch=True)
    
    if not AllCluster:
        rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"
    else:
        rangeEnergy = ""

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: BinLogX(h)

    fill_hist(h, varx)

    c.hist(h, option='HIST', linecolor=2)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(label)
    c.ylabel('Events')
    c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ) ], qualifier='Simulation Internal')
    return c

def Histo1D_2vars(varx1, varx2, minx, maxx, xlabel='', ylabel='', label1='', label2='', logxaxis=False, AllCluster=False):
    c = ap.canvas(num_pads=1, batch=True)

    if not AllCluster:
        rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"
    else:
        rangeEnergy = ""

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h1 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h2 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: 
        BinLogX(h1)
        BinLogX(h2)

    fill_hist(h1, varx1)
    fill_hist(h2, varx2)

    c.hist(h1, option='HIST', linecolor=2, label=label1)
    c.hist(h2, option='HIST', linecolor=4, label=label2)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.legend()
    c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ) ], qualifier='Simulation Internal')
    return c

def Histo1D_3vars(varx1, varx2, varx3, minx, maxx, logxaxis=False, xlabel='', ylabel='', label1='', label2='', label3='', AllCluster=False):
    c = ap.canvas(num_pads=1, batch=True)

    if not AllCluster:
        rangeEnergy = "E^{dep}_{clus} > 0.3 GeV"
    else:
        rangeEnergy = ""

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h1 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h2 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h3 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    #h4 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    #h5 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: 
        BinLogX(h1)
        BinLogX(h2)
        BinLogX(h3)
        #BinLogX(h4)
        #BinLogX(h5)

    fill_hist(h1, varx1)
    fill_hist(h2, varx2)
    fill_hist(h3, varx3)
    #fill_hist(h4, varx4)
    #fill_hist(h5, varx5)

    c.hist(h1, option='HIST', linecolor=2, label=label1)
    c.hist(h2, option='HIST', linecolor=4, label=label2)
    c.hist(h3, option='HIST', linecolor=1, label=label3)
    #c.hist(h4, option='HIST', linecolor=6, label=label4)
    #c.hist(h5, option='HIST', linecolor=7, label=label5)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.legend()
    c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ) ], qualifier='Simulation Internal')
    return c

def main():
    ROOT.gStyle.SetPalette(ROOT.kBird)

    """
    for i in range(0, 1):
        ## Take correct path
        if i==0:  path = 'out/' #'/home/jmsardain/JetCalib/DNN/train_test_1tanh/'
        if i==1:  path = '/home/jmsardain/JetCalib/DNN/train_noweight_100epochs_batch2048_lr0p0001/'
        if i==1:  path = '/home/jmsardain/JetCalib/DNN/train_weightEnergy_100epochs_batch2048_lr0p0001/'
        if i==2:  path = '/home/jmsardain/JetCalib/DNN/train_weightLogEnergy_100epochs_batch2048_lr0p0001/'
        if i==3:  path = '/home/jmsardain/JetCalib/DNN/train_weightResp_100epochs_batch2048_lr0p0001/'
        if i==4:  path = '/home/jmsardain/JetCalib/DNN/train_weightRespWider_100epochs_batch2048_lr0p0001/'
        if i==5:  path = '/home/jmsardain/JetCalib/DNN/train_noweight_100epochs_batch2048_lr0p0001_RatioLoss/'
        if i==6:  path = '/home/jmsardain/JetCalib/DNN/train_weightEnergy_100epochs_batch2048_lr0p0001_RatioLoss/'
        if i==7:  path = '/home/jmsardain/JetCalib/DNN/train_weightLogEnergy_100epochs_batch2048_lr0p0001_RatioLoss/'
        if i==8:  path = '/home/jmsardain/JetCalib/DNN/train_weightResp_100epochs_batch2048_lr0p0001_RatioLoss/'
        if i==9:  path = '/home/jmsardain/JetCalib/DNN/train_weightRespWider_100epochs_batch2048_lr0p0001_RatioLoss/'
        if i==10: path = '/home/jmsardain/JetCalib/old/MLClusterCalibration/BNN/output_moretest5/'

        if i ==10: 
            superscript = 'BNN'
        else:
    """
    
    path = 'out/'
    phase_space = 'AllClusters/' #'AllClusters/' # 'selectedClusters/'
    model_name = "original" # "redRelu" #"original"
    outdir_clus = path + 'plots/' + phase_space + 'cluster/' + model_name + '/'
    outdir_jet  = path + 'plots/' + phase_space + 'jet/'     + model_name + '/'
    superscript = 'DNN training'

    AllCluster_flag = False
    if phase_space == 'AllClusters/':
        AllCluster_flag = True

    try:
        os.system("mkdir {} {}".format(outdir_clus, outdir_jet))
    except ImportError:
        print("{}, {} already exists".format(outdir_clus, outdir_jet))
        pass

    ## Get information
    print('Reading test datasets...\n')

    # dataset with all selections applied
    if phase_space == 'selectedClusters/':
        resp_test = np.load(path+'/trueResponse.npy') # y_test or R_EM
        x_test    = np.load(path+'/x_test.npy')
        if model_name == "redRelu":
            resp_pred = np.load(path+'/predResponse_redRelu.npy') # y_pred or R_DNN
        else:
            resp_pred = np.load(path+'/predResponse_original.npy') # y_pred or R_DNN

    # dataset without any selection other than cluster response > 0.1
    elif phase_space == 'AllClusters/':
        resp_test = np.load(path+'/trueResponse_forjet.npy') # y_test or R_EM
        x_test    = np.load(path+'/x_test_forjet.npy')
        if model_name == "redRelu":
            resp_pred = np.load(path+'/predResponse_forjet_redRelu.npy') # y_pred or R_DNN
        else:
            resp_pred = np.load(path+'/predResponse_forjet_original.npy') # y_pred or R_DNN

    # Re-normalise E_EM from saved scale
    #std_scale = 1.4016793
    #mean_scale =1.4141768
    with open('data/all_info_df_scales.txt') as f:
        lines = f.readlines()
    mean_scale_clusterE = ast.literal_eval(lines[0])['clusterE'][3]
    std_scale_clusterE = ast.literal_eval(lines[0])['clusterE'][4]
    energy_log = x_test[:, 0] * std_scale_clusterE + mean_scale_clusterE
    energy = np.exp(energy_log)

    trueE = energy * 1. / resp_test # E_dep = E_EM/R_EM
    ################# NOT SAFE ########################## TO BE FIXED
    predE = energy * 1. / resp_pred # E_DNN = E_EM/R_DNN
    #####################################################
    recoE = energy                  # E_EM                  # same as clusterE

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
    
    mean_scale_clusterEta = ast.literal_eval(lines[0])['clusterEta'][1]
    std_scale_clusterEta = ast.literal_eval(lines[0])['clusterEta'][2]
    mean_scale_nPrimVtx = ast.literal_eval(lines[0])['nPrimVtx'][1]
    std_scale_nPrimVtx = ast.literal_eval(lines[0])['nPrimVtx'][2]
    mean_scale_avgMu = ast.literal_eval(lines[0])['avgMu'][1]
    std_scale_avgMu = ast.literal_eval(lines[0])['avgMu'][2]

    clusterEta = x_test[:, 1] * std_scale_clusterEta + mean_scale_clusterEta
    nPrimVtx = x_test[:, 10] * std_scale_nPrimVtx + mean_scale_nPrimVtx
    avgMu = x_test[:, 11] * std_scale_avgMu + mean_scale_avgMu

    jet_i = 0
    jetCnt_save = 0
    sum_E_EM = {}
    sum_E_EM_low = {}
    sum_E_LCW = {}
    sum_E_DNN = {}
    sum_E_DNN_low = {}
    sum_E_dep = {}
    pileup_nPV = {}
    pileup_mu = {}
    truthJetE = {}
    truthJetPt = {}
    truthJetRap = {}
    jetCalE = {}

    print('Calculating jet variables from test sample cluster list...\n')
    for i in range(x_test.shape[0]):
        if i%10000000==0:
            print('\t cluster entry ', i, ' ...')
        if x_test[i,15] == 0:
            jetCnt_save = x_test[i,12]
            jet_i += 1
            pileup_nPV[jet_i] = nPrimVtx[i]
            pileup_mu[jet_i] = avgMu[i]
            truthJetE[jet_i] = x_test[i,18]
            truthJetPt[jet_i] = x_test[i,19]
            truthJetRap[jet_i] = x_test[i,20]
            jetCalE[jet_i] = x_test[i,16]
            sum_E_EM[jet_i] = 0
            sum_E_EM_low[jet_i] = 0
            sum_E_LCW[jet_i] = 0
            sum_E_DNN[jet_i] = 0
            sum_E_DNN_low[jet_i] = 0
            sum_E_dep[jet_i] = 0
        if x_test[i,12] == jetCnt_save:
            sum_E_EM[jet_i] += x_test[i,22]  # sum of E_EM, same as jetRawE
            sum_E_LCW[jet_i] += x_test[i,21] # sum of clusterECalib
            sum_E_DNN[jet_i] += x_test[i,24] # sum of E_DNN
            sum_E_dep[jet_i] += x_test[i,23] # sum of E_dep
            if x_test[i, 23] < 0.3:
                sum_E_EM_low[jet_i] += x_test[i,22]
                sum_E_DNN_low[jet_i] += x_test[i,24]

    pileup_nPV = np.array(list(pileup_nPV.values()))
    pileup_mu = np.array(list(pileup_mu.values()))
    truthJetE = np.array(list(truthJetE.values()))
    truthJetPt = np.array(list(truthJetPt.values()))
    truthJetRap = np.array(list(truthJetRap.values()))
    jetCalE = np.array(list(jetCalE.values()))
    sum_E_EM = np.array(list(sum_E_EM.values()))
    sum_E_EM_low = np.array(list(sum_E_EM_low.values()))
    sum_E_LCW = np.array(list(sum_E_LCW.values()))
    sum_E_DNN = np.array(list(sum_E_DNN.values()))
    sum_E_DNN_low = np.array(list(sum_E_DNN_low.values()))
    sum_E_dep = np.array(list(sum_E_dep.values()))

    E_response_EM = sum_E_EM/truthJetE
    E_response_LCW = sum_E_LCW/truthJetE
    E_response_DNN = sum_E_DNN/truthJetE
    E_response_dep = sum_E_dep/truthJetE
    E_response_jet = jetCalE/truthJetE
    fraction_lowE_EM = sum_E_EM_low/sum_E_EM
    fraction_lowE_DNN = sum_E_DNN_low/sum_E_DNN
    E_clusTruthResponse_EM = sum_E_EM/sum_E_dep
    E_clusTruthResponse_DNN = sum_E_DNN/sum_E_dep
    E_clusTruthResponse_truthjet = truthJetE/sum_E_dep

    print('Checking lengths of jet arrays: ', len(truthJetE), len(truthJetPt), len(truthJetRap), len(jetCalE), len(sum_E_EM), len(sum_E_LCW), len(sum_E_DNN), '\n')
    """
    ## Start plotting for jets
    try:
        c = Histo1D_3vars(E_response_jet, E_response_EM, E_response_DNN, -2, 2, logxaxis=True, xlabel=r'Jet Response, E_{jet}^{#kappa}/E_{jet}^{truth}', ylabel=r'No. of jets', 
        label1=r'Calibrated', label2=r'EM scale', label3=r'DNN training', AllCluster=AllCluster_flag) #E_response_dep, E_response_LCW, label2=r'ClusterTruth', label4=r'LCW', 
        c.save(outdir_jet+"/jetresponse_1d.png")
    except AttributeError:
        print('jetresponse_1d.png is not produced')
    
    try:
        c = Histo1D_3vars(E_clusTruthResponse_truthjet, E_clusTruthResponse_EM, E_clusTruthResponse_DNN, -2, 2, logxaxis=True, xlabel=r'ClusterTruth Response, E_{jet}^{#kappa}/#Sigma E_{dep}', ylabel=r'No. of jets', 
        label1=r'Truth Jet Energy', label2=r'#Sigma E_{EM}', label3=r'#Sigma E_{DNN}', AllCluster=AllCluster_flag) #E_response_dep, E_response_LCW, label2=r'ClusterTruth', label4=r'LCW', 
        c.save(outdir_jet+"/jetClusTruthResponse_1d.png")
    except AttributeError:
        print('jetClusTruthResponse_1d.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetPt, E_clusTruthResponse_truthjet, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'E_{jet}^{truth}/#Sigma E_{dep}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetClusTruthResponse_truth_vs_truthJetPt.png")
    except AttributeError:
        print('JetClusTruthResponse_truth_vs_truthJetPt.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetPt, E_clusTruthResponse_EM, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'#Sigma E_{EM}/#Sigma E_{dep}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetClusTruthResponse_EM_vs_truthJetPt.png")
    except AttributeError:
        print('JetClusTruthResponse_EM_vs_truthJetPt.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetPt, E_clusTruthResponse_DNN, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'#Sigma E_{DNN}/#Sigma E_{dep}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetClusTruthResponse_DNN_vs_truthJetPt.png")
    except AttributeError:
        print('JetClusTruthResponse_DNN_vs_truthJetPt.png is not produced')

    
    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_jet, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'E_{jet}^{Calibrated}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_calibrated_vs_truthJetE.png")
    except AttributeError:
        print('JetResponse_calibrated_vs_truthJetE.png is not produced')

    #try:
    #    c = RespVsEnergy_Peter(truthJetE, E_response_dep, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'Jet Response(ClusterTruth)')
    #    c.ylim(0, 2)
    #    c.save(outdir+"/JetResponse_ClusterTruth_vs_truthJetE.png")
    #except AttributeError:
    #    print('JetResponse_ClusterTruth_vs_truthJetE.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_EM, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'E_{jet}^{EM scale}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_EM_vs_truthJetE.png")
    except AttributeError:
        print('JetResponse_EM_vs_truthJetE.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'E_{jet}^{DNN training}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_DNN_vs_truthJetE.png")
    except AttributeError:
        print('JetResponse_DNN_vs_truthJetE.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_jet, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'E_{jet}^{Calibrated}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_calibrated_vs_truthJetPt.png")
    except AttributeError:
        print('JetResponse_calibrated_vs_truthJetPt.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_EM, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'E_{jet}^{EM scale}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_EM_vs_truthJetPt.png")
    except AttributeError:
        print('JetResponse_EM_vs_truthJetPt.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'E_{jet}^{DNN training}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_DNN_vs_truthJetPt.png")
    except AttributeError:
        print('JetResponse_DNN_vs_truthJetPt.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_jet, -3, 3, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel=r'E_{jet}^{Calibrated}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_calibrated_vs_truthJetRap.png")
    except AttributeError:
        print('JetResponse_calibrated_vs_truthJetRap.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_EM, -3, 3, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel=r'E_{jet}^{EM scale}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_EM_vs_truthJetRap.png")
    except AttributeError:
        print('JetResponse_EM_vs_truthJetRap.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_DNN, -3, 3, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel=r'E_{jet}^{DNN training}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_DNN_vs_truthJetRap.png")
    except AttributeError:
        print('JetResponse_DNN_vs_truthJetRap.png is not produced')

    try:
        c = RespVsEnergy_Peter(pileup_nPV, E_response_jet, 0, 60, logxaxis=False, xlabel=r'N_{PV}', ylabel=r'E_{jet}^{Calibrated}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_calibrated_vs_pileup_nPV.png")
    except AttributeError:
        print('JetResponse_calibrated_vs_pileup_nPV.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(pileup_nPV, E_response_EM, 0, 60, logxaxis=False, xlabel=r'N_{PV}', ylabel=r'E_{jet}^{EM scale}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_EM_vs_pileup_nPV.png")
    except AttributeError:
        print('JetResponse_EM_vs_pileup_nPV.png is not produced')

    try:
        c = RespVsEnergy_Peter(pileup_nPV, E_response_DNN, 0, 60, logxaxis=False, xlabel=r'N_{PV}', ylabel=r'E_{jet}^{DNN training}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_DNN_vs_pileup_nPV.png")
    except AttributeError:
        print('JetResponse_DNN_vs_pileup_nPV.png is not produced')

    try:
        c = RespVsEnergy_Peter(pileup_mu, E_response_jet, 0, 80, logxaxis=False, xlabel=r'#mu_{avg}', ylabel=r'E_{jet}^{Calibrated}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_calibrated_vs_pileup_mu.png")
    except AttributeError:
        print('JetResponse_calibrated_vs_pileup_mu.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(pileup_mu, E_response_EM, 0, 80, logxaxis=False, xlabel=r'#mu_{avg}', ylabel=r'E_{jet}^{EM scale}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_EM_vs_pileup_mu.png")
    except AttributeError:
        print('JetResponse_EM_vs_pileup_mu.png is not produced')

    try:
        c = RespVsEnergy_Peter(pileup_mu, E_response_DNN, 0, 80, logxaxis=False, xlabel=r'#mu_{avg}', ylabel=r'E_{jet}^{DNN training}/E_{jet}^{truth}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/JetResponse_DNN_vs_pileup_mu.png")
    except AttributeError:
        print('JetResponse_DNN_vs_pileup_mu.png is not produced')
    
    try:
        c0,c1 = Median_and_Relative_Uncertainty(truthJetE, E_response_EM, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'E_{jet}^{#kappa}/E_{jet}^{truth}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=7, AllCluster=AllCluster_flag)
        c0.save(outdir_jet+"/MedianJet_vs_truthJetE.png")
        c1.save(outdir_jet+"/RelUncJet_vs_truthJetE.png")
    except AttributeError:
        print('MedianJet_vs_truthJetE.png or RelUncJet_vs_truthJetE.png is not produced')
    
    try:
        c0,c1 = Median_and_Relative_Uncertainty(truthJetPt, E_response_EM, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel=r'E_{jet}^{#kappa}/E_{jet}^{truth}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=7, AllCluster=AllCluster_flag)
        c0.save(outdir_jet+"/MedianJet_vs_truthJetPt.png")
        c1.save(outdir_jet+"/RelUncJet_vs_truthJetPt.png")
    except AttributeError:
        print('MedianJet_vs_truthJetPt.png or RelUncJet_vs_truthJetPt.png is not produced')

    try:
        c0,c1 = Median_and_Relative_Uncertainty(truthJetRap, E_response_EM, E_response_DNN, -3, 3, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel=r'E_{jet}^{#kappa}/E_{jet}^{truth}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=7, AllCluster=AllCluster_flag)
        c0.save(outdir_jet+"/MedianJet_vs_truthJetRap.png")
        c1.save(outdir_jet+"/RelUncJet_vs_truthJetRap.png")
    except AttributeError:
        print('MedianJet_vs_truthJetRap.png or RelUncJet_vs_truthJetRap.png is not produced')


    ### plot to check fraction of low Edep clusters in a jet
    try:
        c = RespVsEnergy_Peter(truthJetE, fraction_lowE_EM, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel='Fraction of E_{jet}^{EM scale} from E^{dep}_{clus}<0.3 GeV clusters', miny=-0.5, maxy=1.5, AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/fraction_lowE_EM_vs_truthJetE.png")
    except AttributeError:
        print('fraction_lowE_EM_vs_truthJetE.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetE, fraction_lowE_DNN, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel='Fraction of E_{jet}^{DNN training} from E^{dep}_{clus}<0.3 GeV clusters', miny=-0.5, maxy=1.5, AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_jet+"/fraction_lowE_DNN_vs_truthJetE.png")
    except AttributeError:
        print('fraction_lowE_DNN_vs_truthJetE.png is not produced')

    ### plots for clusters

    try:
        c = Histo1D(resp_pred/resp_test, 0, 10, label=r'R^{'+superscript+r'} / R^{EM scale}', logxaxis=False, AllCluster=AllCluster_flag)
        c.save(outdir_clus+"/ratio.png")
    except AttributeError:
        print('ratio.png is not produced')
    """
    try:
        c = Histo1D_2vars(trueE, recoE, -1, 3, xlabel='Cluster Energy [GeV]', ylabel='No. of clusters', label1=r'E^{dep}_{clus}',label2=r'E^{EM}_{clus}', logxaxis=True, AllCluster=AllCluster_flag)
        c.save(outdir_clus+"/energy_EMvsDep_1d.png")
    except AttributeError:
        print('energy_EMvsDep_1d.png is not produced')
    """
    try:
        c = Histo1D_2vars(trueE, predE, -1, 3, xlabel='Cluster Energy [GeV]', ylabel='No. of clusters', label1=r'E^{dep}_{clus}',label2=r'E^{'+superscript+r'}_{clus}', logxaxis=True, AllCluster=AllCluster_flag)
        c.save(outdir_clus+"/energy_1d.png")
    except AttributeError:
        print('energy_1d.png is not produced')

    try:
        c = Histo1D_2vars(resp_test, resp_pred, -2, 2, xlabel='Cluster Response', ylabel='No. of clusters', label1='EM scale',label2=superscript, logxaxis=True, AllCluster=AllCluster_flag)
        c.save(outdir_clus+"/response_1d.png")
    except AttributeError:
        print('response_1d.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, resp_pred/resp_test, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'R^{'+superscript+r'} / R^{EM scale}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Rpred_Over_Rem_vs_Edep.png")
    except AttributeError:
        print('Rpred_Over_Rem_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, predE/recoE, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'E^{'+superscript+r'}_{clus} / E^{EM scale}_{clus}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Epred_Over_Eem_vs_Edep.png")
    except AttributeError:
        print('Epred_Over_Eem_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(recoE, predE/trueE, -1, 3, logxaxis=True, xlabel=r'E^{EM scale}_{clus} [GeV]', ylabel=r'E^{'+superscript+r'}_{clus} / E^{dep}_{clus}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Epred_Over_Edep_vs_Eem.png")
    except AttributeError:
        print('Epred_Over_Edep_vs_Eem.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, predE/trueE, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'E^{'+superscript+r'}_{clus} / E^{dep}_{clus}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Epred_Over_Edep_vs_Edep.png")
    except AttributeError:
        print('Epred_Over_Edep_vs_Eem.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, resp_pred, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'R^{'+superscript+'}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Rpred_vs_Edep.png")
    except AttributeError:
        print('Rpred_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, resp_test, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'R^{EM scale}', miny=-0.5, maxy=10, AllCluster=AllCluster_flag, printbin=True)
        #c.ylim(-1, 2)
        c.save(outdir_clus+"/Rem_vs_Edep.png")
    except AttributeError:
        print('Rem_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(resp_test, resp_pred/resp_test, 0, 10, logxaxis=False, xlabel=r'R^{EM scale}', ylabel=r'R^{'+superscript+r'} / R^{EM scale}', AllCluster=AllCluster_flag)
        c.ylim(0, 2)
        c.save(outdir_clus+"/Rpred_Over_Rem_vs_Rem.png")
    except AttributeError:
        print('Rpred_Over_Rem_vs_Rem.png is not produced')
    
    try:
        c0,c1 = Median_and_Relative_Uncertainty(trueE, resp_test, resp_test/resp_pred, -1, 3, logxaxis=True, xlabel=r'E^{dep}_{clus} [GeV]', ylabel=r'E^{#kappa}_{clus}/E^{dep}_{clus}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        c0.save(outdir_clus+"/MedianClus_vs_Edep.png")
        c1.save(outdir_clus+"/RelUncClus_vs_Edep.png")
    except AttributeError:
        print('MedianClus_vs_Edep.png or RelUncClus_vs_Edep.png is not produced')
    
    try:
        c0,c1 = Median_and_Relative_Uncertainty(clusterEta, resp_test, resp_test/resp_pred, -3, 3, logxaxis=False, xlabel=r'#eta_{clus}', ylabel=r'E^{#kappa}_{clus}/E^{dep}_{clus}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        c0.save(outdir_clus+"/MedianClus_vs_eta.png")
        c1.save(outdir_clus+"/RelUncClus_vs_eta.png")
    except AttributeError:
        print('MedianClus_vs_Edep.png or RelUncClus_vs_Edep.png is not produced')

    try:
        c0,c1 = Median_and_Relative_Uncertainty(nPrimVtx, resp_test, resp_test/resp_pred, 0, 60, logxaxis=False, xlabel=r'N_{PV}', ylabel=r'E^{#kappa}_{clus}/E^{dep}_{clus}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        c0.save(outdir_clus+"/MedianClus_vs_NPV.png")
        c1.save(outdir_clus+"/RelUncClus_vs_NPV.png")
    except AttributeError:
        print('MedianClus_vs_NPV.png or RelUncClus_vs_NPV.png is not produced')

    try:
        c0,c1 = Median_and_Relative_Uncertainty(avgMu, resp_test, resp_test/resp_pred, 0, 80, logxaxis=False, xlabel=r'#mu_{avg}', ylabel=r'E^{#kappa}_{clus}/E^{dep}_{clus}', label1='EM scale', label2='DNN training', miny=-0.5, maxy=10, AllCluster=AllCluster_flag)
        c0.save(outdir_clus+"/MedianClus_vs_avgMu.png")
        c1.save(outdir_clus+"/RelUncClus_vs_avgMu.png")
    except AttributeError:
        print('MedianClus_vs_avgMu.png or RelUncClus_vs_avgMu.png is not produced')
    """
    return


# Main function call.
if __name__ == '__main__':
    main()
    pass
