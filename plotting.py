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

def RespVsResponse_Peter(energy, resp, xlabel, ylabel):

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(0, 10,  100 + 1, endpoint=True)
    yaxis = np.linspace(0, 10,  100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h1prof      = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    # BinLogX(h1_backdrop)
    # BinLogX(h1)
    # BinLogX(h1prof)

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
            h1slice.Fit('gaus', '', '', x-0.5, x+0.5)
            g = h1slice.GetListOfFunctions().FindObject("gaus")
            h1prof.SetBinContent(ibinx, g.GetParameter(1))
            h1prof.SetBinError(ibinx, g.GetParameter(2))
        else:
            h1prof.SetBinContent(ibinx, -1)
            h1prof.SetBinError(ibinx, 0)




    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)

    c.ylim(0, 2)
    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,         option='COLZ')
    c.hist(h1prof,        option='PE', markercolor=ROOT.kRed)
    c.hist2d(h1_backdrop, option='AXIS')

    # c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

    return c


def RespVsEnergy_Peter(energy, resp, minx, maxx, logxaxis, xlabel, ylabel):

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(minx,  maxx, 100 + 1, endpoint=True)
    yaxis = np.linspace(0, 10,  100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
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
            h1slice.Fit('gaus', '', '', x-0.5, x+0.5)
            g = h1slice.GetListOfFunctions().FindObject("gaus")
            h1prof.SetBinContent(ibinx, g.GetParameter(1))
            h1prof.SetBinError(ibinx, g.GetParameter(2))
        else:
            h1prof.SetBinContent(ibinx, -1)
            h1prof.SetBinError(ibinx, 0)

    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1_backdrop.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)
    # h1prof.GetYaxis().SetRangeUser(0, 2)

    c.ylim(0, 2)
    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,         option='COLZ')
    c.hist(h1prof,        option='PE', markercolor=ROOT.kRed)
    c.hist2d(h1_backdrop, option='AXIS')

    if logxaxis:
        c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

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
    c.xlabel(r'E^{dep}')
    c.ylabel(r'R^{DNN} / R^{EM}')
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

    return c


def Histo1D(varx, minx, maxx, label='', logxaxis=False):
    c = ap.canvas(num_pads=1, batch=True)

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: BinLogX(h)

    fill_hist(h, varx)

    c.hist(h, option='HIST', linecolor=2)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(label)
    c.ylabel('Events')
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

    return c

def Histo1D_2vars(varx1, varx2, minx, maxx, xlabel='', label1='', label2='', logxaxis=False):
    c = ap.canvas(num_pads=1, batch=True)

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h1 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h2 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: BinLogX(h1)
    if logxaxis: BinLogX(h2)

    fill_hist(h1, varx1)
    fill_hist(h2, varx2)

    c.hist(h1, option='HIST', linecolor=2, label=label1)
    c.hist(h2, option='HIST', linecolor=4, label=label2)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(xlabel)
    c.ylabel('Events')
    c.legend()
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

    return c

def Histo1D_5vars(varx1, varx2, varx3, varx4, varx5, minx, maxx, xlabel='', label1='', label2='', label3='', label4='', label5='', logxaxis=False):
    c = ap.canvas(num_pads=1, batch=True)

    xaxis = np.linspace(minx, maxx, 100 + 1, endpoint=True)

    h1 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h2 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h3 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h4 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h5 = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    if logxaxis: 
        BinLogX(h1)
        BinLogX(h2)
        BinLogX(h3)
        BinLogX(h4)
        BinLogX(h5)

    fill_hist(h1, varx1)
    fill_hist(h2, varx2)
    fill_hist(h3, varx3)
    fill_hist(h4, varx4)
    fill_hist(h5, varx5)

    c.hist(h1, option='HIST', linecolor=2, label=label1)
    c.hist(h2, option='HIST', linecolor=4, label=label2)
    c.hist(h3, option='HIST', linecolor=1, label=label3)
    c.hist(h4, option='HIST', linecolor=6, label=label4)
    c.hist(h5, option='HIST', linecolor=7, label=label5)
    c.log()
    if logxaxis: c.logx()
    c.xlabel(xlabel)
    c.ylabel('Events')
    c.legend()
    c.text(["#sqrt{s} = 13 TeV", "E^{dep} > 0.3 GeV" ], qualifier='Simulation Internal')

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
    outdir = path + '/plots/'
    superscript = 'DNN'
    
    try:
        os.system("mkdir {}".format(outdir))
    except ImportError:
        print("{} already exists".format(outdir))
        pass

    ## Get information
    print('Reading test datasets...\n')
    resp_test = np.load(path+'/trueResponse.npy') # y_test or R_EM
    resp_pred = np.load(path+'/predResponse_redRelu.npy') # y_pred or R_DNN
    x_test    = np.load(path+'/x_test.npy')

    # Re-normalise E_EM from saved scale
    #std_scale = 1.4016793
    #mean_scale =1.4141768
    with open('data/all_info_df_scales.txt') as f:
        lines = f.readlines()
    mean_scale = ast.literal_eval(lines[0])['clusterE'][3]
    std_scale = ast.literal_eval(lines[0])['clusterE'][4]
    energy_log = x_test[:, 0] * std_scale + mean_scale
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
    mu              = x_test[:, 11]
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
    
    jet_i = 0
    jetCnt_save = 0
    sum_E_EM = {}
    sum_E_LCW = {}
    sum_E_DNN = {}
    sum_E_dep = {}
    truthJetE = {}
    truthJetPt = {}
    truthJetRap = {}
    jetCalE = {}

    print('Calculating jet variables from test sample event loop...\n')
    for i in range(x_test.shape[0]):
        if i%10000000==0:
            print('\t cluster entry ', i, ' ...')
        if x_test[i,15] == 0:
            jetCnt_save = x_test[i,12]
            jet_i += 1
            truthJetE[jet_i] = x_test[i,18]
            truthJetPt[jet_i] = x_test[i,19]
            truthJetRap[jet_i] = x_test[i,20]
            jetCalE[jet_i] = x_test[i,16]
            sum_E_EM[jet_i] = 0
            sum_E_LCW[jet_i] = 0
            sum_E_DNN[jet_i] = 0
            sum_E_dep[jet_i] = 0
        if x_test[i,12] == jetCnt_save:
            sum_E_EM[jet_i] += x_test[i,22]  # sum of E_EM, same as jetRawE
            sum_E_LCW[jet_i] += x_test[i,21] # sum of clusterECalib
            sum_E_DNN[jet_i] += x_test[i,24] # sum of E_DNN
            sum_E_dep[jet_i] += x_test[i,23] # sum of E_dep

    truthJetE = np.array(list(truthJetE.values()))
    truthJetPt = np.array(list(truthJetPt.values()))
    truthJetRap = np.array(list(truthJetRap.values()))
    jetCalE = np.array(list(jetCalE.values()))
    sum_E_EM = np.array(list(sum_E_EM.values()))
    sum_E_LCW = np.array(list(sum_E_LCW.values()))
    sum_E_DNN = np.array(list(sum_E_DNN.values()))
    sum_E_dep = np.array(list(sum_E_dep.values()))

    E_response_EM = sum_E_EM/truthJetE
    E_response_LCW = sum_E_LCW/truthJetE
    E_response_DNN = sum_E_DNN/truthJetE
    E_response_dep = sum_E_dep/truthJetE
    E_response_jet = jetCalE/truthJetE

    print('Checking lengths of jet arrays: ', len(truthJetE), len(truthJetPt), len(truthJetRap), len(jetCalE), len(sum_E_EM), len(sum_E_LCW), len(sum_E_DNN), '\n')
    
    ## Start plotting for jets
    try:
        c = Histo1D_5vars(E_response_jet, E_response_dep, E_response_EM, E_response_LCW, E_response_DNN, 0, 10, xlabel=r'Jet Energy Response', 
            label1=r'Calibrated', label2=r'ClusterTruth', label3=r'EM', label4=r'LCW', label5=r'DNN', logxaxis=False)
        c.save(outdir+"/jetresponse_1d.png")
    except AttributeError:
        print('jetresponse_1d.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_jet, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel='JER (Calibrated)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_calibrated_vs_truthJetE.png")
    except AttributeError:
        print('JER_calibrated_vs_truthJetE.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_dep, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel=r'JER (ClusterTruth)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_ClusterTruth_vs_truthJetE.png")
    except AttributeError:
        print('JER_ClusterTruth_vs_truthJetE.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_EM, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel='JER (EM)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_EM_vs_truthJetE.png")
    except AttributeError:
        print('JER_EM_vs_truthJetE.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetE, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'E^{truth}_{jet} [GeV]', ylabel='JER (DNN)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_DNN_vs_truthJetE.png")
    except AttributeError:
        print('JER_DNN_vs_truthJetE.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_jet, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel='JER (Calibrated)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_calibrated_vs_truthJetPt.png")
    except AttributeError:
        print('JER_calibrated_vs_truthJetPt.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_EM, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel='JER (EM)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_EM_vs_truthJetPt.png")
    except AttributeError:
        print('JER_EM_vs_truthJetPt.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetPt, E_response_DNN, 0, 3, logxaxis=True, xlabel=r'Truth p^{T}_{jet} [GeV]', ylabel='JER (DNN)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_DNN_vs_truthJetPt.png")
    except AttributeError:
        print('JER_DNN_vs_truthJetPt.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_jet, -2.5, 2.5, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel='JER (Calibrated)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_calibrated_vs_truthJetRap.png")
    except AttributeError:
        print('JER_calibrated_vs_truthJetRap.png is not produced')
    
    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_EM, -2.5, 2.5, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel='JER (EM)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_EM_vs_truthJetRap.png")
    except AttributeError:
        print('JER_EM_vs_truthJetRap.png is not produced')

    try:
        c = RespVsEnergy_Peter(truthJetRap, E_response_DNN, -2.5, 2.5, logxaxis=False, xlabel=r'Truth jet rapidity', ylabel='JER (DNN)')
        c.ylim(0, 2)
        c.save(outdir+"/JER_DNN_vs_truthJetRap.png")
    except AttributeError:
        print('JER_DNN_vs_truthJetRap.png is not produced')

    ## Start plotting for clusters
    """
    try:
        c = Histo1D(resp_pred/resp_test, 0, 10, label=r'R^{'+superscript+'} / R^{EM}', logxaxis=False)
        c.save(outdir+"/ratio.png")
    except AttributeError:
        print('ratio.png is not produced')
    """
    try:
        c = Histo1D_2vars(trueE, predE, -1, 3, xlabel='Energy [GeV]', label1=r'E^{dep}',label2=r'E^{'+superscript+'}', logxaxis=True)
        c.save(outdir+"/energy_1d.png")
    except AttributeError:
        print('energy_1d.png is not produced')

    try:
        c = Histo1D_2vars(resp_test, resp_pred, 0, 10, xlabel='Response', label1=r'R^{em}',label2=r'R^{'+superscript+'}', logxaxis=False)
        c.save(outdir+"/response_1d.png")
    except AttributeError:
        print('response_1d.png is not produced')
    """
    try:
        c = RespVsEnergy_Peter(trueE, resp_pred/resp_test, -1, 3, logxaxis=True, xlabel=r'E^{dep} [GeV]', ylabel=r'R^{'+superscript+'} / R^{EM}')
        c.ylim(0, 2)
        c.save(outdir+"/Rpred_Over_Rem_vs_Edep.png")
    except AttributeError:
        print('Rpred_Over_Rem_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, predE/recoE, -1, 3, logxaxis=True, xlabel=r'E^{dep} [GeV]', ylabel=r'E^{'+superscript+'} / E^{EM}')
        c.ylim(0, 2)
        c.save(outdir+"/Epred_Over_Eem_vs_Edep.png")
    except AttributeError:
        print('Epred_Over_Eem_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(recoE, predE/trueE, -1, 3, logxaxis=True, xlabel=r'E^{EM} [GeV]', ylabel=r'E^{'+superscript+'} / E^{dep}')
        c.ylim(0, 2)
        c.save(outdir+"/Epred_Over_Edep_vs_Eem.png")
    except AttributeError:
        print('Epred_Over_Edep_vs_Eem.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, resp_pred, -1, 3, logxaxis=True, xlabel=r'E^{dep} [GeV]', ylabel=r'R^{'+superscript+'}')
        c.ylim(0, 2)
        c.save(outdir+"/Rpred_vs_Edep.png")
    except AttributeError:
        print('Rpred_vs_Edep.png is not produced')

    try:
        c = RespVsEnergy_Peter(trueE, resp_test, -1, 3, logxaxis=True, xlabel=r'E^{dep} [GeV]', ylabel=r'R^{EM}')
        c.ylim(0, 2)
        c.save(outdir+"/Rem_vs_Edep.png")
    except AttributeError:
        print('Rem_vs_Edep.png is not produced')

    try:
        c = RespVsResponse_Peter(resp_test, resp_pred/resp_test, xlabel=r'R^{EM}', ylabel=r'R^{'+superscript+'} / R^{EM}')
        c.ylim(0, 2)
        c.save(outdir+"/Rpred_Over_Rem_vs_Rem.png")
    except AttributeError:
        print('Rpred_Over_Rem_vs_Rem.png is not produced')
    """
    return


# Main function call.
if __name__ == '__main__':
    main()
    pass
