void skim(){
	// skimming Peter's sample to save with seqNumber%1000==0 out of total events: 138884772
	float cluster_ENG_CALIB_TOT_t=0.;
	float clusterE_t=0.;
	float cluster_CENTER_LAMBDA_t=0.;
	float cluster_FIRST_ENG_DENS_t=0.;
	float cluster_SECOND_TIME_t=0.;
	float cluster_SIGNIFICANCE_t=0.;
	float clusterEta_t=0.;
	float cluster_CENTER_MAG_t=0.;
	int nPrimVtx_t=0;
	float avgMu_t=0.;
	float cluster_ENG_FRAC_EM_t=0.;
	float cluster_LATERAL_t=0.;
	float cluster_LONGITUDINAL_t=0.;
	float cluster_PTD_t=0.;
	float cluster_ISOLATION_t=0.;
	float cluster_time_t=0.;
	Long64_t jetCnt=0;
	int jetNConst=0;
	int nCluster=0;
	int clusterIndex=0;
	float jetCalE=0.;
	float jetRawE=0.;
	float truthJetE=0.;
	float truthJetPt=0.;
	float truthJetRap=0.;
	float clusterECalib=0.;

	TTree *t_out; TFile *outfile;

	//outfile = new TFile("/eos/home-a/arej/temp/MLCalib_skim/skimmed.root", "RECREATE");
	outfile = new TFile("/ceph/groups/e4/users/arej/MLClusterCalibration/data/skimmed_full.root", "RECREATE");
	t_out = new TTree("ClusterTree", "ClusterTree");

	t_out->Branch("cluster_ENG_CALIB_TOT", &cluster_ENG_CALIB_TOT_t);
	t_out->Branch("clusterE", &clusterE_t);
	t_out->Branch("cluster_CENTER_LAMBDA", &cluster_CENTER_LAMBDA_t);
	t_out->Branch("cluster_FIRST_ENG_DENS", &cluster_FIRST_ENG_DENS_t);
	t_out->Branch("cluster_SECOND_TIME", &cluster_SECOND_TIME_t);
	t_out->Branch("cluster_SIGNIFICANCE", &cluster_SIGNIFICANCE_t);
	t_out->Branch("clusterEta", &clusterEta_t);
	t_out->Branch("cluster_CENTER_MAG", &cluster_CENTER_MAG_t);
	t_out->Branch("nPrimVtx", &nPrimVtx_t);
	t_out->Branch("avgMu", &avgMu_t);
	t_out->Branch("cluster_ENG_FRAC_EM", &cluster_ENG_FRAC_EM_t);
	t_out->Branch("cluster_LATERAL", &cluster_LATERAL_t);
	t_out->Branch("cluster_LONGITUDINAL", &cluster_LONGITUDINAL_t);
	t_out->Branch("cluster_PTD", &cluster_PTD_t);
	t_out->Branch("cluster_ISOLATION", &cluster_ISOLATION_t);
	t_out->Branch("cluster_time", &cluster_time_t);
	t_out->Branch("jetCnt", &jetCnt);
	t_out->Branch("jetNConst", &jetNConst);
	t_out->Branch("nCluster", &nCluster);
	t_out->Branch("clusterIndex", &clusterIndex);
	t_out->Branch("jetCalE", &jetCalE);
	t_out->Branch("jetRawE", &jetRawE);
	t_out->Branch("truthJetE", &truthJetE);
	t_out->Branch("truthJetPt", &truthJetPt);
	t_out->Branch("truthJetRap", &truthJetRap);
	t_out->Branch("clusterECalib", &clusterECalib);
	
	//TFile *f = TFile::Open("/eos/home-l/loch/MLClusterCalibration/data.06.21.2023/Akt4EMTopo.topo-cluster.root");
	TFile *f = TFile::Open("/ceph/groups/e4/users/arej/MLClusterCalibration/data/Akt4EMTopo.topo-cluster.root");
	TTree *t = (TTree*)f->Get("ClusterTree");
	cout<<"ClusterTree has "<<t->GetEntries()<<" entries..."<<endl;
	
	t->SetBranchAddress("cluster_ENG_CALIB_TOT", &cluster_ENG_CALIB_TOT_t);
	t->SetBranchAddress("clusterE", &clusterE_t);
	t->SetBranchAddress("cluster_CENTER_LAMBDA", &cluster_CENTER_LAMBDA_t);
	t->SetBranchAddress("cluster_FIRST_ENG_DENS", &cluster_FIRST_ENG_DENS_t);
	t->SetBranchAddress("cluster_SECOND_TIME", &cluster_SECOND_TIME_t);
	t->SetBranchAddress("cluster_SIGNIFICANCE", &cluster_SIGNIFICANCE_t);
	t->SetBranchAddress("clusterEta", &clusterEta_t);
	t->SetBranchAddress("cluster_CENTER_MAG", &cluster_CENTER_MAG_t);
	t->SetBranchAddress("nPrimVtx", &nPrimVtx_t);
	t->SetBranchAddress("avgMu", &avgMu_t);
	t->SetBranchAddress("cluster_ENG_FRAC_EM", &cluster_ENG_FRAC_EM_t);
	t->SetBranchAddress("cluster_LATERAL", &cluster_LATERAL_t);
	t->SetBranchAddress("cluster_LONGITUDINAL", &cluster_LONGITUDINAL_t);
	t->SetBranchAddress("cluster_PTD", &cluster_PTD_t);
	t->SetBranchAddress("cluster_ISOLATION", &cluster_ISOLATION_t);
	t->SetBranchAddress("cluster_time", &cluster_time_t);
	t->SetBranchAddress("jetCnt", &jetCnt);
	t->SetBranchAddress("jetNConst", &jetNConst);
	t->SetBranchAddress("nCluster", &nCluster);
	t->SetBranchAddress("clusterIndex", &clusterIndex);
	t->SetBranchAddress("jetCalE", &jetCalE);
	t->SetBranchAddress("jetRawE", &jetRawE);
	t->SetBranchAddress("truthJetE", &truthJetE);
	t->SetBranchAddress("truthJetPt", &truthJetPt);
	t->SetBranchAddress("truthJetRap", &truthJetRap);
	t->SetBranchAddress("clusterECalib", &clusterECalib);
	
	for(int i=0;i<t->GetEntries();++i){ //t->GetEntries()
		t->GetEntry(i); 
		//if(i%1000==0){
		if(i%1000000==0){cout<<"Passed "<<i<<" ..."<<endl;}
		t_out->Fill();
		//}
	}
	f->Close();

	outfile->ReOpen("UPDATE");
	t_out->Write();
	outfile->Close();
}