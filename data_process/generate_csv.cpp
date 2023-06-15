#include <string>
#include <vector>
#include "TFile.h"
#include "TString.h"
#include "TVector3.h"
#include "TTree.h"
#include "TMath.h"
#include "TRandom3.h"
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
using namespace std;
using json = nlohmann::json;

// constants
double c = 0.2998;
double n_water = 1.35;  // for pure water
double c_n = c / n_water;
double costh = 1 / n_water;
double tanth = sqrt(1 - costh*costh) / costh;
double sinth = costh * tanth;
double time_reso = 10.;


int manageTrees(TTree* pmthits, TTree* sipmhits, TTree* prim, json& mc_particles, std::ofstream &outImage, std::ofstream &outPrim, std::ofstream &outNeutrino, int target_pdg);

void genData(){
    /*
     Modify this region to control input and output
    */ 
	int target_pdg = 11; // 11 for e, 13 for u, 15 for tau
	TString dataDir = "/lustre/collider/mocen/project/hailing/data/shower/10TeV/p";
    TString middle_name = "/job_";
    TString suffix = "/data.root";

    TString outFileDir = "./data/";
    TString mc_event_suffix = "/mc_events.json";
    
    int ipart_start = 1, ipart_end = 1;
    int ijob_start = 0, ijob_end = 1;

    int outFileStart = 0;
    
    // main program starts here
    for(int iPart=ipart_start, iOut = outFileStart; iPart<=ipart_end; iPart++){
        for(int ijob=ijob_start; ijob<=ijob_end; ijob++){
            cout<<"Part "<<iPart<<" job "<<ijob<<" -- slice"<<iOut<<endl<<flush;
            TString outfile = outFileDir+"xyz_"+iOut+".csv";
            TFile* infile = new TFile(dataDir + iPart + middle_name + ijob + suffix);
            TTree* pmthits = (TTree*) infile->Get("PmtHit");
            TTree* sipmhits = (TTree*) infile->Get("SipmHit");
            TTree* prim = (TTree*) infile->Get("Primary");

            if(!pmthits || !sipmhits || !prim){ 
                printf("File open error. Skip.\n");
                continue;
            }
            // mc_event.json
            std::ifstream mc_event_file(dataDir + iPart + middle_name + ijob + mc_event_suffix);
            json mc_particles;
            mc_event_file >> mc_particles;

            // Write into csv file
            ofstream outImage;
            outImage.open(outFileDir+"xyz_"+iOut+".csv",ios::out);
            outImage<<"id"<<","
                <<"nhits"<<","<<"t1st"<<","<<"x0"<<","<<"y0"<<","<<"z0"<<","<<"entry"<<","
                <<"M_dist"<<","<<"tInject"<<","<<"xInject"<<","<<"yInject"<<","<<"zInject"
                <<endl;

            ofstream outPrim;
            outPrim.open(outFileDir+"primary_"+iOut+".csv",ios::out);
            outPrim<<"id"<<","<<"PdgId"<<","<<"x0"<<","<<"y0"<<","<<"z0"<<","<<"t0"<<","<<
                "px"<<","<<"py"<<","<<"pz"<<","<<"e0"<<","<<"entry"<<endl;

            ofstream outNeutrino;
            outNeutrino.open(outFileDir+"neutrino_"+iOut+".csv",ios::out);
            outNeutrino<<"id"<<","<<"PdgId"<<","<<"x0"<<","<<"y0"<<","<<"z0"<<","<<"t0"<<","<<
                "px"<<","<<"py"<<","<<"pz"<<","<<"e0"<<","<<"entry"<<","<<"event_id"<<endl;

            int nImages = manageTrees(pmthits, sipmhits, prim, mc_particles, outImage, outPrim, outNeutrino, target_pdg);

            if(nImages > 0)
                iOut++;
            outImage.close();
            outPrim.close();
            outNeutrino.close();
        }
    }
}

int main(){
    genData();
    return 0;
}

int manageTrees(TTree* pmthits, TTree* sipmhits, TTree* prim, json& mc_particles, std::ofstream &outImage, std::ofstream &outPrim, std::ofstream &outNeutrino, int target_pdg){ 
    TRandom3* rnd = new TRandom3(0);
    double photon_eff = 0.3;

    vector<float> *domid=nullptr, *t=nullptr;
    vector<float> *hitx0=nullptr, *hity0=nullptr, *hitz0=nullptr;
    pmthits->SetBranchAddress("DomId", &domid);
    pmthits->SetBranchAddress("t0", &t);
    pmthits->SetBranchAddress("x0", &hitx0);
    pmthits->SetBranchAddress("y0", &hity0);
    pmthits->SetBranchAddress("z0", &hitz0);

    vector<float> *sdomid=nullptr, *st=nullptr;
    vector<float> *shitx0=nullptr, *shity0=nullptr, *shitz0=nullptr;
    sipmhits->SetBranchAddress("DomId", &sdomid);
    sipmhits->SetBranchAddress("t0", &st);
    sipmhits->SetBranchAddress("x0", &shitx0);
    sipmhits->SetBranchAddress("y0", &shity0);
    sipmhits->SetBranchAddress("z0", &shitz0);


    vector<float> *pdg=nullptr, *x0=nullptr, *y0=nullptr, *z0=nullptr, *t0=nullptr, *nx=nullptr, *ny=nullptr, *nz=nullptr, *e0=nullptr;
    prim->SetBranchAddress("PdgId", &pdg);
    prim->SetBranchAddress("x0", &x0);
    prim->SetBranchAddress("y0", &y0);
    prim->SetBranchAddress("z0", &z0);
    prim->SetBranchAddress("t0", &t0);
    prim->SetBranchAddress("px", &nx);
    prim->SetBranchAddress("py", &ny);
    prim->SetBranchAddress("pz", &nz);
    prim->SetBranchAddress("e0", &e0); 
    
    int imageId = 0;
    for(int ientry=0; ientry<pmthits->GetEntries(); ientry++){
        int dom_OutIdP1[30000] = {0}; // id: domid; value: (index+1), where the index 
                                        // is the location of such dom in output vectors
        vector<float> hitX, hitY, hitZ, hitT1st;
        vector<int> nhits;

        pmthits->GetEntry(ientry);
        sipmhits->GetEntry(ientry);
        prim->GetEntry(ientry);
        // cut on hit num and primary
        if(
            (domid->size() + sdomid->size()<2) || 
            ((find(pdg->begin(),pdg->end(),target_pdg)==pdg->end()) &&
             (find(pdg->begin(),pdg->end(),-target_pdg)==pdg->end()))
        ){
            continue;
        }

        // Get SiPM hits information
        double pmtmin = t->size()==0 ? INFINITY : *min_element((*t).begin(), (*t).end());
        double stmin = st->size()==0 ? INFINITY : *min_element((*st).begin(), (*st).end());
        double tmin = stmin-pmtmin>time_reso? pmtmin : stmin;


        int iout=0;
        for(int ihit=0; ihit<(sdomid->size()); ihit++){
            if(rnd->Uniform() > photon_eff)
                continue;
            // Time window is set to be 10us
            if((*st)[ihit]-tmin > 10000)
                continue;
            int currentDomId = (int) (*sdomid)[ihit];
            if(dom_OutIdP1[currentDomId]==0){
                dom_OutIdP1[currentDomId] = iout+1;
                iout++;
                hitX.emplace_back((*shitx0)[ihit]);
                hitY.emplace_back((*shity0)[ihit]);
                hitZ.emplace_back((*shitz0)[ihit]);
                hitT1st.emplace_back((*st)[ihit]);
                nhits.emplace_back(1);
            } else{
                int outId = dom_OutIdP1[currentDomId] - 1;
                if(nhits[outId]==0){
                    cout<<"ERROR: nhits[outId]==0"<<endl;
                }
                hitT1st[outId] = hitT1st[outId]<(*st)[ihit]? hitT1st[outId]:(*st)[ihit];
                nhits[outId]++;
            }
        }  
        // Get pmt hits information
        vector<int> t_is_pmt_hit(hitT1st.size(), 0);
        for(int ihit=0; ihit<(domid->size()); ihit++){
            if(rnd->Uniform() > photon_eff)
                continue;
            // Time window is set to be 10us
            if((*t)[ihit]-tmin > 10000)
                continue;
            int currentDomId = (int) (*domid)[ihit];
            if(dom_OutIdP1[currentDomId]==0){
                dom_OutIdP1[currentDomId] = iout+1;
                iout++;
                hitX.emplace_back((*hitx0)[ihit]);
                hitY.emplace_back((*hity0)[ihit]);
                hitZ.emplace_back((*hitz0)[ihit]);
                hitT1st.emplace_back((*t)[ihit]);
                t_is_pmt_hit.emplace_back(1);
                nhits.emplace_back(1);
            } else{
                int outId = dom_OutIdP1[currentDomId] - 1;
                if(nhits[outId]==0){
                    cout<<"ERROR: nhits[outId]==0"<<endl;
                }
                if(t_is_pmt_hit[outId]==1 && hitT1st[outId]-(*t)[ihit]>0){
                    hitT1st[outId] = (*t)[ihit];
                } else if(t_is_pmt_hit[outId]==0 && hitT1st[outId]-(*t)[ihit]>time_reso){
                    hitT1st[outId] = (*t)[ihit];
                    t_is_pmt_hit[outId] = 1;
                }
                nhits[outId]++;
            }
        }  

        // Cut on dom num
        if(nhits.size() < 2){
            continue;
        }

        // Get primary information
        double totalE=0, muE=0, muX=0, muY=0, muZ=0, muNX=0, muNY=0, muNZ=0, nmuon=0;
        for(int iPrim=0; iPrim<(pdg->size()); iPrim++){
            totalE += (*e0)[iPrim];
            outPrim<<imageId<<","<<(*pdg)[iPrim]<<","<<(*x0)[iPrim]<<","<<(*y0)[iPrim]<<","<<(*z0)[iPrim]<<
                ","<<(*t0)[iPrim]<<","<<(*nx)[iPrim]<<","<<(*ny)[iPrim]<<","<<(*nz)[iPrim]<<","<<(*e0)[iPrim]<<","<<ientry<<endl;

            // Select the most energetic muon
            if( (abs((*pdg)[iPrim])==target_pdg) && ((*e0)[iPrim] > muE)){
                muE = (*e0)[iPrim];
                muX = (*x0)[iPrim];
                muY = (*y0)[iPrim];
                muZ = (*z0)[iPrim];
                muNX = (*nx)[iPrim];
                muNY = (*ny)[iPrim];
                muNZ = (*nz)[iPrim];
                nmuon++;
            }
        }

        // Get neutrino information
        auto ith_event = mc_particles[ientry];
        if( ith_event["particles_in"].size() > 1 ){
            printf("Multiple particles_in in entry %d \n;", ientry);
        }
        auto ith_neutrino = ith_event["particles_in"][0];
        TVector3 nu_P(ith_neutrino["px"], ith_neutrino["py"], ith_neutrino["pz"]);

        outNeutrino<<imageId<<","<<ith_neutrino["pdgid"]<<","
            <<ith_neutrino["x"]<<","<<ith_neutrino["y"]<<","<<ith_neutrino["z"]<<","<<ith_neutrino["t"]<<","
            <<nu_P[0]<<","<<nu_P[1]<<","<<nu_P[2]<<","<<nu_P.Mag()<<","<<ientry<<","<<ith_event["event_id"]<<endl;

        // Calculate M_dist
        double M_dist = 0;
        for(int idom=0; idom<(nhits.size()); idom++){
            for(int jdom=0; jdom<(nhits.size()); jdom++){
                double dist = TVector3((hitX)[idom]-(hitX)[jdom], (hitY)[idom]-(hitY)[jdom], (hitZ)[idom]-(hitZ)[jdom]).Mag();
                M_dist = M_dist > dist? M_dist : dist;
            }
        }

        // Fill hits information
        TVector3 r0(muX, muY, muZ);
        TVector3 n0(muNX, muNY, muNZ);
        n0.SetMag(1);
        for(int idom=0; idom<(nhits.size()); idom++){
            // Calculate r_inject and r_vertical
            TVector3 ri((hitX)[idom],(hitY)[idom],(hitZ)[idom]);
            TVector3 delta_r = ri - r0;
            double l = delta_r.Dot(n0);
            double d = sqrt(delta_r.Mag()*delta_r.Mag() - l * l);
            // injection point
            double lij = l - d / tanth;
            TVector3 rij = r0 + lij * n0 - ri;
            TVector3 rv = r0 + l * n0 - ri;


            outImage<<imageId<<"," 
                <<(nhits)[idom]<<","<<
                (hitT1st)[idom]<<","<<(hitX)[idom]<<","<<(hitY)[idom]<<","<<(hitZ)[idom]<<","
                <<ientry<<"," 
                <<M_dist<<","<<lij/c<<","<<rij[0]<<","<<rij[1]<<","<<rij[2] 
                <<endl;
        }
        imageId++;
    }
    return imageId;
}

