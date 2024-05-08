// =========================================================================
// HiddenFoot Genome Wide V0.1 (May 2024)
// CREATED BY NACHO MOLINA
// =========================================================================
//
// Expected output for version 1.0
// - Fitted paramters: qU, qB, beta and concentrations
// - Expected binding profiles genome wide 

#include <iomanip>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <algorithm>

#ifdef USE_OMP
#include <omp.h>
#endif

using namespace std;

// Definition of types
// ---------------------------------------------------------
typedef struct {
  int sgd;
  int sam;
  int bnd;
} typerunmod;
  
typedef struct {
  char *seq;
  char *met;
  char *wms;
  char *par;
  char *bnd;
  char *out;
} typefiles; 

typedef struct {
  int nuclen;
  int padlen;
  int maxwms;
  double bgprob;
  double pseudo; 
  double cutoffwms;
} typeparwms;

typedef struct {
  int fitgamma; 
  int numepochs;
  int batchsize;
  double beta1;
  double beta2;
  double lrate;
} typeparsgd;

typedef struct {
  int fitgamma;
  int numiter;
  double sigma;
} typeparsam;

typedef struct {
  string name;
  int len;
  double **W;
  double **rW;
  double maxscore;
} datawms;

typedef struct {
  int lenseq;
  int padlen;
  int numelm;
  int numwms;
  int *seq;
  int *pos;
  int *ind;
  int *bnd;
  int *len;
  int *mapwm;
  double *probW;
} dataseq;

typedef struct {
  int nummol;
  int numsites; 
  int  *msites;
  int **mstate;
  int **U;
  int **M;
} datamet; 

int numthreads = 4;
double outcutoff = 0;
double priorb = 0; 
double priork = 1;

// Definition of functions
// ---------------------------------------------------------
datamet readmet(char *filemet, typeparwms parwms);
dataseq readseq(char *fileseq, typeparwms parwms);
datawms* readwms(char *filewms, typeparwms *parwms);
void parse_argv(int argc, char **argv, typefiles *files, typerunmod *runmod, typeparwms *parwms, typeparsgd *parsgd, typeparsam *parsam, typeparsim *parsim);
void select_wms(dataseq *seq, datawms *wms, typeparwms parwms, char *filebnd, char *fileout);
void get_methcounts(dataseq seq, datamet *met);
void get_probW(dataseq *seq, datawms *wms, double bgprob);
void get_probS(int lenseq, int *sequence, int maxwms, datawms *wms, double **probS);
void get_probB(int numelm, int *wms, double *probW, int numwms, double *params, double *probB);
void get_probM(int *U, int *M, int numelm, int* bnd, int numwms, double *params, double *probM);
void forward_conf(int numelm, int lenseq, int *pos, int *len, double *probB, double *Z);
void forward_data(int numelm, int lenseq, int *pos, int *len, double *probM, double *probB, double *F);
void backward_conf(int numelm, int lenseq, int *pos, int *len, double *probB, double *Z);
void backward_data(int numelm, int lenseq, int *pos, int *len, double *probM, double *probB, double *B);
void forward_conf_grad(int numelm, int lenseq, int numwms, int *pos, int *len, int *ind, double *probB, double *dprobB, double *P, double *G);
void forward_data_grad(int numelm, int lenseq, int numwms, int *pos, int *len, int *ind, double *probM, double *dprobMdU, double *dprobMdB, double *probB, double *dprobB, double *P, double *G);
void get_probM_grad(int *U, int *M, int numelm, int* bnd, int numwms, double *params, double *probM, double *dprobMdU, double *dprobMdB);
void get_probB_grad(int numelm, int *ind, double *probW, int numwms, double *params, double *probB, double *dprobB);
void para_invtransformation(int numwms, double *tparams, double *params);
void para_transformation(int numwms, double *params, double *tparams);
void get_params_MCMC(dataseq seq, datamet met, datawms *wms, double *params, typeparsam parsam, char *file);
void get_params_SGD(dataseq seq, datamet met, datawms *wms, double *params, typeparsgd parsgd, char *file);
void get_binding(dataseq seq, datamet met, datawms *wms, double *params, char *file);
double loglikelihood(dataseq seq, datamet met, double *params);
double loglikelihood_grad(dataseq seq, datamet met, double *params, vector<int> selmol, double *dlogP);
double loglikelihood_numgrad(dataseq seq, datamet met, double *tparams, vector<int> selmol, double *dlogP);
double loglikelihood_semigrad(dataseq seq, datamet met, double *tparams, vector<int> selmol, double *dlogP);
double* init_params(int numwms, char *filepar);
vector<int> random_permutation (int numelem);
int funbnd(int t);
void errorout();


// =========================================================================
// MAIN 
// =========================================================================
int main(int argc, char** argv) {

  // Initialitation: 
  // ----------------------------------------------------

  // Time
  time_t t0, tf;
  double dt;
  t0 = time(0);

  // Variables
  datawms *wms;
  dataseq seq;
  datamet met;
  double *params;
  typefiles files;
  typeparwms parwms;
  typeparsgd parsgd;
  typeparsam parsam;
  typerunmod runmod; 

  // Reading arguments
  parse_argv(argc,argv,&files,&runmod,&parwms,&parsgd,&parsam);
  
  // Reading data from files
  // ----------------------------------------------------
  
  // Read sequence
  seq = readseq(files.seq,parwms);
    
  // Read methylation matrix
  met = readmet(files.met,parwms);

  // Read wms file
  wms = readwms(files.wms,&parwms);


  // Preprocessing
  // ----------------------------------------------------
  
  // Selectiing WMs with high-score BS
  select_wms(&seq,wms,parwms,files.bnd,files.out);

  // Calculating methlyation count matrices
  get_methcounts(seq,&met);

  // Initialziie params 
  params = init_params(seq.numwms,files.par);

  
  // Running HiddenFoot
  // ----------------------------------------------------
  
  // fittiing parameters with SGD
  if (runmod.sgd == 1) {get_params_SGD(seq,met,wms,params,parsgd,files.out);}

  // Estimating posteriors with MCMC 
  //if (runmod.sam == 1) {get_params_MCMC(seq,met,wms,params,parsam,files.out);}

  // Computing binding profiiles
  if (runmod.bnd == 1) {get_binding(seq,met,wms,params,files.out);}

  // Time
  // ----------------------------------------------------
  tf = time(0);
  dt = difftime(tf, t0) / 60.0;
  cout << "TIME: " << setprecision(10) << dt << " min" << endl;
}

// ======================================================
// FUNCTIONS
// ======================================================

// Generates simulated binding profiiles based on the
// prior Boltzman model and the methtlatiion data 
// ------------------------------------------------------
void get_simulations_data(dataseq seq, datamet met, datawms *wms, double *tparams, typeparsim parsim, char *file) {
  int lenseq = seq.lenseq;
  int padlen = seq.padlen;  
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int simmol = parsim.nummol;
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int *mapwm = seq.mapwm;
  int nummol = met.nummol;
  int numsim = nummol*simmol;
  int numsites = met.numsites;
  int *msites = met.msites;
  int **U = met.U;
  int **M = met.M; 
  double *probW = seq.probW;
  
  double B[lenseq+1];
  double probM[numelm];
  double probB[numelm];
  double params[numwms+3];
  
  // Output
  cout << "RUN: Generateed simulated profiles: " << numsim << " molecues, " << numwms << " motifs" << endl; 

  // Open files
  string fileout;  
  string outtag(file);
  ofstream out[numwms];
  for (int t = 0; t < numwms; t++) {
    fileout = outtag + ".simdata." +  wms[mapwm[t]].name + ".txt";
    out[t].open(fileout.c_str());
  }
  
  // Random number generator 
  default_random_engine generator((unsigned) clock());
  uniform_real_distribution<double> uniform(0.0,1.0);
  
  // Transforming parameters
  para_invtransformation(numwms,tparams,params); 
  double q[2] = {params[numwms+1],params[numwms+2]};
  
  // Calculating probB
  get_probB(numelm,ind,probW,numwms,params,probB);

  // Calculatng commulative probabilities in each positon
  double Q[lenseq][numwms];
  int S[lenseq][numwms];
  int L[lenseq][numwms];
  int P[numwms][lenseq];
  int numq[lenseq];

  //exit(1);
  
  // Simulating binding and methylation profiles
  for (int m = 0; m < nummol; m++) {

    // Calculating probabM
    get_probM(U[m], M[m], numelm, bnd, numwms, params, probM);
    
    // Calculating partition function
    backward_data(numelm, lenseq, pos, len, probM, probB, B);

    // Calculatng commulative probabilities in each positon
    for (int i = 0; i < lenseq; i++) {numq[i] = 0;}
    for (int n = 0; n < numelm; n++) {
      int i = pos[n];
      int l = len[n];
      int t = ind[n];
      
      if (numq[i] == 0) {Q[i][numq[i]] = probM[n]*probB[n]*B[i+l]/B[i];}
      else              {Q[i][numq[i]] = Q[i][numq[i]-1] + probM[n]*probB[n]*B[i+l]/B[i];}
      S[i][numq[i]] = t;
      L[i][numq[i]] = l;
      numq[i]++;
    }
  
    for (int r = 0; r < simmol; r++) {
      
      // Initializing P
      for (int t = 0; t < numwms; t++) {
	for (int i = 0; i < lenseq; i++) {
	  P[t][i] = 0.0;
	}
      }
      
      int i = 0;
      while (i < lenseq) {
	int n = 0;
	//while (Q[i][n] < N[i]*uniform(generator)) {n++;}
	double rand  = uniform(generator); 
	while (Q[i][n] < rand) {n++;}
	
	int t = S[i][n];
	int l = L[i][n];
	for (int s = i; s < i+l; s++) {
	  P[t][s] = 1;
	}
	i += l;
      }
  
      // Printing out simulations
      for (int t = 0; t < numwms; t++) {
	for (int i = padlen; i < lenseq-padlen-1; i++) {
	  out[t] << P[t][i] << " ";
	}
	out[t] << P[t][lenseq-padlen-1] << endl;
      }
    }
  }
  
  // Clossing files 
  for (int t = 0; t < numwms; t++) {out[t].close();}
}


// Generates simulated binding profiiles and methylation
// based on the prior Boltzman model
// ------------------------------------------------------
void get_simulations(dataseq seq, datamet met, datawms *wms, double *tparams, typeparsim parsim, char *file) {
  int lenseq = seq.lenseq;
  int padlen = seq.padlen;  
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int nummol = parsim.nummol;
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int *mapwm = seq.mapwm;
  int numsites = met.numsites;
  int *msites = met.msites;
  double *probW = seq.probW;
  
  double B[lenseq+1];
  double probM[numelm];
  double probB[numelm];
  double params[numwms+3];

  // Output
  cout << "RUN: Generateed simulated data: " << nummol << " molecues, " << numwms << " motifs" << endl; 

  // Open files
  string outtag(file);
  string fileout = outtag + ".simdata.methylation.txt";
  ofstream outm(fileout.c_str());
  for (int i = 0; i < numsites-1; i++) {
    outm << msites[i]-padlen << " ";
  }
  outm << msites[numsites-1]-padlen << endl;
  
  ofstream out[numwms];
  for (int t = 0; t < numwms; t++) {
    fileout = outtag + ".simdata." +  wms[mapwm[t]].name + ".txt";
    out[t].open(fileout.c_str());
  }
  
  // Random number generator 
  default_random_engine generator((unsigned) clock());
  uniform_real_distribution<double> uniform(0.0,1.0);
  
  // Transforming parameters
  para_invtransformation(numwms,tparams,params); 
  double q[2] = {params[numwms+1],params[numwms+2]};
  
  // Calculating probB
  get_probB(numelm,ind,probW,numwms,params,probB);

  // Calculating partition function
  backward_conf(numelm, lenseq, pos, len, probB, B);

  // Calculatng commulative probabilities in each positon
  double Q[lenseq][numwms];
  int S[lenseq][numwms];
  int L[lenseq][numwms];
  int P[numwms][lenseq];
  int M[lenseq];
  //int N[lenseq]; 
  int numq[lenseq];

  for (int i = 0; i < lenseq; i++) {numq[i] = 0;}
  
  for (int n = 0; n < numelm; n++) {
    int i = pos[n];
    int l = len[n];
    int t = ind[n];

    if (numq[i] == 0) {Q[i][numq[i]] = probB[n]*B[i+l]/B[i];}
    else              {Q[i][numq[i]] = Q[i][numq[i]-1] + probB[n]*B[i+l]/B[i];}
    S[i][numq[i]] = t;
    L[i][numq[i]] = l;
    //N[i] = Q[i][numq[i]];
    //cerr << i << " " << t << " " << l << " " << numq[i] << " " << probB[n]*B[i+l]/B[i] << " " << Q[i][numq[i]] << endl;
    numq[i]++;
  }
  //for (int i = 0; i < lenseq; i++) {
  //cerr << i << " " << numq[i] << " " << Q[i][numq[i]-1] << endl;
  //}
  //exit(1);
  
  // Simulating binding and methylation profiles
  for (int m = 0; m < nummol; m++) {

    // Initializing P
    for (int t = 0; t < numwms; t++) {
      for (int i = 0; i < lenseq; i++) {
	P[t][i] = 0.0;
      }
    }

    int i = 0;
    while (i < lenseq) {
      int n = 0;
      //while (Q[i][n] < N[i]*uniform(generator)) {n++;}
      double rand  = uniform(generator); 
      while (Q[i][n] < rand) {n++;}
      
      int t = S[i][n];
      int l = L[i][n];
      for (int s = i; s < i+l; s++) {
	P[t][s] = 1;
      }
      i += l;
    }
  
    // Printing out simulations
    for (int i = 0; i < numsites; i++) {
      int s = msites[i];

      if (q[P[0][s]] < uniform(generator)) {M[i] = 0;}
      else                                 {M[i] = 1;}

      if (i < numsites-1) {outm << M[i] << " ";}
      else                {outm << M[i] << endl;}
    }
        
    for (int t = 0; t < numwms; t++) {
      for (int i = padlen; i < lenseq-padlen-1; i++) {
	out[t] << P[t][i] << " ";
      }
      out[t] << P[t][lenseq-padlen-1] << endl;
    }
  }
  
  // Clossing files 
  for (int t = 0; t < numwms; t++) {out[t].close();}
  outm.close();
}


// Initialization of parameters from file 
// or with default values
// ------------------------------------------------------
double* init_params(int numwms, char *filepar) { 
  double *tparams = new double[numwms+3];
  double params[numwms+3];
  
  // Defult parameters 
  if (!filepar) {
      for (int p = 0; p < numwms+1; p++) {tparams[p] = 0.0;};
      tparams[numwms+1] = log(0.05/(1-0.05));
      tparams[numwms+2] = log(0.95/(1-0.95)); 
  }

  // Reading parameters from file
  else {
  
    ifstream in(filepar);
    if(!in) {
      cerr << "ERROR: file could not be opened" << endl;
      exit(1);
    }

    string line;
    string parstr;
    while (getline(in,line)) {parstr = line;};

    int p = 0;
    stringstream ss(parstr);
    ss >> parstr;
    ss >> parstr;
    while (ss >> parstr) {params[p++] = stod(parstr);}
    para_transformation(numwms,params,tparams);
  }

  return(tparams);
}


// Fitting parameters MCMC
// ------------------------------------------------------
void get_params_SGD(dataseq seq, datamet met, datawms *wms, double *params, typeparsgd parsgd,char *file) {
  int numwms = seq.numwms;
  int *mapwm = seq.mapwm;
  int nummol = met.nummol;
  double dlogP[numwms+3];
  double logP = 0;

  double beta1 = parsgd.beta1;
  double beta2 = parsgd.beta2;
  double lrate = parsgd.lrate;
  int numepochs = parsgd.numepochs;
  int batchsize = parsgd.batchsize;
  int numbatches = ceil(nummol/batchsize);
  int numoutput = ceil(numepochs/10.0);
  int fitgamma = parsgd.fitgamma;

  // Open file 
  ofstream out;
  string outtag(file);
  string fileout = outtag + ".params_SGD.txt";
  out.open(fileout.c_str());

  // Printing heading 
  string head = "# loglikelihood ";
  for (int t = 0; t < numwms; t++) {head += wms[mapwm[t]].name + " ";}
  head += "beta qU qB";
  out << head << endl;
  
  // Random number generator 
  default_random_engine generator((unsigned) clock());
  normal_distribution<double> normal(0.0,1.0);
  uniform_real_distribution<double> uniform(0.0,1.0);
  
  // Initialization ADAM
  int t = 0;
  double mom[numwms+3];
  double vec[numwms+3];
  double momnew[numwms+3];
  double vecnew[numwms+3];
  double momhat[numwms+3];
  double vechat[numwms+3];
  double outparams[numwms+3];
  for (int p = 0; p < numwms+3; p++) {
    mom[p] = 0.0;
    vec[p]= 0.0;
  }

  //double NdlogP[numwms+3];
  //double AdlogP[numwms+3];
  // Stochatic Gradient Descent (ADAM) 
  for (int n = 0; n < numepochs; n++) {
    
    // Creating a permutation of molecules
    vector<int> permol = random_permutation(nummol);


    for (int b = 0; b < numbatches; b++) {

      // Selecting molecules
      int m0 = b * batchsize;
      int mf = (b+1) * batchsize;
      if (mf > nummol) {mf = nummol;}
      vector<int> selmol(permol.begin() + m0, permol.begin() + mf);
	   
      // Calculating gradient 
      logP = loglikelihood_grad(seq,met,params,selmol,dlogP);
      if (fitgamma == 0) {dlogP[numwms]=0.0;}

      //cout << b << " " << logP << " ";
      //for (int p = 0; p < numwms+3; p++) {
      //cout << dlogP[p] << " ";
      //}
      //cout << endl;
      
      /*
      clock_t t0 = clock();
      double AlogP = loglikelihood_semigrad(seq,met,params,selmol,AdlogP);
      clock_t t1 = clock();
      double NlogP = loglikelihood_numgrad(seq,met,params,selmol,NdlogP);
      clock_t t2 = clock();
      double sum = 0;
      cerr << AlogP << " " << NlogP << endl;
      for (int p = 0; p < numwms+3; p++) {
	cerr << p << " " << params[p] << " " << AdlogP[p] << " " << NdlogP[p] << " " << abs(NdlogP[p]-AdlogP[p])<< endl; 
	sum += abs(NdlogP[p]-AdlogP[p]);
      }
      clock_t tf = clock();
      double dt1 = 1.0 * (t1 - t0) / CLOCKS_PER_SEC;
      double dt2 = 1.0 * (t2 - t1) / CLOCKS_PER_SEC;
      cerr << "TIME: " << setprecision(10) << dt1 << " " << dt2 << " " << sum << endl << endl;
      //exit(1);
      */
      
      // Updating paramters
      t = t + 1;
      for (int p = 0; p < numwms+3; p++) {
	momnew[p] = beta1*mom[p] + (1-beta1)*dlogP[p];
	vecnew[p] = beta2*vec[p] + (1-beta2)*pow(dlogP[p],2);
	momhat[p] = momnew[p]/(1-pow(beta1,t));
	vechat[p] = vecnew[p]/(1-pow(beta2,t));
	params[p] = params[p] + lrate*momhat[p]/(sqrt(vechat[p])+10e-8);
	mom[p] = momnew[p];
	vec[p] = vecnew[p];
      }
    }

    // Calculating the probability of the data
    logP = loglikelihood(seq,met,params);

    // Output and printing results
    para_invtransformation(numwms, params, outparams);
    out << n << " " << logP << " ";
    for (int p = 0; p < numwms+3; p++) {
      if (p <  numwms+2) {out << outparams[p] << " ";}
      else               {out << outparams[p] << endl;}
    }
    
    if (n % numoutput == 0) {cout << "RUN: fitting SGD: " << n << "/" << numepochs << " " << logP << endl;}
  }
  
  out.close();
}


// Fitting parameters MCMC
// ------------------------------------------------------
void get_params_MCMC(dataseq seq, datamet met, datawms *wms, double *params, typeparsam parsam, char *file) {
  int numwms = seq.numwms;
  int *mapwm = seq.mapwm;
  int numiter = parsam.numiter;
  int numoutput = ceil(numiter/10.0);
  int fitgamma = parsam.fitgamma;
  double sigma = parsam.sigma;;
  double a = 0.0;

  // Open files
  ofstream out;
  string outtag(file);
  string fileout = outtag + ".params_MCMC.txt";
  out.open(fileout.c_str());

  // Printing heading 
  string head = "# loglikelihood ";
  for (int t = 0; t < numwms; t++) {head += wms[mapwm[t]].name + " ";}
  head += "beta qU qB";
  out << head << endl;
  
  // Random number generator 
  default_random_engine generator((unsigned) clock());
  normal_distribution<double> normal(0.0,1.0);
  uniform_real_distribution<double> uniform(0.0,1.0);
  
  // Initialize parameters
  double newparams[numwms+3];
  double outparams[numwms+3];
  newparams[0] = 0.0;
  
  // Calculating probability
  double logP = loglikelihood(seq,met,params);
  double newlogP;

  //MCMC
  for (int n = 0; n < numiter; n++) {
    //if (n == 2000) {sigma = 0.01;}
    
    // Calculating new parameters 
    for (int p = 1; p < numwms+3; p++) {
      newparams[p] = params[p] + sigma*normal(generator);
    }
    if (fitgamma ==0) {newparams[numwms] = params[numwms];}
      
    // Calculating new probability
    newlogP = loglikelihood(seq,met,newparams);

    // Updating
    if (exp(newlogP-logP) > uniform(generator)) {
      for (int p = 0; p < numwms+3; p++) {params[p] = newparams[p];}
      logP = newlogP;
      a++; 
    }
    
    // Printoug
    para_invtransformation(numwms, params, outparams);
    out << n << " " << logP << " ";
    for (int p = 0; p < numwms+3; p++) {
      if (p <  numwms+2) {out << outparams[p] << " ";}
      else               {out << outparams[p] << endl;}
    }
    if (n % numoutput == 0) {cout << "RUN: fitting MCMC: " << n << "/" << numiter << " " << logP << " " << ceil(10000*a/(n+1))/100 << "%" << endl;}
  }

  out.close();
}


// Transforming parameters
// ------------------------------------------------------
void para_invtransformation(int numwms, double *tparams, double *params) {
 
  for (int p = 0; p <= numwms; p++) {
    params[p] = exp(tparams[p]);
  }
  params[numwms+1] = 1/(1+exp(-tparams[numwms+1]));
  params[numwms+2] = 1/(1+exp(-tparams[numwms+2]));
}


// Transforming parameters
// ------------------------------------------------------
void para_transformation(int numwms, double *params, double *tparams) {
 
  for (int p = 0; p <= numwms; p++) {
    tparams[p] = log(params[p]);
  }
  tparams[numwms+1] = log(params[numwms+1]/(1-params[numwms+1]));
  tparams[numwms+2] = log(params[numwms+2]/(1-params[numwms+2]));
}


// Caculating probability of the data
// ------------------------------------------------------
double loglikelihood_numgrad(dataseq seq, datamet met, double *tparams, vector<int> selmol, double *dlogP) {
  int lenseq = seq.lenseq;
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int nummol = selmol.size();
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int **U = met.U;
  int **M = met.M;
  double *probW = seq.probW;
  double prob = 0.0;
  double h = 1e-8;
  
  double Q[lenseq+1];
  double dZ[numwms+3];
  double dF[numwms+3];
  double probM[numelm];
  double probB[numelm];
  double params[numwms+3];
  double mytparams[numwms+3];
  double Z,F,Zh,Fh;
  for (int p = 0; p < numwms+3; p++) {dlogP[p] = 0;}

  // Transforming parameters
  para_invtransformation(numwms,tparams,params);
  
  // Calculating probB
  get_probB(numelm,ind,probW,numwms,params,probB);
  
  // Calculating partition function
  forward_conf(numelm, lenseq, pos, len, probB, Q);
  Z = Q[lenseq];
  
  for (int p = 0; p < numwms+3; p++) {

    // Adding small h to one parmeter at the time
    for (int q = 0; q < numwms+3; q++) {
      if (q == p) {mytparams[q] = tparams[q] + h;}
      else        {mytparams[q] = tparams[q];}
    }
    
    // Transforming parameters
    para_invtransformation(numwms,mytparams,params); 
  
    // Calculating probB
    get_probB(numelm,ind,probW,numwms,params,probB);
    
    // Calculating partition function
    forward_conf(numelm, lenseq, pos, len, probB, Q);
    Zh = Q[lenseq];
    dZ[p] = (Zh - Z)/h;
  }
    
  // Calculating biinding profiles
  double logP = 0.0;
  for (int n = 0; n < nummol; n++) {
    int m = selmol[n];

    // Transforming parameters
    para_invtransformation(numwms,tparams,params);
    
    // Calculating probabM
    get_probM(U[m], M[m], numelm, bnd, numwms, params, probM);
    
    // Calculating forward vector
    forward_data(numelm, lenseq, pos, len, probM, probB, Q);
    F = Q[lenseq];
    
    // Updating probability of data
    logP += log(F)-log(Z);
    
    for (int p = 0; p < numwms+3; p++) {
      
      // Adding small h to one parmeter at the time
      for (int q = 0; q < numwms+3; q++) {
	if (q == p) {mytparams[q] = tparams[q] + h;}
	else        {mytparams[q] = tparams[q];}
      }
      
      // Transforming parameters
      para_invtransformation(numwms,mytparams,params);

      // Calculating probB
      get_probB(numelm,ind,probW,numwms,params,probB);
      
      // Calculating probabM
      get_probM(U[m], M[m], numelm, bnd, numwms, params, probM);
      
      // Calculating forward vector
      forward_data(numelm, lenseq, pos, len, probM, probB, Q);
      Fh = Q[lenseq];

      dF[p] = (Fh-F)/h;
      //cerr << p << " " << Fh << " " << F << " " << dF[p] << endl;
    }
    
    // Updating gradient
    for (int p = 1; p < numwms+3; p++) {
      dlogP[p] += dF[p]/F - dZ[p]/Z;
      //fprintf(stderr,"N: %i %i %.5e %.5e %.5e %.5e %.5e\n",m,p,F,Z,dF[p],dZ[p],dF[p]/F - dZ[p]/Z);
      //cerr << "N: " << m << " " << p << " " << F << " " << Z << " " << dF[p] << " " << dZ[p] << " " << dF[p]/F - dZ[p]/Z << endl; 
    }
    dlogP[numwms] = 0.0;
  }

  return(logP);
}


// Caculating probability of the data
// ------------------------------------------------------
double loglikelihood_grad(dataseq seq, datamet met, double *tparams, vector<int> selmol, double *dlogP) {
  int lenseq = seq.lenseq;
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int nummol = selmol.size();
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int **U = met.U;
  int **M = met.M;
  double *probW = seq.probW;
  double prob = 0.0;
  
  double dZ[numwms+3];
  double dF[numwms+3];
  double probM[numelm];
  double probB[numelm];
  double dprobB[numelm];
  double dprobMdU[numelm];
  double dprobMdB[numelm];
  double params[numwms+3];
  double Z,F;
  for (int p = 0; p < numwms+3; p++) {dlogP[p] = 0;}
  
  // Transforming parameters
  para_invtransformation(numwms,tparams,params); 
  
  // Calculating probB
  get_probB_grad(numelm,ind,probW,numwms,params,probB,dprobB);

  // Calculating partition function
  forward_conf_grad(numelm, lenseq, numwms, pos, len, ind, probB, dprobB, &Z, dZ);
  
  // Calculating binding profiles
  double logP = 0.0;

  #ifdef USE_OMP
  #pragma omp parallel for num_threads(numthreads) private(probM, dprobMdU, dprobMdB, F, dF) reduction(+:logP) reduction(+:dlogP[:numwms+3])
  #endif
  
  for (int n = 0; n < nummol; n++) {
    int m = selmol[n];

    // Calculating probabM
    get_probM_grad(U[m], M[m], numelm, bnd, numwms, params, probM, dprobMdU, dprobMdB);
        
    // Calculating forward vector
    forward_data_grad(numelm, lenseq, numwms, pos, len, ind, probM, dprobMdU, dprobMdB, probB, dprobB, &F, dF);
  
    // Updating probability of data
    logP += log(F)-log(Z);
    //int ID = omp_get_thread_num();
    //double logF = log(F);
    //printf("num: %i log(F): %f logP: %f\n",ID,logF,logP);

    // Updating gradient
    for (int p = 1; p < numwms+3; p++) {
    //for (int p = 1; p < numwms; p++) {
      dlogP[p] += dF[p]/F - dZ[p]/Z;
      //printf("num: %i par: %i dlogP: %f\n",ID,p,dlogP[p]);
      //fprintf(stderr, "A: %i %i %.5e %.5e %.5e %.5e %.5e\n",m,p,F,Z,dF[p],dZ[p],dF[p]/F - dZ[p]/Z); 
      //cerr << "A: " << m << " " << p << " " << F << " " << Z << " " << dF[p] << " " << dZ[p] << " " << dF[p]/F - dZ[p]/Z << endl; 
    }
    //dlogP[numwms] = 0.0;
  }

  // Gamma prior for conentations
  //for (int p = 1; p < numwms; p++) {
  //if (wms[mapwm[p]].name == "PolII") {
  logP += (priork-1)*log(params[2]) - params[2]*priorb;
  dlogP[2] += (priork-1)/params[2] - priorb;
  //}
  //}
  
  return(logP);
}


// Caculating probability of the data
// ------------------------------------------------------
double loglikelihood_semigrad(dataseq seq, datamet met, double *tparams, vector<int> selmol, double *dlogP) {
  int lenseq = seq.lenseq;
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int nummol = selmol.size();
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int **U = met.U;
  int **M = met.M;
  double *probW = seq.probW;
  double prob = 0.0;
  double h = 1e-8;
  
  double Q[lenseq+1];
  double dZ[numwms+3];
  double dF[numwms+3];
  double probM[numelm];
  double probB[numelm];
  double dprobB[numelm];
  double dprobMdU[numelm];
  double dprobMdB[numelm];
  double params[numwms+3];
  double mytparams[numwms+3];
  double Z,F,Zh;
  for (int p = 0; p < numwms+3; p++) {dlogP[p] = 0;}

  // Transforming parameters
  para_invtransformation(numwms,tparams,params); 
  
  // Calculating probB
  get_probB_grad(numelm,ind,probW,numwms,params,probB,dprobB);

  // Calculating partition function
  forward_conf(numelm, lenseq, pos, len, probB, Q);
  Z = Q[lenseq];
  
  for (int p = 0; p < numwms+3; p++) {

    // Adding small h to one parmeter at the time
    for (int q = 0; q < numwms+3; q++) {
      if (q == p) {mytparams[q] = tparams[q] + h;}
      else        {mytparams[q] = tparams[q];}
    }
    
    // Transforming parameters
    para_invtransformation(numwms,mytparams,params); 
  
    // Calculating probB
    get_probB_grad(numelm,ind,probW,numwms,params,probB,dprobB);
    
    // Calculating partition function
    forward_conf(numelm, lenseq, pos, len, probB, Q);
    Zh = Q[lenseq];
    dZ[p] = (Zh - Z)/h;
  }  
    
  // Calculating binding profiles
  double logP = 0.0;
  for (int n = 0; n < nummol; n++) {
    int m = selmol[n];
    
    // Calculating probabM
    get_probM_grad(U[m], M[m], numelm, bnd, numwms, params, probM, dprobMdU, dprobMdB);
        
    // Calculating forward vector
    forward_data_grad(numelm, lenseq, numwms, pos, len, ind, probM, dprobMdU, dprobMdB, probB, dprobB, &F, dF);
  
    // Updating probability of data
    logP += log(F)-log(Z);

    // Updating gradient
    for (int p = 1; p < numwms+3; p++) {
      dlogP[p] += dF[p]/F - dZ[p]/Z;
      //fprintf(stderr, "A: %i %i %.5e %.5e %.5e %.5e %.5e\n",m,p,F,Z,dF[p],dZ[p],dF[p]/F - dZ[p]/Z); 
      //cerr << "A: " << m << " " << p << " " << F << " " << Z << " " << dF[p] << " " << dZ[p] << " " << dF[p]/F - dZ[p]/Z << endl; 
    }
    dlogP[numwms] = 0.0;
  }

  return(logP);
}


// Caculating probability of the data
// ------------------------------------------------------
double loglikelihood(dataseq seq, datamet met, double *tparams) {
  int lenseq = seq.lenseq;
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int nummol = met.nummol;
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int **U = met.U;
  int **M = met.M;
  double *probW = seq.probW;
  double prob = 0.0;
  
  double F[lenseq+1];
  double B[lenseq+1];
  double Q[lenseq+1];
  double probM[numelm];
  double probB[numelm];
  double params[numwms+3];
    
  // Transforming parameters
  para_invtransformation(numwms,tparams,params); 
  
  // Calculating probB
  get_probB(numelm,ind,probW,numwms,params,probB);

  // Calculating partition function
  forward_conf(numelm, lenseq, pos, len, probB, Q);
  double Z = Q[lenseq];
  
  // Calculating binding profiles
  double logP = 0.0;

  #ifdef USE_OMP
  #pragma omp parallel for num_threads(numthreads) private(probM, F) reduction(+:logP)
  #endif
  
  for (int m = 0; m < nummol; m++) {

    // Calculating probabM
    get_probM(U[m], M[m], numelm, bnd, numwms, params, probM);
        
    // Calculating forward vector
    forward_data(numelm, lenseq, pos, len, probM, probB, F);

    // Updating probability of data
    logP += log(F[lenseq])-log(Z);
  }

  // Gamma prior
  //for (int p = 1; p < numwms; p++) {
  //if (wms[mapwm[p]].name == "PolII") {
  logP += (priork-1)*log(params[2]) - params[2]*priorb;
  //}
  //}
  // prior
  //double priorU = (1/(1+exp(1000*(params[numwms+1]-0.1))));//*params[numwms+1]*(1-params[numwms+1]); 
  //double priorB = (1/(1+exp(-1000*(params[numwms+2]-0.9))));//params[numwms+2]*(1-params[numwms+2]);
  //if (params[numwms+1] > 0.05) {logP = -1000000;}
  //if (params[numwms+2] < 0.95) {logP = -1000000;}
  //logP += 1000*log(priorU) + 1000*log(priorB);
  
  return(logP);
}


// Comlculating probability of data F summing
// over all configurations using forward algorithm
// ------------------------------------------------------
void get_binding(dataseq seq, datamet met, datawms *wms, double *tparams, char *file) {
  int lenseq = seq.lenseq;
  int numelm = seq.numelm;
  int numwms = seq.numwms;
  int padlen = seq.padlen;
  int nummol = met.nummol;
  int *len = seq.len;
  int *pos = seq.pos;
  int *ind = seq.ind;
  int *bnd = seq.bnd;
  int *mapwm = seq.mapwm;
  int **U = met.U;
  int **M = met.M;
  double *probW = seq.probW;
  double prob = 0.0;
  
  double P[numwms][lenseq];
  double S[numwms][lenseq]; 
  double F[lenseq+1];
  double B[lenseq+1];
  double probM[numelm];
  double probB[numelm];
  double params[numwms+3];

  // Output
  cout << "RUN: Printing binding profiles" << endl; 

  // Open files
  string outtag(file);
  ofstream out1[numwms];
  ofstream out2[numwms];
  for (int t = 0; t < numwms; t++) {
    string fileout1 = outtag + ".hfprofile." +  wms[mapwm[t]].name + ".txt";
    string fileout2 = outtag + ".hfbinding." +  wms[mapwm[t]].name + ".txt";
    out1[t].open(fileout1.c_str());
    out2[t].open(fileout2.c_str());
  }

  // Transforming parameters
  para_invtransformation(numwms,tparams,params);

    // Calculating probB
  get_probB(numelm,ind,probW,numwms,params,probB);
      
  // Calculating biinding profiles 
  for (int m = 0; m < nummol; m++) {

    // Calculating probabM
    get_probM(U[m], M[m], numelm, bnd, numwms, params, probM);

    //for (int n = 0; n < numelm; n++) {
    //cerr << m << " " << pos[n] << " " << ind[n] << " " << len[n] << " " << U[m][n] << " " <<  M[m][n] << " " << C[m][n] << " " << probB[n] << " " << probM[n] << endl;
    //}
    //exit(1);
    
    // Calculating forward vector
    forward_data(numelm, lenseq, pos, len, probM, probB, F);

    // Calculating backward vector
    backward_data(numelm, lenseq, pos, len, probM, probB, B);

    // Initializing
    for (int t = 0; t < numwms; t++) {
      for (int i = 0; i < lenseq; i++) {
	P[t][i] = 0.0;
	S[t][i] = 0.0;
      }
    }

    // Calculating binding probabilites 
    for (int n = 0; n < numelm; n++) {
      int i = pos[n];
      int l = len[n];
      int t = ind[n];

      prob = exp(log(F[i]) + log(probM[n]) + log(probB[n]) + log(B[i+l]) - log(F[lenseq]));

      S[t][i] = prob;
      for (int s = i; s < i+l; s++) {
	P[t][s] += prob;
      }
    }
    
    // Printing binding probabilites into files (sparse format if outcutoff > 0)
    for (int t = 0; t < numwms; t++) {
      if (t < 2 || outcutoff == 0) {
	for (int i = padlen; i < lenseq-padlen-1; i++) {
	  out1[t] << P[t][i] << " ";
	  out2[t] << S[t][i] << " ";
	}
	out1[t] << P[t][lenseq-padlen-1] << endl;
	out2[t] << S[t][lenseq-padlen-1] << endl;
      }
      else {
	for (int i = padlen; i < lenseq-padlen; i++) {
	  if (P[t][i] >= outcutoff) {out1[t] << m << " " << i << " " << P[t][i] << endl;}
	  if (S[t][i] >= outcutoff) {out2[t] << m << " " << i << " " << S[t][i] << endl;}
	}
      }
    }
  }
  
  // Clossing files 
  for (int t = 0; t < numwms; t++) {
    out1[t].close();
    out2[t].close();
  }
}


// Comlculating probability of data F and gradient
// ------------------------------------------------------
void forward_data_grad(int numelm, int lenseq, int numwms, int *pos, int *len, int *ind, double *probM, double *dprobMdU, double *dprobMdB, double *probB, double *dprobB, double *P, double *G) {

  // Initialization
  double F[lenseq+1];
  double dFdp[lenseq+1][numwms+3];
  
  F[0] = 1.0;
  for (int i = 1; i <= lenseq; i++) {
    F[i] = 0.0;
    for (int p = 0; p < numwms+3; p++) {
      dFdp[i][p] = 0.0;
    }
  }
  
  // Calculating prob of data and gradient 
  for (int n = 0; n < numelm; n++) {
    int i = pos[n];
    int l = len[n];
    int t = ind[n];
    
    F[i+l] += probM[n]*probB[n]*F[i];

    for (int r = 0; r < numwms; r++) {

      if (r == t) {
	dFdp[i+l][r] += probM[n]*probB[n]*dFdp[i][r] + probM[n]*probB[n]*F[i];
      }
      else {
	dFdp[i+l][r] += probM[n]*probB[n]*dFdp[i][r];
      }
    }
    
    dFdp[i+l][numwms]   += probM[n]*probB[n]*dFdp[i][numwms]   +   probM[n]*dprobB[n]*F[i];
    dFdp[i+l][numwms+1] += probM[n]*probB[n]*dFdp[i][numwms+1] + dprobMdU[n]*probB[n]*F[i];
    dFdp[i+l][numwms+2] += probM[n]*probB[n]*dFdp[i][numwms+2] + dprobMdB[n]*probB[n]*F[i];
  }

  // Return relavant values 
  *P = F[lenseq];
  for (int p = 0; p < numwms+3; p++) {
    G[p] = dFdp[lenseq][p];
  }
}


// Comlculating probability of data F and gradient
// ------------------------------------------------------
void forward_conf_grad(int numelm, int lenseq, int numwms, int *pos, int *len, int *ind, double *probB, double *dprobB, double *P, double *G) {

  // Initialization
  double Z[lenseq+1];
  double dZdp[lenseq+1][numwms+3];
  
  Z[0] = 1.0;
  for (int i = 1; i <= lenseq; i++) {
    Z[i] = 0.0;
    for (int p = 0; p < numwms+3; p++) {
      dZdp[i][p] = 0.0;
    }
  }

  // Calculating prob of data and gradient 
  for (int n = 0; n < numelm; n++) {
    int i = pos[n];
    int l = len[n];
    int t = ind[n];
    
    Z[i+l] += probB[n]*Z[i];

    for (int r = 0; r < numwms; r++) {

      if (r == t) {
	dZdp[i+l][r] += probB[n]*dZdp[i][r] + probB[n]*Z[i];
      }
      else {
	dZdp[i+l][r] += probB[n]*dZdp[i][r];
      }
    }
    
    dZdp[i+l][numwms] += probB[n]*dZdp[i][numwms] + dprobB[n]*Z[i];
  }
  
  // Return relavant values 
  *P = Z[lenseq];
  for (int p = 0; p < numwms+3; p++) {
    G[p] = dZdp[lenseq][p];
  }
  //G[numwms+1] = 0.0;
  //G[numwms+2] = 0.0;
}


// Comlculating probability of data F summing
// over all configurations using forward algorithm
// ------------------------------------------------------
void backward_data(int numelm, int lenseq, int *pos, int *len, double *probM, double *probB, double *B) {
    
  B[lenseq] = 1.0;
  for (int i = 0; i < lenseq; i++) {
    B[i] = 0.0;
  }

  for (int n = numelm-1; n >= 0; n--) {
    int i= pos[n];
    int l = len[n];
    B[i] += probM[n]*probB[n]*B[i+l];
  }
}


// Comlculating partition function Z
// ------------------------------------------------------
void backward_conf(int numelm, int lenseq, int *pos, int *len, double *probB, double *Z) {
  
  Z[lenseq] = 1.0;
  for (int i = 0; i < lenseq; i++) {
    Z[i] = 0.0;
  }

  for (int n = numelm-1; n >= 0; n--) {
    int i = pos[n];
    int l = len[n];
    Z[i] += probB[n]*Z[i+l];
  }
}


// Comlculating probability of data F summing
// over all configurations using forward algorithm
// ------------------------------------------------------
void forward_data(int numelm, int lenseq, int *pos, int *len, double *probM, double *probB, double *F) {

  F[0] = 1.0;
  for (int i = 1; i <= lenseq; i++) {
    F[i] = 0.0;
  }
  
  for (int n = 0; n < numelm; n++) {
    int i = pos[n];
    int l = len[n];

    F[i+l] += probM[n]*probB[n]*F[i];
  }
}


// Comlculating partition function Z
// ------------------------------------------------------
void forward_conf(int numelm, int lenseq, int *pos, int *len, double *probB, double *Z) {
    
  Z[0] = 1.0;
  for (int i = 1; i <= lenseq; i++) {
    Z[i] = 0.0;
  }
  
  for (int n = 0; n < numelm; n++) {
    int i = pos[n];
    int l = len[n];
    Z[i+l] += probB[n]*Z[i];
  }
}


// Comlculating probability of binding
// for given concentratiions and exponents
// ------------------------------------------------------
void get_probB_grad(int numelm, int *ind, double *probW, int numwms, double *params, double *probB, double *dprobB) {
  
  for (int n = 0; n < numelm; n++) {
    int t = ind[n];
    probB[n] = params[t]*pow(probW[n],params[numwms]);
    dprobB[n] = params[numwms]*log(probW[n])*probB[n];
  }
}


// Comlculating probability of binding
// for given concentratiions and exponents
// ------------------------------------------------------
void get_probB(int numelm, int *ind, double *probW, int numwms, double *params, double *probB) {
  
  for (int n = 0; n < numelm; n++) {
    int t = ind[n];
    probB[n] = params[t]*pow(probW[n],params[numwms]);
  }
}


// Comlculating probability of the methylation data
// given the bound/unbound state of the methylation sites 
// ------------------------------------------------------
void get_probM_grad(int *U, int *M, int numelm, int* bnd, int numwms, double *params, double *probM, double *dprobMdU, double *dprobMdB) {
  double qU = params[numwms+1];
  double qB = params[numwms+2];
  double q[2] = {log(qU),log(qB)};
  double p[2] = {log(1-qU),log(1-qB)};
  
  for (int n = 0; n < numelm; n++) {
    double myq = q[bnd[n]];
    double myp = p[bnd[n]];
    
    probM[n] = exp(U[n]*myq + M[n]*myp);
    
    if (bnd[n] == 0) {
      dprobMdU[n] = probM[n]*(U[n]*(1-qU) - M[n]*qU);
      dprobMdB[n] = 0.0;
    }
    else {
      dprobMdU[n] = 0.0;
      dprobMdB[n] = probM[n]*(U[n]*(1-qB) - M[n]*qB);
    }
  }
  /*
  double h = 1e-6;
  qU = params[numwms+1];
  qB = 1/(1+exp(-log(params[numwms+2]/(1-params[numwms+2])) - h)); 
  double p2[2] = {log(qU),log(qB)};
  double q2[2] = {log(1-qU),log(1-qB)};
  for (int n = 0; n < numelm; n++) {
    double myp = q2[bnd[n]];
    double myq = p2[bnd[n]];
    //dprobMdB[n] = (exp(C[n] + U[n]*myp + M[n]*myq)-probM[n])/h;  
    double dprobM = (exp(C[n] + U[n]*myp + M[n]*myq)-probM[n])/h;  
    cerr << "1: " << n << " " << dprobM << " " << dprobMdB[n] << endl;
  }

  qU = 1/(1+exp(-log(params[numwms+1]/(1-params[numwms+1])) - h)); 
  qB = params[numwms+2]; 
  double p3[2] = {log(qU),log(qB)};
  double q3[2] = {log(1-qU),log(1-qB)};
  for (int n = 0; n < numelm; n++) {
    double myp = q3[bnd[n]];
    double myq = p3[bnd[n]];
    //dprobMdU[n] = (exp(C[n] + U[n]*myp + M[n]*myq)-probM[n])/h;
    double dprobM = (exp(C[n] + U[n]*myp + M[n]*myq)-probM[n])/h;  
    cerr << "2: " << n << " " << dprobM << " " << dprobMdU[n] << endl;
  }
  exit(1);
  */
}


// Comlculating probability of the methylation data
// given the bound/unbound state of the methylation sites 
// ------------------------------------------------------
void get_probM(int *U, int *M, int numelm, int* bnd, int numwms, double *params, double *probM) {
  double qU = params[numwms+1];
  double qB = params[numwms+2];
  double q[2] = {log(qU),log(qB)};
  double p[2] = {log(1-qU),log(1-qB)};
  
  for (int n = 0; n < numelm; n++) {
    double myq = q[bnd[n]];
    double myp = p[bnd[n]];
      
    probM[n] = exp(U[n]*myq + M[n]*myp);
  }
}


// Computing the methylation counts for different windows
// ------------------------------------------------------
void get_methcounts(dataseq seq, datamet *met) {
  int nummol = met -> nummol;
  int numelm = seq.numelm;
  
  // Initiailzation
  met -> U = new int*[nummol];
  met -> M = new int*[nummol];
  for (int m = 0; m < nummol; m++) {
    met -> U[m] = new int[numelm];
    met -> M[m] = new int[numelm];
  }
    
  // Calculating methylation matrices U and M
  for (int m = 0; m < nummol; m++) {
    int s0 = 0;
    int sf = 0;
    
    for (int n = 0; n < numelm; n++) {
      int i = seq.pos[n];
      int l = seq.len[n];
      
      met -> U[m][n] = 0;
      met -> M[m][n] = 0;
      while (met -> msites[s0] < i) {s0++;}
      
      sf = s0;
      while (sf < met -> numsites && met -> msites[sf] < i+l) {
	if      (met -> mstate[m][sf] == 0) {met -> U[m][n]++;}
	else if (met -> mstate[m][sf] == 1) {met -> M[m][n]++;}
	sf++;
      }
    }
  }
}


// Selecting WMs with a good BS in each position i
// ------------------------------------------------------
void select_wms(dataseq *seq, datawms *wms, typeparwms parwms, char *filebnd, char *file) {
  double **ProbS;
  double cutoffwms = parwms.cutoffwms;
  double bgprob = parwms.bgprob;
  int maxwms = parwms.maxwms;
  int lenseq = seq -> lenseq;
  int padlen = seq -> padlen;
  int *sequence = seq -> seq;

  int invec[maxwms];
  int mapwm[maxwms];
  vector <int> myind;
  vector <int> mypos;
  vector <int> mylen;
  vector <int> mybnd;
  int numelm = 0;

  cout << "RUN: Selecting WMs: ";
    
  // Initialitzation
  double **probS = new double*[lenseq];
  for (int i = 0; i < lenseq; i++) {probS[i] = new double[maxwms];}
  for (int t = 0; t < maxwms; t++) {invec[t]=-1;};

  // Calculate probability of BSs
  get_probS(lenseq, sequence, maxwms, wms, probS);

  // Selecting wm that can bind in each position i
  int numwms = 0;
  if (!filebnd) {

    // Calculating from scracht 
    for (int i = 0; i < lenseq; i++) {
      for (int t = 0; t < maxwms; t++) {
	if (probS[i][t] >= cutoffwms) {
	  if (invec[t] == -1) {
	    invec[t] = numwms;
	    mapwm[numwms] = t;
	    numwms++;
	  }
	
	  mypos.push_back(i);	
	  myind.push_back(invec[t]);
	  mylen.push_back(wms[t].len);
	  mybnd.push_back(funbnd(t));
	  numelm++;
	}
      }
    }
  }
  else {

    // Reading from file
    ifstream in(filebnd);
    if(!in) {
      cerr << "ERROR: file could not be opened" << endl;
      exit(1);
    }

    string strpos,strfac,strlen,strprob,strname;
    while (in >> strpos >> strfac >> strlen >> strprob >> strname) {
      int t = stoi(strfac);
      if (invec[t] == -1) {
	invec[t] = numwms;
	mapwm[numwms] = t;
	numwms++;
      }
      mypos.push_back(stoi(strpos)+padlen);
      myind.push_back(invec[t]);
      mylen.push_back(stoi(strlen));
      mybnd.push_back(funbnd(t));
      numelm++;
    }
  }

  // Storing and prining results in structure seq
  string outtag1(file);
  string fileout1 = outtag1 + ".states.txt";
  ofstream out1(fileout1.c_str());
  seq -> numelm = numelm;
  seq -> numwms = numwms;
  seq -> ind = new int[numelm];
  seq -> len = new int[numelm];
  seq -> pos = new int[numelm];
  seq -> bnd = new int[numelm];
  seq -> mapwm = new int[numwms];
  seq -> probW = new double[numelm];

  for (int t = 0; t < numwms; t++) {
    seq -> mapwm[t] = mapwm[t];
  }
  
  for (int n = 0; n < numelm; n++)  {
    seq -> ind[n] = myind[n];
    seq -> pos[n] = mypos[n];
    seq -> len[n] = mylen[n];
    seq -> bnd[n] = mybnd[n];
    int pwmid = mapwm[myind[n]];
    
    out1 << mypos[n] - padlen << " " << pwmid << " " << mylen[n] << " " << probS[mypos[n]][pwmid] << " " << wms[pwmid].name << endl;
  }
  out1.close();

  // Printing out the results 
  string outtag2(file);
  string fileout2 = outtag2 + ".probS.txt";
  ofstream out2(fileout2.c_str());

  for (int t = 0; t < numwms-1; t++) {
    out2 << wms[mapwm[t]].name << " ";
  }
  out2 << wms[mapwm[numwms-1]].name << endl;
  
  for (int i = padlen; i < lenseq-padlen; i++) {
    for (int t = 0; t < numwms-1; t++) {
       out2 << probS[i][mapwm[t]] << " ";
    }
    out2 << probS[i][mapwm[numwms-1]] << endl;
  }
  out2.close();

  // Delate variable ProbS
  for (int i = 0; i < lenseq; i++) {delete[] probS[i];}
    
  // get probW
  get_probW(seq, wms, bgprob);

  // Output
  cout << numwms << " selected motifs, " << numelm << " elements" << endl;  
}


// Comlculating bolttzman weights form the PWs 
// ------------------------------------------------------
void get_probW(dataseq *seq, datawms *wms, double bgprob) {
  int numelm = seq -> numelm;
  int *sequence = seq -> seq;
  
  for (int n = 0; n < numelm; n++) {
    int i = seq -> pos[n];
    int l = seq -> len[n];
    int t = seq -> ind[n];
    int r = seq -> mapwm[t];
    datawms myWM = wms[r];

    double myprob = 0;
    double myprobR = 0;
    for (int s = 0; s < l; s++) {
      myprob  += myWM.W[s][sequence[i+s]]-log(bgprob);
      myprobR += myWM.rW[s][sequence[i+s]]-log(bgprob);
    } 
    //seq -> probW[n] = 0.5*exp(myprob/l) + 0.5*exp(myprobR/l);
    seq -> probW[n] = 0.5*exp(myprob) + 0.5*exp(myprobR);
  }
}


// Calculating probability of binsing sites 
// ------------------------------------------------------
void get_probS(int lenseq, int *sequence, int maxwms, datawms *wms, double **probS) {

  int i0 = 0;
    
  // Calculating probabiliity of 
  for (int t = 0; t < maxwms; t++) {    
    datawms myWM = wms[t];

    if (myWM.name == "PolII") {i0 = postss;}
    else                      {i0 = 0;}
    //cerr << t << " " << myWM.name.c_str() << " " << i0 << " " << postss << endl; 
    
    
    for (int i = 0; i < lenseq; i++) {
      if (i + myWM.len <= lenseq) {
	double myprob  = 0;
	double myprobR = 0;

	if (i >= i0) {
	  for (int s = 0; s < myWM.len; s++) {
	    myprob  += myWM.W[s][sequence[s+i]];
	    myprobR += myWM.rW[s][sequence[s+i]];
	  }
	  myprob = (0.5*exp(myprob-myWM.maxscore)+0.5*exp(myprobR-myWM.maxscore));
	}
	probS[i][t] = myprob;
      }
      else {
	probS[i][t] = 0.0;
      }
    }
  }
}
  
// Reading wms file
// ------------------------------------------------------
datawms* readwms(char *filewms, typeparwms *parwms) {
  stringstream ss;
  string line,name,slen;
  string sI, sA,sC,sG,sT;
  double vA,vC,vG,vT;
  double norm;
  int s,t,len;

  double pseudo = parwms -> pseudo;
  double bgprob = parwms -> bgprob;
  int nuclen = parwms -> nuclen;
    
  // Open file
  cout << "RUN: Reading WMs: ";
  ifstream in(filewms);
  if(!in) {
    cerr << "ERROR: file could not be opened" << endl;
    exit(1);
  }

  // Counting WMSs and initialzng the array of datawms (stupid code)
  int maxwms = 2;; 
  while (in >> line) {if (line == ">") {maxwms++;}}
  in.close();
  in.clear();
  datawms *wms = new datawms[maxwms];
  parwms -> maxwms = maxwms;
    
  // Storing free WM 
  wms[0].name = "free";
  wms[0].len = 1;
  wms[0].W = new double*[1];
  wms[0].rW = new double*[1];
  wms[0].W[0] = new double[5];
  wms[0].rW[0] = new double[5];
  for (int b = 0; b < 5; b++) {wms[0].W[0][b]  = log(bgprob);}
  for (int b = 0; b < 5; b++) {wms[0].rW[0][b] = log(bgprob);}
  
  // Storing nucleosome WM
  wms[1].name = "nucleosome";
  wms[1].len = nuclen;
  wms[1].W = new double*[nuclen];
  wms[1].rW = new double*[nuclen];
  for (int s = 0; s < nuclen; s++) {
    wms[1].W[s] = new double[5];
    wms[1].rW[s] = new double[5];
    for (int b = 0; b < 5; b++) {wms[1].W[s][b]  = log(bgprob);}
    for (int b = 0; b < 5; b++) {wms[1].rW[s][b] = log(bgprob);}
  }

  // Reading TF WMs from file
  s = 0;
  t = 1;
  in.open(filewms);
  while (in >> line) {
    if (line == ">") {
      in >> name >> slen;
      len = stoi(slen);

      t++;
      s = 0;
      wms[t].name = name;
      wms[t].len  = len;
      wms[t].W = new double*[len];
      wms[t].rW = new double*[len];
      for (int i = 0; i < len; i++) {
	wms[t].W[i] = new double[5];
	wms[t].rW[i] = new double[5];
      }
    }
    else if (!line.empty()) {
      in >> sA >> sC >> sG >> sT;

      vA = stof(sA)+pseudo;
      vC = stof(sC)+pseudo;
      vG = stof(sG)+pseudo;
      vT = stof(sT)+pseudo;
      norm = vA + vC + vG + vT;
	
      wms[t].W[s][0] = log(vA/norm);
      wms[t].W[s][1] = log(vC/norm);
      wms[t].W[s][2] = log(vG/norm);
      wms[t].W[s][3] = log(vT/norm);
      wms[t].W[s][4] = 0.25*(log(vA/norm) + log(vC/norm) + log(vG/norm) + log(vT/norm));
      s++;
      
      wms[t].rW[len-s][0] = log(vT/norm);
      wms[t].rW[len-s][1] = log(vG/norm);
      wms[t].rW[len-s][2] = log(vC/norm);
      wms[t].rW[len-s][3] = log(vA/norm);
      wms[t].rW[len-s][4] = 0.25*(log(vA/norm) + log(vC/norm) + log(vG/norm) + log(vT/norm));
    }   
  }

  // Calculating maximum score for each WMs
  for (int t = 0; t < maxwms; t++) {
    double maxscore = 0; 
    int len = wms[t].len;
    for (int s = 0; s < len; s++) {
      double mymax = 0; 
      for (int n = 0; n < 4; n++) {
	if (n == 0 || mymax < wms[t].W[s][n]) {
	  mymax = wms[t].W[s][n];
	}
      }
      maxscore += mymax;
    }
    wms[t].maxscore = maxscore;
  }
  
  // Output 
  cout << maxwms << " motifs" << endl;
  
  return(wms);
}


// Reading methylation data
// ------------------------------------------------------
datamet readmet(char *filemet, typeparwms parwms) {
  stringstream ss;
  string substr = "N";
  string line;
  int padlen = parwms.padlen;
  int i;
  int m;

  // Open file
  cout << "RUN: Reading methylation: ";
  ifstream in(filemet);
  if(!in) {
    cerr << "ERROR: file could not be opened" << endl;
    exit(1);
  }

  // Calculating columns and rows of the file (number of methylation sites and number of molecules)
  getline(in, line);
  ss << line;

  int nummol = 0;
  int numsites = 0;
  while(ss >> substr) {numsites++;}
  while(getline(in, line)) {nummol++;}
  in.close();
  in.clear();
  ss.clear();
  
  // Initialization of methylation variables and storing them in the structure met
  datamet met;
  met.nummol = nummol;
  met.numsites = numsites;
  met.msites = new int[numsites];
  met.mstate = new int*[nummol];
  for (int m = 0; m < nummol; m++) {met.mstate[m] = new int[numsites];}

  // Loading methylation sites (first line of the file)
  i = 0;
  in.open(filemet);
  getline(in, line);
  ss << line;
  while (ss >> substr) {
    met.msites[i] = stoi(substr)+padlen;
    //met.msites[i] = stoi(substr);
    i++;
  }
  
  // Loading methylation states (rest of the file)
  i = 0;
  m = 0;
  ss.clear();
  while (getline(in, line)) {
    i = 0;
    ss.clear();
    ss << line;
    while (ss >> substr) {
      met.mstate[m][i] = stoi(substr); 
      i++;
    }
    m++;
  }
  in.close();
  
  // Output
  cout << nummol << " molecules, " << numsites << " meth sites" << endl;

  return(met);
}

// Reading sequence information
// ------------------------------------------------------
dataseq readseq(char *fileseq, typeparwms parwms) {
  dataseq seq;
  string strseq;
  int padlen = parwms.padlen;
  int i = 0;
  
  // Open file
  cout << "RUN: Reading sequence: ";
  ifstream in(fileseq);
  if(!in) {
    cerr << "ERROR: file could not be opened" << endl;
    exit(1);
  }

  // Reading sequence
  in >> strseq;
  int lenseq = strseq.length() + 2*padlen;
  seq.lenseq = lenseq;
  seq.padlen = padlen;
  seq.seq = new int[lenseq];

  for (int i = 0; i < lenseq; i++) {seq.seq[i] = 4;}
    
  while (i < lenseq) {
    if      (strseq[i] == 'A' || strseq[i] == 'a') {
      seq.seq[i+padlen] = 0;
    }
    else if (strseq[i] == 'C' || strseq[i] == 'c') {
      seq.seq[i+padlen] = 1;
    }
    else if (strseq[i] == 'G' || strseq[i] == 'g') {
      seq.seq[i+padlen] = 2;
    }
    else if (strseq[i] == 'T' || strseq[i] == 't') {
      seq.seq[i+padlen] = 3;
    }
    else if (strseq[i] == 'N' || strseq[i] == 'n') {
      seq.seq[i+padlen] = 4;
    }
    i += 1;
  }
  in.close();
    
  // Output
  cout << lenseq << " bp" << endl;

  return(seq);
}


// Reading arguments
// ------------------------------------------------------
void parse_argv(int argc, char **argv, typefiles *files, typerunmod *runmod, typeparwms *parwms, typeparsgd *parsgd, typeparsam *parsam) {
  
  // printing help if there are no arguments or help is called
  string mode;
  if (argc > 1) {mode = argv[1];}
  else          {mode = "help";}          
  if (argc % 2 == 1 || mode == "-h" || mode == "-help" || mode == "--help" || mode == "help") {errorout();}

  // run mode: setting control flow variables 
  if      (mode == "run") {runmod -> sgd  = 1; runmod -> sam  = 1; runmod -> bnd  = 1;}
  else if (mode == "fit") {runmod -> sgd  = 1; runmod -> sam  = 1; runmod -> bnd  = 0;}
  else if (mode == "bnd") {runmod -> sgd  = 0; runmod -> sam  = 0; runmod -> bnd  = 1;}
  else                    {errorout();}
    
  // Setting defoult parameters
  files -> par = NULL;
  files -> bnd = NULL;
  parwms -> padlen = 150;
  parwms -> nuclen = 147;
  parwms -> pseudo = 0.5;
  parwms -> bgprob = 0.25;
  parwms -> cutoffwms = 0.001;
  
  if (runmod -> sgd == 1) {
    parsgd -> numepochs = 100;
    parsgd -> batchsize = 64;
    parsgd -> beta1 = 0.9;
    parsgd -> beta2 = 0.99;
    parsgd -> lrate = 0.01;
    parsgd -> fitgamma = 1;
  }
  if (runmod -> sam == 1) {
    parsam -> numiter = 1000;
    parsam -> sigma = 0.01;
    parsam -> fitgamma = 1;
  }
  if (runmod -> sim == 1) {
    parsim -> nummol = 1000;
  }
  
    
  // Readibg arguments
  int i = 2;
  while (i < argc) {
    string parstr(argv[i]);
    
    if (parstr  == "-h" || parstr == "--help") {errorout();}
    else if (parstr == "-s")          {files -> seq = argv[++i];}
    else if (parstr == "-m")          {files -> met = argv[++i];}
    else if (parstr == "-w")          {files -> wms = argv[++i];}
    else if (parstr == "-p")          {files -> par = argv[++i];}
    else if (parstr == "-o")          {files -> out = argv[++i];}
    else if (parstr == "-b")          {files -> bnd = argv[++i];}
    else if (parstr == "-padlen")     {parwms -> padlen = atoi(argv[++i]);}  
    else if (parstr == "-nuclen")     {parwms -> nuclen = atoi(argv[++i]);}
    else if (parstr == "-pseudo")     {parwms -> pseudo = atof(argv[++i]);}
    else if (parstr == "-bgprob")     {parwms -> bgprob = atof(argv[++i]);}
    else if (parstr == "-cutoffwms")  {parwms -> cutoffwms = atof(argv[++i]);}
    else if (parstr == "-numepochs")  {parsgd -> numepochs = atoi(argv[++i]);}
    else if (parstr == "-batchsize")  {parsgd -> batchsize = atoi(argv[++i]);}
    else if (parstr == "-beta1")      {parsgd -> beta1 = atof(argv[++i]);}
    else if (parstr == "-beta2")      {parsgd -> beta2 = atof(argv[++i]);}
    else if (parstr == "-lrate")      {parsgd -> lrate = atof(argv[++i]);}
    else if (parstr == "-numiter")    {parsam -> numiter = atoi(argv[++i]);}
    else if (parstr == "-sigma")      {parsam -> sigma = atof(argv[++i]);}
    else if (parstr == "-nummol")     {parsim -> nummol = atoi(argv[++i]);}
    else if (parstr == "-fitgamma")   {parsgd -> fitgamma = atoi(argv[++i]); parsam -> fitgamma = parsgd -> fitgamma;}
    else if (parstr == "-numthreads") {numthreads = atoi(argv[++i]);}
    else if (parstr == "-sparse")     {outcutoff = atof(argv[++i]);}
    else if (parstr == "-tss")        {postss = atoi(argv[++i]);}
    else if (parstr == "-k")          {priork = atof(argv[++i]);}
    else if (parstr == "-theta")      {priorb = 1/atof(argv[++i]);}
    i++;
  }
}


// Printingout error messages
// ------------------------------------------------------
void errorout() {
  cerr << endl;
  cerr << "Usage: hiddenfoot <runmode> [options]:" << endl;
  cerr << endl;
  cerr << "  Runmode:" << endl;
  cerr << "    run: fit parameters using SGD, sample posterior distribution using MCMC and caluclate binding profiles for each moelcule" << endl;
  cerr << "    fit: fit parameters using SGD, sample posterior distribution using MCMC" << endl;
  cerr << "    bnd: caluclate binding profiles for each moelcule" << endl;
  cerr << "    sim: simulate binding profiles and methylation data" << endl;
  cerr << endl;
  cerr << "  Files:" << endl;
  cerr << "    -s <fileseq.txt>   file with the DNA sequence" << endl;
  cerr << "    -m <filemet.txt>   file with the methylation matrix" << endl;
  cerr << "    -w <filewms.txt>   file with the PWMs" << endl;
  cerr << "    -p <filepar.txt>   file with parameters values" << endl;
  cerr << "    -b <filennd.txt>   file with a list of bindng states to be considered" << endl;
  cerr << "    -o <filetag>       file tag use to name output files" << endl;
  cerr << endl;
  cerr << "  Perfomance:" << endl;
  cerr << "    -numthreads <int>  number of threads used (default: 1)" << endl;
  cerr << "    -sparse <int>      probability cutoff for sparse output (default: 0)" << endl;
  cerr << endl;
  cerr << "  Prior:" << endl;
  cerr << "    -k                 shape parameter of the prior gamma distribution for polII concentration (default: 1)" << endl;
  cerr << "    -theta             scale parameter of the prior gamma distribution for polII concentration (default: inf)" << endl;  
  cerr << endl;
  cerr << "  PWMs:" << endl;
  cerr << "    -padlen <int>      length of padding sequence added at the flanks (default: 150)" << endl;
  cerr << "    -nuclen <int>      nucleosome length (default: 147)" << endl;
  cerr << "    -pseudo <float>    pseudo count used to calculate PWMs (default: 0.5)" << endl;
  cerr << "    -cutoffwms <float> smallest binding score with respect to the best score to be considered (default: 0.001)" << endl;
  cerr << endl;
  cerr << "  SGD:" << endl;
  cerr << "    -numepochs <int>   number of epochs for the SGD (default: 100)" << endl;
  cerr << "    -batchsize <int>   number of molecules used to estimate the gradient (default: 64)" << endl;
  cerr << "    -lrate <float>     learning rate in the ADAM SGD algorithm (default: 0.01)" << endl;
  cerr << "    -beta1 <float>     exponential decay rate for the 1st moment in the ADAM SGD algorithm (default: 0.9)" << endl;
  cerr << "    -beta2 <float>     exponential decay rate for the 2st moment in the ADAM SGD algorithm (default: 0.99)" << endl;
  cerr << "    -fitgamma <0/1>    sample the parameter gamma (exponent of the Boltzman weiight (default: 1)" << endl; 
  cerr << endl;
  cerr << "  MCMC:" << endl;
  cerr << "    -numiter <int>     number of samples for the MCMC (default: 1000)" << endl;
  cerr << "    -sigma <float>     std used for the proposl normal distribution in MCMC (default: 0.01)" << endl;
  cerr << "    -fitgamma <0/1>    sample the parameter gamma (exponent of the Boltzman weiight (default: 1)" << endl; 
  cerr << endl; 
  cerr << "  SIM:" << endl;
  cerr << "    -nummol <int>      number of simulated molecules (default: 1000)" << endl;
  cerr << endl;
  exit(1);
}


// Dummy function that returns 0/1 (unbound/bound)  
// ------------------------------------------------------
int funbnd(int t) {
  int b = 0;   
  if (t > 0) {b = 1;}
  
  return b; 
}


// Function that returns a random permutation
// ------------------------------------------------------
vector<int> random_permutation(int numelem) {
  //srand(unsigned(time(NULL)));
  //random_device rd;
  //mt19937 g(rd());
  default_random_engine generator((unsigned) clock());
  vector<int> myvector(numelem);

  // Create vector with indexes:
  for (int i = 0; i < numelem; i++) {myvector[i] = i;} 
  
  // Using built-in random generato
  shuffle(myvector.begin(),myvector.end(), generator);
  //random_shuffle(myvector.begin(),myvector.end(), myrandom);
  
  return(myvector);
}

//int myrandom (int i) {return rand()%i;}
