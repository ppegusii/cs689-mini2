%Make an HMM with mixture of Gaussian observations
%    Q1 ---> Q2
%  /  |   /  |
% M1  |  M2  | 
%  \  v   \  v
%    Y1     Y2 
% where Pr(m=j|q=i) is a multinomial and Pr(y|m,q) is a Gaussian     
%-------------------------------------------------------------------

%*************** YOU CAN CUSTOMIZE THIS ***************************
%fileloc =[pwd,'/'];  %load the data stored in directory fileloc	
fileloc =[pwd,'/../../data/'];  %load the data stored in directory fileloc	
resultloc = [fileloc,'/results_15_3/'];  % store the output in this subdirectory

% DBN Information
ss        = 3;       % nodes per time slice
state     = 1;
mixt      = 2;
obs       = 3;
nextState = 4;
ns = zeros(1,ss);    % node sizes
ns(state) = 15;       % num hidden states: VARY THIS
ns(mixt)  = 3;       % num mixture components per state: VARY THIS 
ns(obs)   = 2;       % size of observed vector

% DBN Links
intra = zeros(ss);                   % intra time slice connections
intra(state, [mixt, obs]) = 1;
intra(mixt,  [obs])       = 1;
inter = zeros(ss);                   % inter time slice connections
inter(state, [state])     = 1;

% DBN Nodes (discrete, observed) and Equivalence Classes (1: for t=1, 2: for t>1)
dnodes  = [state mixt];
onodes  = [obs];
eclass1 = [state mixt obs];
eclass2 = [nextState mixt obs];

% Dataset Info
num_people = 7;
people     = strvcat('a','b','c','d','e','f','g');   % access this by people(i,:)
lr         = 'l';      % this tells us we will use the images from the "left" camera when accessing the data
                       %       you could change this to 'r' for the right camera

% Make a DBN for Each Person
bn_a   = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_d   = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_b  = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_c  = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_e = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_f    = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bn_g  = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);

% For each Person, run his/her data through EM to get the parameters of his/her DBN
for k=1:num_people

   if (k==1)
      bnet = bn_a;
   elseif (k==2)
      bnet = bn_b;
   elseif (k==3)
      bnet = bn_c;
   elseif (k==4)
      bnet = bn_d;
   elseif (k==5)
      bnet = bn_e;
   elseif (k==6)
      bnet = bn_f;
   elseif (k==7)
      bnet = bn_g;
   end;

   %------------------------------------------------------------------------%
   % NOTE: There are 15 sequences per person.  I am using the first ten for %
   %       training and the last five for testing.  If you want to change   %
   %       this, you need to make a change here as well as in the testing   %
   %       section below where I say:                                       %
   %                for i=11:15                                             %
   %------------------------------------------------------------------------%
   numEpisodes    = 10;
   data           = cell(1,numEpisodes);    % cell array for all data (observable & hidden)
   obsData_helper = cell(1,numEpisodes);    % helper to go from cell array to regular array
   obsData        = [];                     % array just for the observable data
   % Insert all the Training Data into 'data'
   for i = 1:numEpisodes
      % Load the Data File into Variable 'rawdata'
      filename = strcat(fileloc,people(k,:),int2str(i),'.dat');	
      rawdata  = load(filename);      % load the data from the file
      rawdata  = rawdata(:,1:2)';     % just use height and width (cols 1 and 2), then transpose the data
      T        = length(rawdata);     % number of timesteps in this sequence
      data{i}  = cell(ss,T);          % make the cell array for this sequence to be the correct size
      obsData_helper{i} = cell(1,T);

      % Insert the Data into both 'data' and 'obsData'
      obsCount = obs;
      for j = 1:T
         data{i}{obsCount} = zeros(2,1);
         data{i}{obsCount} = rawdata(:,j);
         obsData_helper{i}{j} = zeros(2,1);
         obsData_helper{i}{j} = rawdata(:,j);
         obsData = [obsData rawdata(:,j)];
         obsCount = obsCount + ss;
      end;
   end;
   % Use K-Means to Initialize the Gaussian Parameters
   [mu0, Sigma0] = mixgauss_init(ns(mixt)*ns(state),obsData,'full', 'rnd');
   covpriorweight = 0.1;

   % Insert the Gaussian CPT along with random CPTs for the other parameters into the Current DBN
   bnet.CPD{state}     = tabular_CPD(bnet, state, 'CPT', 'rnd');
   bnet.CPD{mixt}      = tabular_CPD(bnet, mixt, 'CPT', 'rnd');
   bnet.CPD{obs}       = gaussian_CPD(bnet, obs, 'mean', mu0, 'cov', Sigma0, 'cov_prior_weight', covpriorweight);
   bnet.CPD{nextState} = tabular_CPD(bnet, nextState, 'CPT', 'rnd');

   % Use EM to get the CPDs for the DBN. 
   engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
   max_iter = 100;
   %thr = 1e-5;
   LL = cell(1,1);      % log likelihood
   time = zeros(1,1);
   tic
   %[bnet2, LL] = learn_params_dbn_em(engine, data, 'max_iter', max_iter,'thresh',thr);
   [bnet2, LL] = learn_params_dbn_em(engine, data, 'max_iter', max_iter);
   time(i) = toc;
   fprintf('engine took %6.4f seconds\n', time);

   % Save the Resulting DBNs
   if (k==1)
      bn_a = bnet2;
   elseif (k==2)
      bn_b = bnet2;
   elseif (k==3)
      bn_c = bnet2;
   elseif (k==4)
      bn_d = bnet2;
   elseif (k==5)
      bn_e = bnet2;
   elseif (k==6)
      bn_f = bnet2;
   elseif (k==7)
      bn_g = bnet2;
   end;

end;


% Check the 5 test cases to see how they get categorized by the 7 Models
for k=1:num_people
   % Process test cases in the following order:  a, b, c, d, e, f, g
   testperson  = people(k,:);
   exp_results = [];                  % save experimental results (ie. log likelihoods) in a matrix

   % Access the Test Data Files
   for i=11:15
      % Load the Data File into Variable 'testdata'
      test_filename = strcat(fileloc,testperson,int2str(i),'.dat');
      testData      = load(test_filename);      % load the test data from the file
      testData      = testData(:,1:2)';    % just use height and width (cols 1 and 2), then transpose the data
      loglik        = [];

      % Run the Test Data versus all Seven Learned Models
      % The model that gives the best likelihood wins
      for j=1:num_people
         if (j==1)
            bn = bn_a;
         elseif (j==2)
            bn = bn_b;
         elseif (j==3)
            bn = bn_c;
         elseif (j==4)
            bn = bn_d;
         elseif (j==5)
            bn = bn_e;
         elseif (j==6)
            bn = bn_f;
         elseif (j==7)
            bn = bn_g;
         end;

         % Access the Learned CPTs for the particular Person
         cpd        = struct(bn.CPD{1});
         prior1     = cpd.CPT;
         cpd        = struct(bn.CPD{4});
         transmat1  = cpd.CPT;
         cpd        = struct(bn.CPD{3});
         mu1        = cpd.mean;
         Sigma1     = cpd.cov;
         cpd        = struct(bn.CPD{2});
         mixmat1    = cpd.CPT;

         % Run the Test Data through the Learned Model and record the Log Likelihoods
         [ll1, err1] = mhmm_logprob(testData,prior1,transmat1,mu1,Sigma1,mixmat1);
         loglik      = [loglik; ll1];

      end;
      % Record the Log Likelihoods
      exp_results = [exp_results loglik];

   end;
   % Save Results in a File for each Test
   test_savename = strcat(resultloc,testperson,'.dat');
   save(test_savename,'exp_results','-ascii');
end;

