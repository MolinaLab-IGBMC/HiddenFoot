mol = [1918,2708,1617,1321];

S = importdata('test.params_SGD.txt');
M = importdata('test.params_MCMC.txt');
N = importdata('test.hfprofile.nucleosome.txt');
F = importdata('test.hfprofile.free.txt');

So = importdata('test_omp.params_SGD.txt');
Mo = importdata('test_omp.params_MCMC.txt');
No = importdata('test_omp.hfprofile.nucleosome.txt');
Fo = importdata('test_omp.hfprofile.free.txt');


clustergram([N,1-F-N],'Colormap','redbluecmap','Cluster','column')
clustergram([No,1-F-N],'Colormap','redbluecmap','Cluster','column')

figure
for m = 1:length(mol)
    subplot(2,2,m)
    hold on
    plot(N(mol(m),:))
    plot(No(mol(m),:))
    plot(1-F(mol(m),:)-N(mol(m),:))
    plot(1-Fo(mol(m),:)-No(mol(m),:))
end

figure
subplot(1,2,1)
hold on
plot(S.data(:,2))
plot(So.data(:,2))

subplot(1,2,2)
hold on
plot(M.data(:,2))
plot(Mo.data(:,2))