clearvars;clc;
N=15;n=0:N-1;%Number of samples of incoming signal
K = 10;%Number of Users
theta = [-50 -35 -20 -10 0 10 20 35 50 70];%directions
a_k = ones(K,1);%amplitude levels for individual users
SNR_dB=20;gamma=10^(SNR_dB/10);sigma=1/sqrt(gamma);
M=1024;t=0;
theta=theta.*(pi/180);
f=sin(theta)/2;
phi_k=2*pi*rand(K,1)-pi;%uniform distb. [-pi pi]
f=f(:);
%% PSD
xn_k = a_k.*exp(1j*2*pi*theta').*exp(1j*2*pi*f*(0:N-1));
for kk=1:N
    xnk=xn_k(:,kk);
    R=xnk*xnk';
    t=t+R;
    f_k=f.';
    W=exp(-1j*2*pi*n'*f_k);
    u = sigma*randn(N,1)+1j*randn(N,1);
    xn = W*xnk+u;
end
R_ss=t/N;
Rss = diag(a_k.^2);Rxx=W*R_ss*W'+sigma^2*eye(N);
fo = linspace(-0.5,0.5,M);
r = fftshift(fft(Rxx(1,:),M));
r1 = abs(r);r1 = 10*log10(r1/max(r1));
plot(fo,(r1),'linewidth',2);hold on;
%% MUSIC
[Q,D]=eig(Rxx);
k=sum(sum(round(D)~=0));
K1 = N-k;
V = Q(:,[k+1:end]);
W1 = exp(-1j*2*pi*n'*fo); 
for i = 1:length(fo)
PsePSD(i) = 1/((W1(:,i)'*(V*V')*W1(:,i)));
end
Pmusic=10*log10(abs(PsePSD)/max(abs(PsePSD))); %Spatial spectrum function
plot(fo,Pmusic,'linewidth',2) 
%% ESPRIT
%ESPRIT algorithm is an algebraic algorithm that calculates DoAs without
%spectrum. 
Xn = W*xn_k+sigma^2*eye(N);
[U,S,V] = svd(Xn);
MM  = diag(S);
[rr,cc]=find(abs(MM)-sigma^2<0.00001);
kapa = sum(cc);
KK = N-kapa;
U_est = U(:,[1:K]);val = 1;
J_x = [eye(N-val) zeros(N-val,1)];
J_y = [zeros(N-val,1) eye(N-val)];
Ux = J_x*U_est;
Uy = J_y*U_est;
Z = pinv(Ux)*Uy;
[T,thetas] = eig(Z);
ang = diag(asin(angle(thetas)/pi)*180/pi);
%% MVDR
W1 = exp(-1j*2*pi*n'*fo); 
for i = 1:M
P_MVDR(i) = 1/(W1(:,i)'*inv(Rxx)*W1(:,i));
end
Pmvdr=10*log10(abs(P_MVDR)/max(abs(P_MVDR))); %Spatial spectrum function
plot(fo,(Pmvdr),'linewidth',2);hold on;grid minor;
%% Matched Filter
y = xn'*W1;
y_mf=10*log10(abs(y)/max(abs(y)));
plot(fo,(y_mf),'linewidth',2);hold on;
%% Expectation Maximization
%EM is an iterative algorithm. Greater the number of iterations better
%performance
W1 = exp(-1j*2*pi*n'*fo);%Reference filter bank
%Random initialization of Rzz
rnf = rand(M,1);
Rzz = diag(rnf.^2);
Xn = W*xn_k+sigma^2*eye(N);
% Rzz = eye(M);
for i = 1:20
    Rxx1 = W1*Rzz*W1';
    Rxx1 = Rxx1/N;
    Rxx_inv = inv(Rxx1);
    for mm = 1:M
        w_m =W1(:,mm);
        w_mvdr = Rxx_inv*w_m/(w_m'*Rxx_inv*w_m);
        y_em = w_mvdr'*Xn;
% Multiple Measurement Vector
%         for aa = 1:12
%             y_em(:,aa) = w_mvdr'*Xn(:,aa);
%         end
        Rzz(mm,mm) = y_em*y_em';
    end
end
P_em = abs(diag(Rzz));
P_EM=10*log10((P_em)/max((P_em))); hold on;
plot(fo,((P_EM)),'linewidth',2);hold on;
hold on
plot(f_k,-75,'ro','markerfacecolor','r')
legend({'PSD','MUSIC','MVDR','MF','EM'},'Location','southwest','NumColumns',2)
% end
