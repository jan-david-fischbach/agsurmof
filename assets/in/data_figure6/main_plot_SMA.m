
clear all


dataFile='Data_SMA.mat';

loaded = load(dataFile, 'kn');
kn = loaded.kn;

thickness_surmof_t=linspace(kn(1).T(2),kn(end).T(2),size(kn,1)); % 
Nr=length(thickness_surmof_t); % number of radius points

poles=sort(kn(1).poles(1:3));




q_rse=zeros(Nr,4); % RSE
gcoeff=zeros(Nr,3); % coupling |iV_1^(p)|
kcshift=zeros(Nr,1); % kn_shift
kc=zeros(Nr,1); % unpeturbed kn

for ind=1:Nr
q_rse(ind,:)=sort(kn(ind).knNew);

gcoeff(ind,:)=(kn(ind).coupl_coeffs(end:-1:1));

kcshift(ind)=kn(ind).knShift;
kc(ind)=kn(ind).knVal{1};

end


X_rse = thickness_surmof_t*1e6;




%% plot

x_lim=[2 14];
y_lim=[6 12.5];

ggg=figure('Units','Centimeter');
set(ggg,'DefaultAxesFontSize',24,'DefaultAxesFontName','Helvetica');
x0=50;
y0=50;
width=550;
height=400;
set(ggg,'position',[x0,y0,width,height])
sub1=subplot(3,1,[1,2]);

hold on  

nline=2.25;


h2=plot(1./X_rse(1:1:end),real(q_rse(1:1:end,1)),'b-',LineWidth= nline); % rse
for ind_rse=2:4
plot(1./X_rse(1:1:end),real(q_rse(1:1:end,ind_rse)),'b-',LineWidth= nline); % rse
end

h6=plot(1./X_rse(1:1:end),real(kcshift),'b:',LineWidth= 1.15); % rse
h5=plot(1./X_rse(1:1:end),real(kc),'b-',LineWidth= 1.15); % rse

h4=yline(real(poles(1)),'r-',LineWidth=2); % material pole
yline(real(poles(2)),'c-',LineWidth=2); % material pole
yline(real(poles(3)),'g-',LineWidth=2); % material pole
xlim(x_lim);
ylim(y_lim)
box on





lh=legend([h2,h5,h6],{'optical poles','$k_1$','$k_1 - i \sum_p V_{11}^{(p)}$'},'Interpreter','latex');

lh.Color = 'none';      
lh.EdgeColor = 'none'; 

xlabel('Inverse Cavity Thickness (µm^{-1})')

ylabel('Re(k) (2π/µm)' )


sub2=subplot(3,1,3);


hold on
plot(1./X_rse(1:1:end),abs((gcoeff(1:1:end,1))),'r-',LineWidth= nline); % rse
plot(1./X_rse(1:1:end),abs( (gcoeff(1:1:end,2))),'c-',LineWidth= nline); % rse
plot(1./X_rse(1:1:end),abs( (gcoeff(1:1:end,3))),'g-',LineWidth= nline); % rse
box on
lh2=legend('mat. pole p_1','mat. pole p_2','mat. pole p_3','Location','west');
lh2.EdgeColor = 'none'; 
xlim(x_lim);
ylabel('$| iV^{(p_i)}_{11} |$','Interpreter','latex')
xlabel('Inverse Cavity Thickness (µm^{-1})')

set(sub1, 'Position', [0.10 0.410 0.75 0.500]);

set(sub2, 'Position', [0.10 0.110 0.75 0.200]);

