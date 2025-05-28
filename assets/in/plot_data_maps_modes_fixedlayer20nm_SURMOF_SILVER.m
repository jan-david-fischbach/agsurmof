clear all
%% TM 

load('3poles_surmof_20nmAu/Data_Material1SURMOF_Material2SILVERSCS_ABS_maps_SURMOF_SILVER_upquadrupole_fixedlayer20nm_allpoles.mat')


%dipole tm
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode1_fixedthickness20nm.mat')
mode1=full_tm;
mode1(:,2)=mode1(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode2_fixedthickness20nm.mat')
mode2=full_tm;
mode2(:,2)=mode2(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode3_fixedthickness20nm.mat')
mode3=full_tm;
mode3(:,2)=mode3(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode4_fixedthickness20nm.mat')
mode4=full_tm;
mode4(:,2)=mode4(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode5_fixedthickness20nm.mat')
mode5=full_tm;
mode5(:,2)=mode5(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode6_fixedthickness20nm.mat')
mode6=full_tm;
mode6(:,2)=mode6(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode7_fixedthickness20nm.mat')
mode7=full_tm;
mode7(:,2)=mode7(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode8_fixedthickness20nm.mat')
mode8=full_tm;
mode8(:,2)=mode8(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_dipole_mode9_fixedthickness20nm.mat')
mode9=full_tm;
mode9(:,2)=mode9(:,2)+20*1e-3;


%dipole TE
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode1_fixedthickness20nm.mat')
mode10=full_tm;
mode10(:,2)=mode10(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode2_fixedthickness20nm.mat')
mode11=full_tm;
mode11(:,2)=mode11(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode3_fixedthickness20nm.mat')
mode12=full_tm;
mode12(:,2)=mode12(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode4_fixedthickness20nm.mat')
mode13=full_tm;
mode13(:,2)=mode13(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode5_fixedthickness20nm.mat')
mode14=full_tm;
mode14(:,2)=mode14(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_dipole_mode6_fixedthickness20nm.mat')
mode15=full_tm;
mode15(:,2)=mode15(:,2)+20*1e-3;


%quadrupole TM 
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode1_fixedthickness20nm.mat')
mode16=full_tm;
mode16(:,2)=mode16(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode2_fixedthickness20nm.mat')
mode17=full_tm;
mode17(:,2)=mode17(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode3_fixedthickness20nm.mat')
mode18=full_tm;
mode18(:,2)=mode18(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode4_fixedthickness20nm.mat')
mode19=full_tm;
mode19(:,2)=mode19(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode5_fixedthickness20nm.mat')
mode20=full_tm;
mode20(:,2)=mode20(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TM_particle_quadrupole_mode6_fixedthickness20nm.mat')
mode21=full_tm;
mode21(:,2)=mode21(:,2)+20*1e-3;

%quadrupole TE

load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_quadrupole_mode1_fixedthickness20nm.mat')
mode22=full_tm;
mode22(:,2)=mode22(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_quadrupole_mode2_fixedthickness20nm.mat')
mode23=full_tm;
mode23(:,2)=mode23(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_quadrupole_mode3_fixedthickness20nm.mat')
mode24=full_tm;
mode24(:,2)=mode24(:,2)+20*1e-3;
load('3poles_surmof_20nmAu/SURMOF_SILVER_TE_particle_quadrupole_mode4_fixedthickness20nm.mat')
mode25=full_tm;
mode25(:,2)=mode25(:,2)+20*1e-3;






%--------------------------------------------------------------------



 lam = lam*1e6; % [µm]
 radiusf=radiusf*1e9; %[nm]


 [XX,YY]=meshgrid(radiusf(2,:),lam);

% -------------------------------------------------------------------------
figure;
set(0,'DefaultAxesFontSize',20,'DefaultAxesFontName','Helvetica');
x0=20;
y0=20;
width=1400;
height=650;
set(gcf,'position',[x0,y0,width,height])

hold on
h3=pcolor(1./XX.*1e3,2*pi./YY,(exblk));
h3.EdgeColor='none';
set(gca, 'YDir', 'normal');
colorbar
 colormap turbo
h4=plot(1./mode1(:,2),real(mode1(:,1)),'-y',LineWidth=2.5);
plot(1./mode2(:,2),real(mode2(:,1)),'-y',LineWidth=2.5)
plot(1./mode3(:,2),real(mode3(:,1)),'-y',LineWidth=2.5)
plot(1./mode4(:,2),real(mode4(:,1)),'-y',LineWidth=2.5)
plot(1./mode5(:,2),real(mode5(:,1)),'-y',LineWidth=2.5)
plot(1./mode6(:,2),real(mode6(:,1)),'-y',LineWidth=2.5)
plot(1./mode7(:,2),real(mode7(:,1)),'-y',LineWidth=2.5)
plot(1./mode8(:,2),real(mode8(:,1)),'-y',LineWidth=2.5)
plot(1./mode9(:,2),real(mode9(:,1)),'-y',LineWidth=2.5)

h44=plot(1./mode10(:,2),real(mode10(:,1)),'-m',LineWidth=2.5)
plot(1./mode11(:,2),real(mode11(:,1)),'-m',LineWidth=2.5)
plot(1./mode12(:,2),real(mode12(:,1)),'-m',LineWidth=2.5)
plot(1./mode13(:,2),real(mode13(:,1)),'-m',LineWidth=2.5)
plot(1./mode14(:,2),real(mode14(:,1)),'-m',LineWidth=2.5)
plot(1./mode15(:,2),real(mode15(:,1)),'-m',LineWidth=2.5)

h444=plot(1./mode16(:,2),real(mode16(:,1)),':y',LineWidth=2.5)
plot(1./mode17(:,2),real(mode17(:,1)),':y',LineWidth=2.5)
plot(1./mode18(:,2),real(mode18(:,1)),':y',LineWidth=2.5)
plot(1./mode19(:,2),real(mode19(:,1)),':y',LineWidth=2.5)
plot(1./mode20(:,2),real(mode20(:,1)),':y',LineWidth=2.5)
plot(1./mode21(:,2),real(mode21(:,1)),':y',LineWidth=2.5)

h4444=plot(1./mode22(:,2),real(mode22(:,1)),':m',LineWidth=2.5)
h4444=plot(1./mode23(:,2),real(mode23(:,1)),':m',LineWidth=2.5)
h4444=plot(1./mode24(:,2),real(mode24(:,1)),':m',LineWidth=2.5)
h4444=plot(1./mode25(:,2),real(mode25(:,1)),':m',LineWidth=2.5)
 h1=yline(real(poles(1)),'--r',LineWidth=1.7);
yline(real(poles(2)),'--r',LineWidth=1.7);
yline(real(poles(3)),'--r',LineWidth=1.7);
 h2=yline(real(zn(1)),':k',LineWidth=1.7);
 yline(real(zn(2)),':k',LineWidth=1.7);
 yline(real(zn(3)),':k',LineWidth=1.7);
%   legend('l=1 - TM','l=2 - TM','l=3 - TM','l=4 - TM','l=5 - TE','Material resonance')
  legend([h1,h2,h4,h44,h444,h4444],{'Material poles','Material zeros','TM dipole modes', ...
      'TE dipole modes','TM quadrupole modes','TE quadrupole modes',});
%   legend([h1,h2],{'Material resonance','Material zeros'});
  set(legend, 'Color', [0.85, 0.85, 0.85]);
ylim([2*pi./lam(end) 2*pi./lam(1)]); xlim([1./radiusf(2,end)*1e3 1./radiusf(2,37)*1e3]);
title('The extinction cross-section '); 
% caxis([0 0.07])
ylabel('k (2pi/µm)','Interpreter','latex');
% xlabel('radius$_{shell}$, nm','Interpreter','latex');
xlabel('1./radius_{total} (µm^{-1})');


