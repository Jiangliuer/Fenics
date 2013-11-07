format long
h = [1/8,1/16,1/32,1/64,1/128,1/256];
%S2D_P2
E =[   0.07296670 0.01885580 0.00475379  0.00119096 0.00029790  0.00007448
       0.02359000 0.00594300 0.00148900  0.00037230 0.00009310  0.00002328];
E1 = E(1,:); 
E2 = E(2,:);
p1_k1 = polyfit(log(h),log(E1),1); %polyfit
p1_k2 = polyfit(log(h),log(E2),1);
x = [-6:0.1:-1]; 
F1 = polyval(p1_k1,x); 
F2 = polyval(p1_k2,x); 

figure(1) %||u - u_h||_0
plot(x,F1,'r',x,F2,'b',log(h),log(E1),'r*',log(h),log(E2),'b*')%plot all  
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||u-u_h||_0)');  
legend('S2DI','S2DII');
title('Error rate for P_1 approximations');
format short;
p = [p1_k1(1),p1_k2(1)]
gtext(num2str(p1_k1(1)));
gtext(num2str(p1_k2(1))); 

E = [0.11155700 0.05581560 0.02791220 0.01395670 0.00697840 0.00348921 
     0.1653     0.08297    0.04152    0.02077    0.01038    0.005192];
E1 = E(1,:);
E2 = E(2,:); 
p1_k1 = polyfit(log(h),log(E1),1); %polyfit
p1_k2 = polyfit(log(h),log(E2),1);
x = [-6:0.1:-1]; 
F1 = polyval(p1_k1,x); 
F2 = polyval(p1_k2,x); 

figure(2) %\|\triangledown\times(\mathbf{u} - \mathbf{u}_h)\|
plot(x,F1,'r',x,F2,'b',log(h),log(E1),'r*',log(h),log(E2),'b*')%plot all
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||curl(u- u_h)||_0 + ||div(u - u_h)||)');  
legend('S2DI','S2DII');
title('Error rates for P_1 approximations');
format short;
p= [p1_k1(1),p1_k2(1)]
gtext(num2str(p1_k1(1)));
gtext(num2str(p1_k2(1))); 