format long
h0 = [1/8,1/16,1/32];
h1 = [1/8,1/16,1/32,1/64];
h = [1/8,1/16,1/32,1/64,1/128];
%S2D_P2
E = [   0.00003615 0.00000226  0.0000001411 0.000000009839  0
        0.00000333489 0.000000204372 0.0000000128842 0.00000000309369 0.00000000163783
        0.0000654284000   0.0000160153000   0.0000039613700   0.0000009852650   0.000000245937000]
E1 = E(1,1:4); 
E2 = E(2,:);
E3 = E(3,:);
E4 = E(2,1:3)
p3_k1 = polyfit(log(h1),log(E1),1); %polyfit
p3_k2 = polyfit(log(h),log(E2),1);
p3_k3 = polyfit(log(h),log(E3),1);
p3_k4 = polyfit(log(h0),log(E4),1)
x = [-6:0.1:-1]; 
F1 = polyval(p3_k1,x); 
F2 = polyval(p3_k2,x);
F3 = polyval(p3_k3,x);
F4 = polyval(p3_k4,x);

figure(1) %||u - u_h||_0
plot(x,F1,'r',log(h),log(E2),'b*',x,F4,'k',log(h1),log(E1),'r*')
%plot(x,F1,'r',log(h),log(E2),'b*',x,F3,'g',x,F4,'k',log(h1),log(E1),'r*',log(h),log(E3),'g*')%plot all  
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||u-u_h||_0)');  
legend('S2DI','S2DII')
%legend('S2DI','S2DII','N2D');
title('Error rate for P_3 approximations');
format short;
 p=[p3_k1(1),p3_k2(1),p3_k3(1),p3_k4(1)]
 gtext(num2str(p3_k1(1)));
 gtext(num2str(p3_k4(1))); 
 %gtext(num2str(p3_k3(1)));


E = [ 0.0005061  0.00006429  0.000008086 0.000001013 0.0000001267 
      0.0001353  0.00001702  0.000002133 0.0000002669 0.00000003338 
      0.00126119 0.000224337 0.0000397802 0.00000704317 0.00000124605]   
E1 = E(1,:); 
E2 = E(2,:);
E3 = E(3,:);
p3_k1 = polyfit(log(h),log(E1),1); %polyfit
p3_k2 = polyfit(log(h),log(E2),1);
p3_k3 = polyfit(log(h),log(E3),1);
x = [-6:0.1:-1]; 
F1 = polyval(p3_k1,x); 
F2 = polyval(p3_k2,x);
F3 = polyval(p3_k3,x);

figure(2) %\|\triangledown\times(\mathbf{u} - \mathbf{u}_h)\|
plot(x,F1,'r',x,F2,'b',log(h),log(E1),'r*',log(h),log(E2),'b*')
%plot(x,F1,'r',x,F2,'b',x,F3,'g',log(h),log(E1),'r*',log(h),log(E2),'b*',log(h),log(E3),'g*')%plot all  
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||curl(u- u_h)||_0 + ||div(u - u_h)||_0)');  
legend('S2DI','S2DII')
%legend('S2DI','S2DII','N2D');
title('Error rates for P_3 approximations');
format short;
 p=[p3_k1(1),p3_k2(1),p3_k3(1)]
 gtext(num2str(p3_k1(1)));
 gtext(num2str(p3_k2(1)));
% gtext(num2str(p3_k3(1)));
 

 
