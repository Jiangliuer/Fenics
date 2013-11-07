format long
h0 = [1/8,1/16,1/32,1/64,1/128];
h = [1/8,1/16,1/32,1/64,1/128,1/256];
%S2D_P2
E =[     0.001414000 0.000176600 0.000022100 0.000002770  0.000000344   0
         0.000441900 0.000055640 0.000006981 0.0000008732 0.00000010206 0 
         0.001973110 0.000927175 0.000445322 0.0002169660 0.000106746000000   0.000052855800000];
E1 = E(1,1:5); 
E2 = E(2,1:5);
E3 = E(3,:);
p2_k1 = polyfit(log(h0),log(E1),1); %polyfit
p2_k2 = polyfit(log(h0),log(E2),1);
p2_k3 = polyfit(log(h),log(E3),1);
x = [-6:0.1:-1]; 
F1 = polyval(p2_k1,x); 
F2 = polyval(p2_k2,x);
F3 = polyval(p2_k3,x);

figure(1) %||u - u_h||_0
plot(x,F1,'r',x,F2,'b',log(h0),log(E1),'r*',log(h0),log(E2),'b*')%plot all
%plot(x,F1,'r',x,F2,'b',x,F3,'g',log(h0),log(E1),'r*',log(h0),log(E2),'b*',log(h),log(E3),'g*')%plot all  
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||u-u_h||_0)');  
legend('S2DI','S2DII')
%legend('S2DI','S2DII','N2D');
title('Error rate for P_2 approximations');
format short;
p=[p2_k1(1),p2_k2(1),p2_k3(1)]
gtext(num2str(p2_k1(1)));
gtext(num2str(p2_k2(1)));
%gtext(num2str(p2_k3(1)));
 

E = [    0.0170820 0.0043392 0.0010914 0.0002735 0.0000685 0.0000171 
         0.006832  0.001721  0.0004317 0.0001081 0.00002704     0
          0.015068800000000   0.003496030000000   0.000836508000000   0.000203841000000   0.000050222700000   0.000012454100000]
E1 = E(1,:); 
E2 = E(2,1:5);
E3 = E(3,:);
p2_k1 = polyfit(log(h),log(E1),1); %polyfit
p2_k2 = polyfit(log(h0),log(E2),1);
p2_k3 = polyfit(log(h),log(E3),1);
x = [-6:0.1:-1]; 
F1 = polyval(p2_k1,x); 
F2 = polyval(p2_k2,x);
F3 = polyval(p2_k3,x);

figure(2) %\|\triangledown\times(\mathbf{u} - \mathbf{u}_h)\|
plot(x,F1,'r',x,F2,'b',log(h),log(E1),'r*',log(h0),log(E2),'b*')
%plot(x,F1,'r',x,F2,'b',x,F3,'g',log(h),log(E1),'r*',log(h0),log(E2),'b*',log(h),log(E3),'g*')%plot all  
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('ln(h)'); 
ylabel('ln(||curl(u- u_h)||_0 + ||div(u - u_h)||)');  
legend('S2DI','S2DII')
%legend('S2DI','S2DII','N2D');
title('Error rates for P_2 approximations');
format short;
p=[p2_k1(1),p2_k2(1),p2_k3(1)]
gtext(num2str(p2_k1(1)));
gtext(num2str(p2_k2(1)));
%gtext(num2str(p2_k3(1)));
 

 
