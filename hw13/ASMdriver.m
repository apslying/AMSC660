function ASMdriver()
 %% the Rosenbrock function
 a = 5;
 func = @(x,y)(x-1).^2 +(y-2.5).^2;
 %func = @(x,y)(1-x).^2 + a*(y- x.^2).^2; % Rosenbrock's function
 gfun = @(x)[2*x(1)-2; 2*x(2)-5]; %y in terms of x?
 %gfun = @(x)[-2*(1-x(1))-4*a*(x(2)-x(1)^2)*x(1);2*a*(x(2)-x(1)^2)]; % gradient of f
 Hfun = @(x)[2,0;0,2];
 %Hfun = @(x)[2 + 12*a*x(1)^2- 4*a*x(2),-4*a*x(1);-4*a*x(1), 2*a]; % Hessian of f
 lsets = exp([-3:0.5:2]);
 %% constraints
 Nv = 6;
 t = linspace(0,2*pi,Nv+1);
 t(end) = [];
 t0 = 0.1;
 vertsT = [2,2;0,1;0,0;2,0;4,1];
 verts = transpose(vertsT);
 % verts = [0.1+cos(t0+t);0.1+sin(t0+t)];
 R = [0,-1;1,0];

 A = [1 -2; 
     -1 -2; 
     -1 2; 
      1 0; 
      0 1];

 %A = (R*(circshift(verts,[0,-1])-verts))';
 b=[-2;-6;-2;0;0];
 %b = verts(1,:)'.*A(:,1) + verts(2,:)'.*A(:,2); % b_i = a_i*verts(:,i)
 x=[2;0];
 %x = [-0.5;0.5];
 W = [3;5];
 [xiter,lm] = ASM(x,gfun,Hfun,A,b,W);
 %% graphics
  close all
 fsz = 16;
 figure(1);
 hold on;
 n = 100;
 txmin = min(verts(1,:))-0.2;
 txmax = max(verts(1,:))+0.2;
 tymin = min(verts(2,:))-0.2;
 tymax = max(verts(2,:))+0.2;
 tx = linspace(-1,4,n);
 ty = linspace(-1,4,n);
 [txx,tyy] = meshgrid(tx,ty);
 ff = func(txx,tyy);
 contour(tx,ty,ff,lsets,'Linewidth',1);
 edges = [verts,verts(:,1)];
 plot(edges(1,:),edges(2,:),'Linewidth',2,'color','k');
 plot(xiter(1,:),xiter(2,:),'Marker','.','Markersize',20,'Linestyle','-',...
 'Linewidth',2,'color','r');
 xlabel('x','Fontsize',fsz);
 ylabel('y','Fontsize',fsz);
 set(gca,'Fontsize',fsz);
 colorbar;
 grid;
 daspect([1,1,1]);
 end
 function [xiter,lm] = ASM(x,gfun,Hfun,A,b,W)
 %% minimization using the active set method (Nocedal & Wright, Section 16.5)
 % Solves f(x)--> min subject to Ax >= b
 % x = initial guess, a column vector
 TOL = 1e-10;
 dim = length(x);
 g = gfun(x);
 H = Hfun(x);
 iter = 0;
 itermax = 1000;
 m = size(A,1); % the number of constraints
 % W = working set, the set of active constrains
 I = (1:m)';
 Wc = I; % the compliment of W
 xiter = x;
 while iter < itermax
 % compute step p: solve 0.5*p'*H*p + g'*p--> min subject to A(W,:)*p = 0
  AW = A(W,:); % LHS of active constraints
  % iter
  x
  % AW
 % fix H if it is not positive definite
 ee = sort(eig(H),'ascend');
 if ee(1) < 1e-10
 lam =-ee(1) + 1;
 else
 lam = 0;
 end
 H = H + lam*eye(dim);
 if ~isempty(W)
 M = [H,-AW';AW,zeros(size(W,1))];
 RHS = [-g;zeros(size(W,1),1)];
 else
 M = H;
 RHS =-g;
 end
 % M
 % RHS
 aux = M\RHS;
 p = aux(1:dim);
 lm = aux(dim+1:end);
 if norm(p) < TOL % if step == 0
 %p
 if ~isempty(W)
 lm = AW'\g; % find Lagrange multipliers
 if min(lm) >= 0 % if Lagrange multipliers are positive, we are done
 % the minimizer is one of the corners
 fprintf('A local solution is found, iter = %d\n',iter);
 fprintf('x = [\n'); fprintf('%d\n',x);fprintf(']\n');
 break;
 else % remove the index of the most negative multiplier from W
 [lmin,imin] = min(lm);
 W = setdiff(W,W(imin));
 Wc = setdiff(I,W);
 end
 else
 fprintf('A local solution is found, iter = %d\n',iter);
 fprintf('x = [\n'); fprintf('%d\n',x);fprintf(']\n');
 break;
 end
 else % if step is nonzero
 %p
 alp = 1;
 % check for blocking constraints
 Ap = A(Wc,:)*p;
 icand = find(Ap <-TOL);
 if ~isempty(icand)
      % find step lengths to all possible blocking constraints
 al = (b(Wc(icand))- A(Wc(icand),:)*x)./Ap(icand);
 % find minimal step length that does not exceed 1
 [almin,kmin] = min(al);
 alp = min(1,almin);
 end
 x = x + alp*p;
 g = gfun(x);
 H = Hfun(x);
 if alp < 1
 W = [W;Wc(icand(kmin))];
 Wc = setdiff(I,W);
 end
 end
 iter = iter + 1;
 xiter = [xiter,x];
 end
 if iter == itermax
 fprintf('Stopped because the max number of iterations %d is performed\n',iter);
 end
 end