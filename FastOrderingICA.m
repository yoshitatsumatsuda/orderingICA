% Fast Ordering ICA with MATLAB Implementation
% Author: Yoshitatsu Matsuda    Jul. 19 2024

function [W,Y] =  FastOrderingICA(X,L)
% W - separating matrix
% Y - estimated source
% X - observed signal (mixture)
% L - number of multiple candidates (default: 100)
if nargin < 2
      L = 100;
end;

[N,M]	= size(X);
epsilon = 10^-6;
K = 30;
% N - number of observed signals (N) 
% M - sample size
% epsilon - threshold of iterations in fast ICA
% K - max number of iterations in fast ICA

% pre-whitening
X = X - mean(X')' * ones(1,M);
[V,E] = eig((X*X')/M);
scales = diag(E);
C = diag(1./sqrt(scales))*V';
X = C*X;

% main process;
W = [];
for i = 1:N
    if i>1,
        F = eye(N) - W'*W;
        F = F(1:(N-i+1),:);
        G = inv(sqrtm(F*F'))*F;
        Xt = G*X;
    else
        G = eye(N);
        Xt = X;
    end;
    B = randn(L,N-i+1);
    Bc = [];
    B = diag(1./sqrt(sum(B.^2,2)))*B;
    for t = 1:K
        Bp = B;
        Z = B*Xt;
        B = (Z.^3)*Xt'/M - 3*B;
        B = diag(1./sqrt(sum(B.^2,2)))*B;
        index = find(min(sum((B-Bp).^2,2),sum((B+Bp).^2,2))<epsilon);
        if ~isempty(index)
            Bc = [Bc;B(index,:)];
            B(index,:) = [];
        end;
        if isempty(B)
            break;
        end;
    end;
    if isempty(Bc)
        Bc = B;
    end
    Z = Bc*Xt;
    alpha = sum(Z.^4,2)/M - 3;
    Upsilon = alpha-2*log((alpha+2)/2);
    [maxU,ind]=max(Upsilon);
    if(maxU < (2*(N+1-i)*(N+2-i))/M)
        break;
    end;
    w = Bc(ind,:)*G;
    W = [W;w];
end

% results
Y = W*X;
W = W*C;
alpha = mean(Y'.^4)-3;
Upsilon = alpha-2*log((alpha+2)/2);
[sortedUpsilon,ind]=sort(Upsilon,'descend');
W = W(ind,:);
Y = Y(ind,:);
