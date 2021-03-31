% Parallel Ordering ICA with MATLAB Implementation using Parallel Computing Toolbox
% Author: Yoshitatsu Matsuda    Apr. 1 2021

function [W,Y] =  ParallelOrderingICA(X)
% W - separating matrix
% Y - estimated source
% X - observed signal (mixture)

[N,M]	= size(X);
L = feature('numcores');
Upsilon = @(u) -2*log((u+2)/2) + u;
epsilon = 10^-6;
maxIteration = 30;
% N - number of observed signals (N) 
% M - sample size
% L - number of multiple candidates
% Upsilon - criterion of non-Gaussianity
% epsilon - threshold of iterations in fast ICA
% maxIteration - max number of iterations in fast ICA

% pre-whitening
X = X - mean(X')' * ones(1,M);
[V,E] = eig((X*X')/M);
scales = diag(E);
B = diag(1./sqrt(scales))*V';
X = B*X;

% main process;
W = [];
for i = 1:N
    [w,alpha] = multiple_estimation();
    W = [W;w];
end

% results
Y = W*X;
W = W*B;
alphas = mean(Y'.^4)-3;
Upsilons = alphas-2*log((alphas+2)/2);
[sortedUpsilons,ind]=sort(Upsilons,'descend');
W = W(ind,:);
Y = Y(ind,:);

%fast ICA for a single candidate
    function [v,alpha] = single_estimation()
        v = randn(1,N);
        for t = 1:maxIteration
            pv = v;
            Y = v*X;
            v = (X * ((Y') .^ 3))' / M - 3 * v;
            if i > 1
                v = v - (W*v')'*W;
            end
            v = v/sqrt(v*v');
            if norm(pv - v)<epsilon | norm(pv + v)<epsilon
                break;
            end
        end
        Y = v*X;
        alpha = mean(Y.^4)-3;
    end

%selection from multiple candidates
    function [v,alpha] = multiple_estimation()
        U = zeros(L,1);
        alphas = zeros(L,1);
        vs = zeros(L,N);
        fn = @single_estimation;
        parfor c = 1:L
            [vc,alphac]=feval(fn);
            vs(c,:) = vc;
            alphas(c) = alphac;
            U(c)=Upsilon(alphac);
        end
        [tmp,ind]=max(U);
        v = vs(ind,:);
        alpha = alphas(ind);
    end
end
