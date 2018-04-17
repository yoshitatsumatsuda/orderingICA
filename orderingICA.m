% Ordering ICA with MATLAB Implementation
% Author: Yoshitatsu Matsuda    Apr. 17 2018

function [W,Y] =  orderingICA(X)
% W - separating matrix
% Y - estimated source
% X - observed signal (mixture)

L = 100;
[N,M]	= size(X);
Upsilon = @(u) -2*log((u+2)/2) + u; 
% L - number of multiple candidates
% N - number of observed signals (N) 
% M - sample size
% Upsilon - criterion of non-Gaussianity

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
    if Upsilon(alpha) > (2*(N+1-i)*(N+2-i))/M,
        W = [W;w];
    else
        break;
    end
end

% results
Y = W*X;
W = W*B;

%fast ICA for a single candidate
    function [v,alpha] = single_estimation()

        v = randn(1,N);
        for t = 1:100
            pv = v;
            Y = v*X;
            v = (X * ((Y') .^ 3))' / M - 3 * v;
            for j = 1:(i-1)
                v = v - (W(j,:)*v')*W(j,:);
            end
            v = v/sqrt(v*v');
            if norm(pv - v)<10^-6 | norm(pv + v)<10^-6
                break;
            end
        end
        
        Y = v*X;
        alpha = mean(Y.^4)-3;
    end

%selection from multiple candidates
    function [v,alpha] = multiple_estimation()
        U = -1;
        for c = 1:L
            [vc,alphac]=single_estimation();
            if(Upsilon(alphac)>U)
              U = Upsilon(alphac);
              v = vc;
              alpha = alphac;
            end
        end
    end
end
