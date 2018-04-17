% An example MATLAB script for orderingICA function.
% It extracts an exponentially distributed variable and a uniformly
% distributed one in the unique order from eight Gaussian noises.
M=10000;
S=[exprnd(1,1,M)-1;(rand(1,M)-0.5)*sqrt(12);randn(8,M)];
A=randn(size(S,1));
X=A*S;
[W,Y]=orderingICA(X);
% abs(W*A) is expected to be eye(2,10).
disp(abs(W*A));