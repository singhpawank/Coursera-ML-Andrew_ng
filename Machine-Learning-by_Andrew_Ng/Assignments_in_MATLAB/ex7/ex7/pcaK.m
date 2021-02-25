function K = pcaK(X)
%THis function will return optimal value of k
% X = normalised matrix, m*n

% Initial value of K
K = 100;

% Useful values
[m, n] = size(X);

sigma = X'*X / m;
[U, S, V] = svd(sigma);

total = trace(S);

for k = K:n
    totalK = trace(S(1:k,1:k));
    ratio = totalK/total;
  
    if ratio >= 0.99
        break
    end
end
K = k;
end

