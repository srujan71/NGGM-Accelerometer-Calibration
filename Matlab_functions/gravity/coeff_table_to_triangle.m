function [C,S] = coeff_table_to_triangle(T,varargin)
% [C,S] = coeff_table_to_triangle(T)
%
% Table is: [degree order C(degree,order) S(degree,order)]

if nargin == 1
    n_max = max(T(:,1));
else
    n_max = max(max(T(:,1)),varargin{1});
end

C = zeros(n_max+1,n_max+1);
S = zeros(n_max+1,n_max+1);

for k = 1:size(T,1)
    n = T(k,1);
    m = T(k,2);
    C(n+1,m+1) = T(k,3);
    if m > 0
        S(n+1,m+1) = T(k,4);
    end
end
