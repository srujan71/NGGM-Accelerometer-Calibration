function [x,l_min,l_max] = coeff_table_to_vec(T,varargin)


if nargin > 1
    l_max = varargin{1};
else
    l_max = max(T(:,1));
end

[C,S] = coeff_table_to_triangle(T,l_max);

x = sh_coeff_triangles_to_vec(C(1:l_max+1,1:l_max+1),S(1:l_max+1,1:l_max+1),0,l_max);