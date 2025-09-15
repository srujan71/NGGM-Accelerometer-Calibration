function [x] = sh_coeff_triangles_to_vec(C,S,l_min,l_max)
% [x] = sh_coeff_triangles_to_vec(C,S,l_min,l_max)

x = zeros((l_max+1)^2-l_min^2,1);

[map_c, map_s] = sh_maps( l_min, l_max );

for l = l_min:l_max
    for m = 0:l
        x(map_c(l+1,m+1)) = C(l+1,m+1);
        if m > 0
            x(map_s(l+1,m+1)) = S(l+1,m+1);
        end
    end
end

