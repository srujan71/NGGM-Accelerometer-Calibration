function [map_c, map_s] = sh_maps( l_min, l_max )

map_c = zeros(l_max+1,l_max+1);
map_s = zeros(l_max+1,l_max+1);

counter = 0;
for m = 0:l_max
	for l = max(m,l_min):l_max
        counter = counter+1;
        map_c(l+1,m+1) = counter;
        if m > 0
            counter = counter+1;
            map_s(l+1,m+1) = counter;
        end
    end
end
