function calculate_gravity_gradients(l_max, ecrf_path, RBV_path, save_path)
    GM = 0.39860044150e+15;
    R = 0.63781363000e+07;


    pos = load(ecrf_path);

    gravity_table = load("C:/Users/sruja/Desktop/Astro_class_code/nggm/Matlab_functions/gravity/GOCO05c.csv");
    gravity_table = gravity_table(:, 1:4);

    rot_vector_BV = load(RBV_path);


    coeffs_vector = coeff_table_to_vec(gravity_table, l_max);

    [Vxx,Vxy,Vxz,Vyy,Vyz,Vzz] = gravity_gradient(coeffs_vector, pos, GM, R, 0, l_max);

    n = size(pos, 1);
    V_Vxx = zeros(n, 1);
    V_Vyy = zeros(n, 1);
    V_Vzz = zeros(n, 1);
    V_Vxy = zeros(n, 1);
    V_Vxz = zeros(n, 1);
    V_Vyz = zeros(n, 1);

    R = [1, 0, 0;
         0, 1, 0;
         0, 0, -1];
    for i=1:n
        V = [Vxx(i), Vxy(i), Vxz(i);
             Vxy(i), Vyy(i), Vyz(i);
             Vxz(i), Vyz(i), Vzz(i)];

        V_V = R * V * R';

        R_BV = reshape(rot_vector_BV(i, :), 3,3)';

        V_V = R_BV * V_V * R_BV';

        V_Vxx(i) = V_V(1,1);
        V_Vyy(i) = V_V(2,2);
        V_Vzz(i) = V_V(3,3);
        V_Vxy(i) = V_V(1,2);
        V_Vxz(i) = V_V(1,3);
        V_Vyz(i) = V_V(2,3);

    end

    gravity_gradients.Vxx = V_Vxx;
    gravity_gradients.Vyy = V_Vyy;
    gravity_gradients.Vzz = V_Vzz;
    gravity_gradients.Vxy = V_Vxy;
    gravity_gradients.Vyz = V_Vyz;
    gravity_gradients.Vxz = V_Vxz;

    name = sprintf("gravity_gradients_%d.mat", l_max);

    save(fullfile(save_path, name), 'gravity_gradients');

end

