function calculate_angular_rates()

dt = 1e-3;

state = load("C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/Output/Orbit_data/state_history.dat");
t = state(:,1);
% Supply the quaternion from body to inertial to get the correct results. Christian's code has a formatting issue. It should be the other way around.
q = load("C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/Output/Orbit_data/quaternions_history_RIB.txt");
w = quat2rate_vec(q,t,dt);
dw = squeeze(diff(interp1(t,w,[t-dt t+dt],'spline','extrap'),1,2))/(2*dt);

save(fullfile("C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/Output/Orbit_data","angular_rates.mat"),'w')
save(fullfile("C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/Output/Orbit_data","angular_accelerations.mat"),'dw')
