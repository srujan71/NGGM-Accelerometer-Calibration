clear
close all
clc
%%
% phi = (-90:1:90)';
% phi = deg2rad(phi);
phi = (-1:0.01:1)';

colat = pi/2 - phi;

data(size(phi, 1)) = struct('P0', [], 'P1', [], 'P2', [], 'P3', [], 'P4', []);

for i = 1:size(phi, 1)
    [P, dP, ddP] = legendre_polynomial_ext(4, colat(i));

    data(i).P0 = P(1, 1);
    data(i).P1 = P(2, 1);
    data(i).P2 = P(3, 1);
    data(i).P3 = P(4, 1);
    data(i).P4 = P(5, 1);
end

data = data';

%%
figure()
plot(phi, [data.P0])
hold on
plot(phi, [data.P1])
plot(phi, [data.P2])
plot(phi, [data.P3])
plot(phi, [data.P4])
hold off
legend("P0", "P1", "P2", "P3", "P4")
grid on
xlabel('\phi (degrees)');
ylabel('Legendre Polynomial Value');

