function [w] = quat2rate_vec(q,t,dt)
% [w] = quat2rate_vec(q,t,dt) ... t = time (e.g. GPS seconds)
%
% q = [q0 q1 q2 q3] ... q0 = real part / q1, q2, q3 = imaginary part
% w = [wx wy wz]
%
% first time derivative is approximated by splines

% make quaternions continuous
for n = 2:size(q,1)
    if q(n,:)*q(n-1,:)' < 0
        q(n,:) = -q(n,:);
    end
end

% in principle, this has to be calculated: [0; w] = 2*quat_mult(quat_conj(q),dq)

% first time derivative
dq = squeeze(diff(interp1(t,q,[t-dt t+dt],'spline','extrap'),1,2))/(2*dt);

W = quat_mult_vec(quat_conj_vec(q),dq);

w = 2 * W(:,2:4);
