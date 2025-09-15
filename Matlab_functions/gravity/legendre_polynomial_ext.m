function [P,dP,ddP] = legendre_polynomial_ext( l_max, colat )
% [P,dP,ddP] = legendre_polynomial_ext( l_max, colat )
%
% l_max ... maximum degree
% colat ... colatitude in radians
% lat ..... latitude in radians
% P ....... normalized Legendre polynomials
% dP ...... first derivative wrt. to colatitude --> nabla P(cos(theta)) / nabla theta
% ddP ..... second derivative wrt. to colatitude --> nabla^2 P(cos(theta)) / nabla theta^2
%           (nabla = differential operator)

lat = pi/2 - colat;

ct = sin(lat); % note the switch from latitude to colatitude
st = cos(lat);

p  = zeros(l_max+1,1);
P = zeros(l_max+1,l_max+1);
dP = zeros(l_max+1,l_max+1);
ddP = zeros(l_max+1,l_max+1);

r = zeros(2*l_max+1,1);

% pre-compute square-roots
for i=1:(2*l_max+1)
    r(i) = sqrt(i);
end

% init
p(1) = 1.0;
if l_max > 0
    p(2) = r(3)*st;
end

% pre-compute factors: P(l+1,l+1)
for l = 2:l_max
    l2 = l*2;
    fac = r(l2+1)/r(l2);
    p(l+1) = p(l)*fac*st;
end


% compute Legendre polynomials
for m = 0:l_max
	l = m;
    P(l+1,m+1) = p(m+1);
 	if l < l_max
       	l = m+1;
        l2 = l*2;
        fac = r(l2+1);
        P(l+1,m+1) = P(l,m+1)*ct*fac;

        for l = m+2:l_max
            l2 = l*2;
            fac1 = r(l2+1)/r(l-m)/r(l+m);
            fac2 = r(l2-1);
            fac3 = r(l-m-1)*r(l+m-1)/r(l2-3);
            P(l+1,m+1) = fac1*(P(l,m+1)*fac2*ct-P(l-1,m+1)*fac3);
        end
    end
end


% compute first derivative
dP(1,1) = 0;
if l_max > 0
    dP(2,2) = sqrt(3)*ct;
end
for m = 2:l_max
    dP(m+1,m+1) = sqrt((2*m+1)/(2*m)) * (ct*P(m,m)+st*dP(m,m));
end

for m = 0:l_max-1
    dP(m+2,m+1) = sqrt(2*m+3) * (-st*P(m+1,m+1)+ct*dP(m+1,m+1));
end
    
for m = 0:l_max-2
    for l = m+2:l_max
        dP(l+1,m+1) =   sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-st*P(l,m+1)+ct*dP(l,m+1)) ...
                      - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * dP(l-1,m+1);
    end
end

% compute second derivative
ddP(1,1) = 0;
if l_max > 0
    ddP(2,2) = -sqrt(3)*st;
end
for m = 2:l_max
    ddP(m+1,m+1) = sqrt((2*m+1)/(2*m)) * (-st*P(m,m)+ct*dP(m,m)+ct*dP(m,m)+st*ddP(m,m));
end

for m = 0:l_max-1
    ddP(m+2,m+1) = sqrt(2*m+3) * (-ct*P(m+1,m+1)-st*dP(m+1,m+1)-st*dP(m+1,m+1)+ct*ddP(m+1,m+1));
end
    
for m = 0:l_max-2
    for l = m+2:l_max
        ddP(l+1,m+1) =   sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-ct*P(l,m+1)-st*dP(l,m+1)-st*dP(l,m+1)+ct*ddP(l,m+1)) ...
                      - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * ddP(l-1,m+1);
    end
end

