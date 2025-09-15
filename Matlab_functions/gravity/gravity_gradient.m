function [Vxx,Vxy,Vxz,Vyy,Vyz,Vzz] = gravity_gradient(coeff,pos,GM,R,n_min,n_max)
%
% Vxx,Vxy,Vxz,Vyy,Vyz,Vzz ... gradients in LNOF (left hand system)
% coeff ... spherical harmonic coefficients
% pos ..... positions in EFRF (pos = [x y z])
% GM ...... gravitational constant times mass of Earth
% R ....... reference radiuss/equatorial radius of Earth
% n_min, n_max ... minimum/maximum spherical harmonic degree

Vxx = zeros(size(pos,1),1);
Vxy = zeros(size(pos,1),1);
Vxz = zeros(size(pos,1),1);
Vyy = zeros(size(pos,1),1);
Vyz = zeros(size(pos,1),1);
Vzz = zeros(size(pos,1),1);

for k = 1:size(pos,1)
    
    % divide by the reference radius to improve the numerical precision (xa ~= ya ~= za ~= 1.xxx)
    xa = pos(k,1)/R;
    ya = pos(k,2)/R;
    za = pos(k,3)/R;
		
    p = sqrt(xa*xa + ya*ya);
    r = sqrt(xa*xa + ya*ya + za*za);
        
    sin_lambda = ya/p;
    cos_lambda = xa/p;
    cos_phi = p/r;
    sin_phi = za/r;
    tan_phi = sin_phi/cos_phi;
    
	p = R * p;
	r = R * r;
        
	GMor3 = GM/r/r/r;
    
    % sin(phi) = cos(theta) --> Legendre function works with colatitude (= theta)
    colat = acos(sin_phi);
    
    lambda = atan2(sin_lambda,cos_lambda);
				
    % pre-compute associated Legendre functions + derivatives
    [P,dP,ddP] = legendre_polynomial_ext( n_max, colat );
		
	% compute gravity gradients
    counter = 1;
		
    %keyboard
    
    % order m = 0
    for n = n_min:n_max % loop over parameters
        aorn_GMor3 = (R/r)^n * GMor3;
			
        Pnm = P(n+1,1);
        dPnm = dP(n+1,1);
        ddPnm = ddP(n+1,1);
			
        cnm = coeff(counter);
        counter = counter + 1;
			
        Vxx(k) = Vxx(k) + aorn_GMor3 * (ddPnm - (n+1) * Pnm) * cnm;
        % Vxy(k) = Vxy(k) + 0.0;
        Vxz(k) = Vxz(k) + (n+2) * aorn_GMor3 * dPnm * cnm;
        Vyy(k) = Vyy(k) - aorn_GMor3 * ((n+1) * Pnm - tan_phi * dPnm) * cnm;
        % Vyz(k) = Vyz(k) + 0.0;
        Vzz(k) = Vzz(k) + (n+1)*(n+2) * aorn_GMor3 * Pnm * cnm;
    end
    
    % order greater than zero: m > 0
    for m = 1:n_max
        cp = cos(m*lambda);
        sp = sin(m*lambda);
			
        for n = max(n_min,m):n_max
            aorn_GMor3 = (R/r)^n * GMor3;

            aorn_GMor3_cm = aorn_GMor3 * cp;
            aorn_GMor3_sm = aorn_GMor3 * sp;

            Pnm = P(n+1,m+1);
            dPnm = dP(n+1,m+1);
            ddPnm = ddP(n+1,m+1);
			    
            cnm = coeff(counter);
            counter = counter + 1;
            
            snm = coeff(counter);
            counter = counter + 1;
			    
            h = ddPnm - (n+1) * Pnm;
            Vxx(k) = Vxx(k) + aorn_GMor3_cm * h * cnm;
            Vxx(k) = Vxx(k) + aorn_GMor3_sm * h * snm;

            h = m * (tan_phi * Pnm - dPnm) / cos_phi;
            Vxy(k) = Vxy(k) - aorn_GMor3_sm * h * cnm;
            Vxy(k) = Vxy(k) + aorn_GMor3_cm * h * snm;

            h = (n+2) * dPnm;
            Vxz(k) = Vxz(k) + aorn_GMor3_cm * h * cnm;
            Vxz(k) = Vxz(k) + aorn_GMor3_sm * h * snm;
				
            h = - ( (n+1 + (m/cos_phi)^2) * Pnm - tan_phi * dPnm );
            Vyy(k) = Vyy(k) + aorn_GMor3_cm * h * cnm;
            Vyy(k) = Vyy(k) + aorn_GMor3_sm * h * snm;

            h = (n+2)*m * Pnm / cos_phi;
            Vyz(k) = Vyz(k) + aorn_GMor3_sm * h * cnm;
            Vyz(k) = Vyz(k) - aorn_GMor3_cm * h * snm;

            h = (n+1)*(n+2) * Pnm;
            Vzz(k) = Vzz(k) + aorn_GMor3_cm * h * cnm;
            Vzz(k) = Vzz(k) + aorn_GMor3_sm * h * snm;
        end
    end
end








