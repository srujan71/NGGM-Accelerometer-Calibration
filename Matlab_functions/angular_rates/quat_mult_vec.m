function [s] = quat_mult_vec(p,q,varargin)
% [s] = quat_mult_vec(p,q)
% [s] = quat_mult_vec(p,q,type)
%
% returns the product of two quaternions: s = p*q
% (elementwise "vector times vector" or "vector times scalar" or "scalar times vector")
%
% type = 
%    'row' ... one quaternion in each row
%    'col' ... one quaternion in each column
%
% q = [q0 q1 q2 q3]
%
% q0 ... real part
% q1,q2,q3 ... imaginary part


if nargin == 2
    type = 'undef';
elseif nargin == 3
    type = varargin{1};
    if ~(strcmp(type,'row') || strcmp(type,'col'))
        error('Third argument (type) must be ''row'' or ''col''.')
    else
        if strcmp(type,'row') && (size(p,2) ~= 4 || size(q,2) ~= 4)
            error('Quaternions not stored in rows.')
        end
        if strcmp(type,'col') && (size(p,1) ~= 4 || size(q,1) ~= 4)
            error('Quaternions not stored in columns.')
        end
    end
end


if size(p,1) == 4 && size(p,2) == 4 && strcmp(type,'undef')
    error('4 x 4 matrix detected and no type defined (type must be ''row'' or ''col'')')
end
if size(q,1) == 4 && size(q,2) == 4 && strcmp(type,'undef')
    error('4 x 4 matrix detected and no type defined (type must be ''row'' or ''col'')')
end

%--------------------------------------------------------------------------
%  "vector times vector"
%--------------------------------------------------------------------------

% one quaternion in each column of 'p' and 'q'
if size(p,1) == 4 && size(p,2) ~= 1 && size(q,1) == 4 && size(q,2) ~= 1 && (strcmp(type,'col') || strcmp(type,'undef'))
    
    s = [p(1,:).*q(1,:) - p(2,:).*q(2,:) - p(3,:).*q(3,:) - p(4,:).*q(4,:)
         p(2,:).*q(1,:) + p(1,:).*q(2,:) - p(4,:).*q(3,:) + p(3,:).*q(4,:)
         p(3,:).*q(1,:) + p(4,:).*q(2,:) + p(1,:).*q(3,:) - p(2,:).*q(4,:)
         p(4,:).*q(1,:) - p(3,:).*q(2,:) + p(2,:).*q(3,:) + p(1,:).*q(4,:)];

% one quaternion in each row of 'p' and 'q'
elseif size(p,1) ~= 1 && size(p,2) == 4 && size(q,1) ~= 1 && size(q,2) == 4 && (strcmp(type,'row') || strcmp(type,'undef'))
    
    s = [p(:,1).*q(:,1) - p(:,2).*q(:,2) - p(:,3).*q(:,3) - p(:,4).*q(:,4) ...
         p(:,2).*q(:,1) + p(:,1).*q(:,2) - p(:,4).*q(:,3) + p(:,3).*q(:,4) ...
         p(:,3).*q(:,1) + p(:,4).*q(:,2) + p(:,1).*q(:,3) - p(:,2).*q(:,4) ...
         p(:,4).*q(:,1) - p(:,3).*q(:,2) + p(:,2).*q(:,3) + p(:,1).*q(:,4)];

%--------------------------------------------------------------------------
%  "vector times scalar"
%--------------------------------------------------------------------------

% one quaternion in each column of 'p' whereas 'q' is a single quaternion
elseif size(p,1) == 4 && size(p,2) ~= 1 && ((size(q,1) == 1 && size(q,2) == 4) || (size(q,1) == 4 && size(q,2) == 1)) && (strcmp(type,'col') || strcmp(type,'undef'))
    
    s = [p(1,:)*q(1) - p(2,:)*q(2) - p(3,:)*q(3) - p(4,:)*q(4)
         p(2,:)*q(1) + p(1,:)*q(2) - p(4,:)*q(3) + p(3,:)*q(4)
         p(3,:)*q(1) + p(4,:)*q(2) + p(1,:)*q(3) - p(2,:)*q(4)
         p(4,:)*q(1) - p(3,:)*q(2) + p(2,:)*q(3) + p(1,:)*q(4)];
     
% one quaternion in each row of 'p' whereas 'q' is a single quaternion
elseif size(p,1) ~= 1 && size(p,2) == 4 && ((size(q,1) == 1 && size(q,2) == 4) || (size(q,1) == 4 && size(q,2) == 1)) && (strcmp(type,'row') || strcmp(type,'undef'))
    
    s = [p(:,1)*q(1) - p(:,2)*q(2) - p(:,3)*q(3) - p(:,4)*q(4) ...
         p(:,2)*q(1) + p(:,1)*q(2) - p(:,4)*q(3) + p(:,3)*q(4) ...
         p(:,3)*q(1) + p(:,4)*q(2) + p(:,1)*q(3) - p(:,2)*q(4) ...
         p(:,4)*q(1) - p(:,3)*q(2) + p(:,2)*q(3) + p(:,1)*q(4)];
     
%--------------------------------------------------------------------------
%  "scalar times vector"
%--------------------------------------------------------------------------

% one quaternion in each column of 'q' whereas 'p' is a single quaternion
elseif ((size(p,1) == 1 && size(p,2) == 4) || (size(p,1) == 4 && size(p,2) == 1)) && size(q,1) == 4 && size(q,2) ~= 1 && (strcmp(type,'col') || strcmp(type,'undef'))
    
    s = [p(1)*q(1,:) - p(2)*q(2,:) - p(3)*q(3,:) - p(4)*q(4,:)
         p(2)*q(1,:) + p(1)*q(2,:) - p(4)*q(3,:) + p(3)*q(4,:)
         p(3)*q(1,:) + p(4)*q(2,:) + p(1)*q(3,:) - p(2)*q(4,:)
         p(4)*q(1,:) - p(3)*q(2,:) + p(2)*q(3,:) + p(1)*q(4,:)];
     
     

% one quaternion in each row of 'q' whereas 'p' is a single quaternion
elseif ((size(p,1) == 1 && size(p,2) == 4) || (size(p,1) == 4 && size(p,2) == 1)) && size(q,1) ~= 1 && size(q,2) == 4 && (strcmp(type,'row') || strcmp(type,'undef'))
    
    s = [p(1)*q(:,1) - p(2)*q(:,2) - p(3)*q(:,3) - p(4)*q(:,4) ...
         p(2)*q(:,1) + p(1)*q(:,2) - p(4)*q(:,3) + p(3)*q(:,4) ...
         p(3)*q(:,1) + p(4)*q(:,2) + p(1)*q(:,3) - p(2)*q(:,4) ...
         p(4)*q(:,1) - p(3)*q(:,2) + p(2)*q(:,3) + p(1)*q(:,4)];
     
%--------------------------------------------------------------------------
%  "scalar times scalar"
%--------------------------------------------------------------------------
   
elseif size(p,1) == 1 && size(p,2) == 4 && size(q,1) == 1 && size(q,2) == 4 && (strcmp(type,'row') || strcmp(type,'undef'))
    
    s = [p(1)*q(1) - p(2)*q(2) - p(3)*q(3) - p(4)*q(4) ...
         p(2)*q(1) + p(1)*q(2) - p(4)*q(3) + p(3)*q(4) ...
         p(3)*q(1) + p(4)*q(2) + p(1)*q(3) - p(2)*q(4) ...
         p(4)*q(1) - p(3)*q(2) + p(2)*q(3) + p(1)*q(4)];
    
elseif size(p,1) == 4 && size(p,2) == 1 && size(q,1) == 4 && size(q,2) == 1 && (strcmp(type,'col') || strcmp(type,'undef'))

    s = [p(1)*q(1) - p(2)*q(2) - p(3)*q(3) - p(4)*q(4)
         p(2)*q(1) + p(1)*q(2) - p(4)*q(3) + p(3)*q(4)
         p(3)*q(1) + p(4)*q(2) + p(1)*q(3) - p(2)*q(4)
         p(4)*q(1) - p(3)*q(2) + p(2)*q(3) + p(1)*q(4)];
     
else
    error('Size of q and p do not match.')
end






