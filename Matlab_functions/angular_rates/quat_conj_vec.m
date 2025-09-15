function [qc] = quat_conj_vec(q,varargin)
% [qc] = quat_conj_vec(q)
% [qc] = quat_conj_vec(q,type)
%
% conjugates a quaternion
%
% (this is equivalent to changing the direction of the rotation; in the
% same sense as transposing a rotation matrix)
%
% type = 
%    'row' ... one quaternion in each row
%    'col' ... one quaternion in each column
%
% q = [q0(1) q1(1) q2(1) q3(1)]
%       ...   ...   ...   ...
%     [q0(N) q1(N) q2(N) q3(N)]
%
% q0 ... real part
% q1,q2,q3 ... imaginary part

if nargin == 1
    type = 'undef';
else
    type = varargin{1};
    if ~(strcmp(type,'row') || strcmp(type,'col'))
        error('Third argument (type) must be ''row'' or ''col''.')
    else
        if strcmp(type,'row') && size(q,2) ~= 4
            error('Quaternions not stored in rows.')
        end
        if strcmp(type,'col') && size(p,1) ~= 4
            error('Quaternions not stored in columns.')
        end
    end
end

if size(q,1) == 4 && size(q,2) == 4 && strcmp(type,'undef')
    error('4 x 4 matrix detected and no type defined (type must be ''row'' or ''col'')')
end 

if size(q,1) == 4 && (strcmp(type,'col') || strcmp(type,'undef'))
    qc = [q(1,:); -q(2:4,:)];
elseif size(q,2) == 4  && (strcmp(type,'row') || strcmp(type,'undef'))
    qc = [q(:,1) -q(:,2:4)];
else
    error('Size of q and type (row o r column) do not match.')
end
