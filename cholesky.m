%part (a)
function L= myChol(A)
    n=size(A,1)
    
    L = zeros(n)
    
    for j = 1 : n
        L(j,j) = (A(j,j) - sum(L(j,1:j-1).^2))^(1/2); % 2j-1 flops
        for i = j + 1 : n
            L(i,j) = (A(i,j) - sum(L(i,1:j-1).*L(j,1:j-1)))/L(j,j); % 2j-1 flops
        end
    end
    d = diag(L)
    nonzeros(d)
    if isreal(d) & isequal(nonzeros(d) , d) 
        L
    else
        fprintf('Your matrix is not positive definite!')
    end
end

%part (b)

n=10
A_tilde=rand(n)
A= A_tilde+A_tilde'

myChol(A)
lam=eig(A)

%norm of difference
%norm = chol(A,'lower') - myChol(A)


%part (c)
A_spd= A_tilde'*A_tilde
norm = chol(A_spd,'lower') - myChol(A_spd)
