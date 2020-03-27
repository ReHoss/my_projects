function[x,error,iter,flag]= cgp(A,x,k,P,tol)
n= size(A,1);
flag = 0;
iter = 0;
knorm = norm(k);
if (knorm == 0) %Pour éviter les divisions par zéro, si k est nul, 
    knorm = 1; %on considère l'erreur absolue au lieu de l'erreur relative.
end
r = k - A*x;
error = norm(r)/knorm;
if (error < tol ) return 
end
for iter = 1:n
    z = P \ r; %on sous-traite le calcul de P*z=r, avec une fonction de Matlab
    rho = (r'*z); 
    if iter > 1
        b = - rho /rho1;
        p = z - b*p;
    else
        p= z;
    end
    q = A*p; 
    a = rho/ (p'*q);
    x = x + a*p;
    r = r - a*q;
    error=norm(r)/knorm;
    if (error <= tol)
        break
    end
    rho1 = rho; %rho1 permet de stocker la valeur de rho à l'étape antérieure
end
if (error <= tol)
        flag = 1;
end
end

