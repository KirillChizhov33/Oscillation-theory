function dXdtheta = asymDuffingFunc(theta, X, omega)
    dx1=X(2)./omega;
    dx2=(X(1)-X(1)^3)./omega;
    dXdtheta = [dx1; dx2];
end