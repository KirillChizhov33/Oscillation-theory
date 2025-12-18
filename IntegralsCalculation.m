function integrals_calculation
% �?сходное ДУ: x'' - x + x^3 = p1*x*x' + p2*sin(t)
% H(x, y) = y^2/2 − x^2/2 + x^4/4 = h, h ≥ −0.25.
h = -0.25:0.01:0 % Значения h первого интеграла, соотв. траекториям внутри сепаратрисы
x=[];
for k=1:length(h)
    poly = [1/4 0 -1/2 0 -h(k)]; % Полином, соответствующий 1му интегралу  
    p_roots = roots(poly); % Корни полинома
    j=1;
    im = imag(p_roots);
    re = real(p_roots);
    for i=1:4
    if (im(i) == 0)
       x(j,k) = re(i);
       j=j+1;
    end
     end
end 
x=sort(x);
x1=x(1,:);
x2=x(2,:);
x3=x(3,:);
x4=x(4,:);

% Вычисление периода
T=zeros(1,length(h));
fun =@(s,hh,aa)1./sqrt(2*(hh-s.^4./4+1/2*s.^2));
for i=1:length(h)
    T(i)=2*integral(@(s)fun(s,h(i),0),x3(i),x4(i));
end
hold on;

% Вычисление частоты 1м способом
omega = 2*pi./T(1:(length(T)));
%plot(h,omega,'blue');

% Вычисление частоты 2м способом
omega = zeros(1,length(h));
rho = 2*sqrt(1+4*h)./(1+sqrt(1+4*h));
[k1, e1] = ellipke(rho);
for i=1:length(h)
    omega(i) = (pi.*sqrt(1+sqrt(1+4*h(i))))./(sqrt(2)*k1(i));
end
%plot(h,omega,'red');

B=zeros(1,length(h));
C1=zeros(1,length(h));
C2=zeros(1,length(h));
C3=zeros(1,length(h));
C4=zeros(1,length(h));
C5=zeros(1,length(h));
C6=zeros(1,length(h));
C7=zeros(1,length(h));
C8=zeros(1,length(h));
%T=real(T);
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
for i=2:length(h)
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    % theta = omega(i)*t;
    % p = 1
    f1 = trapz(theta, X(:,1).*X(:,2).^2);
    B(i) = (f1)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    f2 = trapz(theta, sin(theta).*X(:,2));
    C1(i) = (f2)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    f3 = trapz(theta, cos(theta).*X(:,2));
    C2(i) = (f3)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 2
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f4 = trapz(theta, sin(2.*theta).*X(:,2));
    C3(i) = (f4)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 3
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f5 = trapz(theta, sin(3.*theta).*X(:,2));
    C4(i) = (f5)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 4
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f6 = trapz(theta, sin(4.*theta).*X(:,2));
    C5(i) = (f6)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 5
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f7 = trapz(theta, sin(5.*theta).*X(:,2));
    C6(i) = (f7)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 6
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f8 = trapz(theta, sin(6.*theta).*X(:,2));
    C7(i) = (f8)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
    
    % p = 7
    [theta X] = ode45(@(theta, X)asymDuffingFunc(theta, X, omega(i)), [0 2*pi], [x4(i) 0],opts);
    f9 = trapz(theta, sin(7.*theta).*X(:,2));
    C8(i) = (f9)./(2*pi*omega(i)); % (p1 * omega(i)* f1)./(2*pi);
end
plot(h,B, 'black');
E = linspace(-0.249, -1e-6, 400);
DH = (sqrt(2)/8) * (1 + 4*E);
% Вывод графиков зависимостей Ci(h)
plot(E, DH), grid on
plot(h,C1, 'red');
plot(h,C2, 'green');
plot(h,C3, 'blue');
plot(h,C4, 'black');
plot(h,C5, 'black');
plot(h,C6, 'blue');
plot(h,C7, 'yellow');
plot(h,C8, 'red');
end

function dXdtheta = asymDuffingFunc(theta, X, omega)
    dx1=X(2)./omega;
    dx2=(X(1)-X(1)^3)./omega;
    dXdtheta = [dx1; dx2];
end