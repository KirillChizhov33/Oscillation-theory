function duffing_poincare_fixedpoints_nox_fmt
% Поиск фикс-точек Poincar? (t=0, T=2?) и их мультипликаторов
% БЕЗ Optimization Toolbox. Формат вывода: 5 знаков после запятой, комплексные полностью.

% -------- параметры модели --------
p1 = 0.9;
p2 = 0.522;
T  = 2*pi; t0 = 0;

% -------- численные настройки --------
optsODE_fast = odeset('AbsTol',1e-7,'RelTol',1e-5,'MaxStep',0.25); % PASS 1 (быстро)
optsODE_ref  = odeset('AbsTol',1e-10,'RelTol',1e-8,'MaxStep',0.05); % PASS 2 (точно)

newton_tol   = 1e-6;    % PASS 1
newton_tol2  = 1e-9;    % PASS 2
newton_itmax = 40;
armijo_c     = 1e-4;

% стартовая сетка (сфокусировано)
xs = [-1.6 -1.3 -1.1 -0.9 -0.7  -0.2  0  0.2  0.7 0.9 1.1 1.3 1.6];
ys = linspace(-0.6,0.6,10);
[XX,YY] = ndgrid(xs,ys);
starts = [XX(:) YY(:)];

tol_merge  = 1e-4;  % PASS 1 склейка
tol_merge2 = 5e-5;  % финальная склейка

% -------- PASS 1 --------
Z = [];
for k=1:size(starts,1)
    z0 = starts(k,:).';
    [z1, ok] = newton_P(z0, @(z) F_and_J(z,T,p1,p2,t0,optsODE_fast), newton_tol, newton_itmax, armijo_c);
    if ok
        if isempty(Z) || all(vecnorm(Z.' - z1(:),2,1) > tol_merge)
            Z = [Z; z1(:).']; %#ok<AGROW>
        end
    end
end
if isempty(Z)
    fprintf('PASS 1: ничего не найдено. Расширьте xs/ys или ослабьте допуски.\n');
    return
end

% --- Symmetry completion: refine Rz for each found z ---
Z = dedupe_fixpoints(Z, tol_merge2);
Zsym_add = [];
for i = 1:size(Z,1)
    z    = Z(i,:).';
    zsym = [-z(1); z(2)];                  % Rz = (-x, y)
    [z2, ok2] = newton_P(zsym, @(zz) F_and_J(zz,T,p1,p2,t0,optsODE_ref), newton_tol2, 60, 1e-4);
    if ok2
        Zsym_add = [Zsym_add; z2(:).'];    %#ok<AGROW>
    end
end
% слить и удалить дубли
Z = dedupe_fixpoints([Z; Zsym_add], tol_merge2);

% -------- PASS 2 + печать с форматом --------
fprintf('Найдено кандидатов после PASS 1: %d\n', size(Z,1));
fprintf('%3s  %10s %10s   %12s          %26s        %15s   %s\n', ...
        '#','x','y','||P(z)-z||','eigs(?1, ?2)','|?1|, |?2|','Type');

for i=1:size(Z,1)
    zinit = Z(i,:).';
    [z, ok] = newton_P(zinit, @(zz) F_and_J(zz,T,p1,p2,t0,optsODE_ref), newton_tol2, 60, armijo_c);
    [F,J]  = F_and_J(z,T,p1,p2,t0,optsODE_ref);   % J = DP - I
    DP     = J + eye(2);
    ev     = eig(DP);
    mods   = abs(ev).';

    % форматированные строки
    zxs = sprintf('%.5f', z(1));
    zys = sprintf('%.5f', z(2));
    ress = sprintf('%.5e', norm(F));                 % маленькие числа удобнее в e-формате
    lam1s = cplx5(ev(1));                            % комплекс полностью с 5 знаками
    lam2s = cplx5(ev(2));
    mod1s = sprintf('%.5f', mods(1));
    mod2s = sprintf('%.5f', mods(2));
    typ   = classify_point(ev, 5e-3);

    fprintf('%3d  %10s %10s   %12s   (%s, %s)   [%7s, %7s]   %s%s\n', ...
        i, zxs, zys, ress, lam1s, lam2s, mod1s, mod2s, typ, tern(ok,'','  (Newton not strict)'));
end
end

% ===================== вспомогательные =====================

function s = cplx5(z)
% Строка комплексного числа с 5 знаками после запятой: a+bi / a-bi
s = sprintf('%.5f%+.5fi', real(z), imag(z));
end

function [F,J] = F_and_J(z, T, p1, p2, t0, opts)
% За одну интеграцию: F(z)=P(z)-z и J = DP(z) - I
z0  = [z(:); reshape(eye(2),4,1)];
ode = @(t,zz) rhs_aug(t,zz,p1,p2);
[~,Z] = ode113(ode, [t0 t0+T], z0, opts);
xT   = Z(end,1:2).';
PhiT = reshape(Z(end,3:6),2,2);
F = xT - z(:);
J = PhiT - eye(2);
end

function dz = rhs_aug(t,z,p1,p2)
x  = z(1); 
y  = z(2);
Phi = reshape(z(3:6),2,2);
fx = y;
fy = x - x^3 + p1*x*y + p2*sin(t);
J  = [0, 1;
      1 - 3*x^2 + p1*y,  p1*x];
dPhi = J*Phi;
dz = [fx; fy; dPhi(:)];
end

function [z, ok] = newton_P(z0, FJ, tol, itmax, c_armijo)
% Демпфированный Ньютон с бэктрекингом Армихо: FJ(z)-> [F,J], J=DP-I
z = z0(:); ok = false; Fnorm_prev = Inf;
for it = 1:itmax
    [F,J] = FJ(z); Fn = norm(F);
    if Fn < tol, ok = true; return, end
    % шаг Ньютона
    try, s = -J\F; catch, s = -(J.'*J + 1e-8*eye(2)) \ (J.'*F); end
    % бэктрекинг
    alpha = 1.0; f0 = Fn;
    while alpha > 1e-6
        ztry = z + alpha*s;
        Ftry = FJ(ztry); if iscell(Ftry), Ftry = Ftry{1}; end
        if norm(Ftry) <= (1 - c_armijo*alpha)*f0, z = ztry; break, end
        alpha = alpha/2;
    end
    if alpha <= 1e-6, z = z + 1e-3*s; end
    if abs(Fnorm_prev - Fn) < 1e-12 && Fn < 10*tol, ok = Fn < tol; return, end
    Fnorm_prev = Fn;
end
end

function Zuniq = dedupe_fixpoints(Zin, tol)
if isempty(Zin), Zuniq = Zin; return; end
mask = true(size(Zin,1),1); Zuniq = [];
for i=1:size(Zin,1)
    if ~mask(i), continue; end
    d = vecnorm((Zin - Zin(i,:)).',2,1).';
    cluster = d <= tol;
    Zuniq = [Zuniq; mean(Zin(cluster,:),1)]; %#ok<AGROW>
    mask(cluster) = false;
end
end

function typ = classify_point(ev, tol1)
m1 = abs(ev(1)); m2 = abs(ev(2));
isRealPair = max(abs(imag(ev))) < 1e-12;
if isRealPair
    if (m1>1+tol1 && m2<1-tol1) || (m2>1+tol1 && m1<1-tol1)
        typ = 'saddle';
    elseif m1<1-tol1 && m2<1-tol1
        typ = 'stable node';
    elseif m1>1+tol1 && m2>1+tol1
        typ = 'unstable node';
    else
        typ = 'parabolic/degenerate (|?|?1)';
    end
else
    if m1<1-tol1, typ='stable focus'; elseif m1>1+tol1, typ='unstable focus'; else, typ='elliptic'; end
end
end

function s = tern(cond, a, b)
if cond, s=a; else, s=b; end
end
