%                      MAT 580 Project Source Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 1
% In this part, we plot the two orbits X1 and X2 of the two planets.
% All parameters or constants labeled with j (j=1,2) is related to the 
% orbit of planet j.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We first define our time points in our time vector.
time = linspace(0,2*pi);
% Next, we define some useful constants which define the shapes of the
% orbits.
A1 = 10; P1 = 2; theta1 = pi/8;
A2 = 4; P2 = 1; theta2 = -pi/7;
% We will use the following symbolic functions which will simplify the 
% code:
% R(z) is a 2x2 rotation matrix,
% v(P,A,t) is a 2x1 vector.
syms z P A t
R(z) = [cos(z), sin(z); -sin(z), cos(z)]; 
v(P,A,t) = [0.5*(P-A)+0.5*(P+A)*cos(t); sqrt(P*A)*sin(t)];
% We now compute the orbits X1 and X2
X1 = R(theta1)*v(P1,A1,time); X2 = R(theta2)*v(P2,A2,time); 
% We extract the inputs from X1 and X2 which represent the x and y
% positions of each planet.
x1 = X1(1,:); y1 = X1(2,:);
x2 = X2(1,:); y2 = X2(2,:);
% To conclude part 1, we plot both orbits on one graph.
figure(1)
plot(x1,y1,'k',x2,y2,'--k');
xlabel('x-axis'); ylabel('y-axis');
title('The Plots of Two Orbits');
legend('Orbit 1', 'Orbit 2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part  2
% In this part we plot contour lines of our dist(t1,t2) function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We first define the domain of our distance function that we seek to 
% minimize. The domain is [0,2pi]^2.
[t1, t2] = meshgrid(time,time);
% Now we define our dist function in terms of t1 and t2. We use the
% functions x11, x22, y11, and y22 to simplify code.
x11 = cos(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(t1))+...
    sin(theta1)*sqrt(P1*A1)*sin(t1);
x22 = cos(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(t2))+...
    sin(theta2)*sqrt(P2*A2)*sin(t2);
y11 = -sin(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(t1))+...
    cos(theta1)*sqrt(P1*A1)*sin(t1);
y22 = -sin(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(t2))+...
    cos(theta2)*sqrt(P2*A2)*sin(t2);
dist = 0.5*( (x11-x22)^2+(y11-y22)^2 );
% To end part 2, we plot contour lines of dist(t1,t2).
figure(2)
contour(t1,t2,dist,'k')
xlabel('t_1-axis'); ylabel('t_2-axis');
title('Contour Lines of dist(t_1,t_2)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 3
% In this last part, we will minimize our dist function over its domain
% using the steepest descent method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We first determine the gradient of dist.
syms T1 T2
X11 = cos(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(T1))+...
    sin(theta1)*sqrt(P1*A1)*sin(T1);
X22 = cos(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(T2))+...
    sin(theta2)*sqrt(P2*A2)*sin(T2);
Y11 = -sin(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(T1))+...
    cos(theta1)*sqrt(P1*A1)*sin(T1);
Y22 = -sin(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(T2))+...
    cos(theta2)*sqrt(P2*A2)*sin(T2);
Dist(T1,T2) = 0.5*( (X11-X22)^2+(Y11-Y22)^2 );
GradDist(T1,T2) = [diff(Dist,T1); diff(Dist,T2)];
epsilon = 0.01;
minT1=3;minT2=5.5;
%double(GradDist(minT1,minT2))
while norm(GradDist(minT1,minT2)) >= epsilon
    d = -GradDist(minT1,minT2);
    %minimize f(minT+lambda*d)
    syms L
    f(L)= Dist(minT1+L*d(1), minT2+L*d(2));
    fminsearch(f,0)

