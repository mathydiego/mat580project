%                      MAT 580 Project Source Code
%                             Diego Avalos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 1
% In this part, we plot the two orbits X1 and X2 of the two planets.
% All parameters or constants labeled with j (j=1,2) are related to the 
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
dist = 0.5*( (x11-x22).^2+(y11-y22).^2 );
% To end part 2, we plot contour lines of dist(t1,t2).
figure(2)
contour(t1,t2,dist,'k','ShowText','on');
xlabel('t_1-axis'); ylabel('t_2-axis');
title('Contour Lines of dist(t_1,t_2)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 3
% In this last part, we will minimize our dist function over its specified 
% domain [0,2*pi]^2 using the steepest descent method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We first symbolically redefine our dist function (this time we call it
% 'Dist') in terms of the shorter functions X11, X22, Y11, Y22.
syms T1 T2
X11 = cos(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(T1))+...
    sin(theta1)*sqrt(P1*A1)*sin(T1);
X22 = cos(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(T2))+...
    sin(theta2)*sqrt(P2*A2)*sin(T2);
Y11 = -sin(theta1)*(0.5*(P1-A1)+0.5*(P1+A1)*cos(T1))+...
    cos(theta1)*sqrt(P1*A1)*sin(T1);
Y22 = -sin(theta2)*(0.5*(P2-A2)+0.5*(P2+A2)*cos(T2))+...
    cos(theta2)*sqrt(P2*A2)*sin(T2);
% Hence, the function we seek to minimize is the following
Dist(T1,T2) = 0.5*( (X11-X22).^2+(Y11-Y22).^2 );
% We now let Matlab compute the gradient of Dist symbolically.
GradDist(T1,T2) = [diff(Dist,T1); diff(Dist,T2)];
% This is the part where we implement the steepest descent method.
% Here we tweak our epsilon value. Some values that I will look closely are
% epsilon = 1, 0.5, 0.1, etc.
epsilon = 0.5;
% Now, from our contour plot in part 2, it seems that the following
% t-values are a good place to start our steepest descent.
minT1 = 5; minT2 = 3;
% We wish to count the number of iterations in our process and also monitor
% the computation time. The computation time may seem long, and it could be
% because I am making Matlab do a bunch of symbolic computations.
iterations=0;
tic;
% Let the steepest descent method begin! Observe that I am using the double
% function to make Matlab treat some values as doubles rather than symbolic
% values.
while norm(double(GradDist(minT1,minT2))) >= epsilon
% d is the negative of the gradient at the current point
    d = -double(GradDist(minT1,minT2));
% We seek to minimize the single-variable function of lambda 
% f(minT+lambda*d) over lambda >= 0. We first define the function Dist_L
% symbolically.
    syms L
    Dist_L = Dist(minT1+L*d(1), minT2+L*d(2));
% In order to use Matlab's fminsearch function we need to convert our above
% function into a function handle.
    Dist_L = matlabFunction(Dist_L);
% We trust fminsearch to find our minimum lambda Lmin.
    Lmin = fminsearch(Dist_L,0);
% Update our point and iteration number.    
    minT1 = minT1 + Lmin*d(1); minT2 = minT2 + Lmin*d(2);
    iterations = iterations+1;
end
% Keep track of the computation time.
TimeElapsed=toc;
% We store useful information in the info vector
info = [epsilon, minT1, minT2, iterations, TimeElapsed];
% and state my conclusions as follows
fprintf(['With epsilon = %1.3f, our t-values that minimize our dist ',...
    'function are t_1 = %2.2f and t_2 = %2.2f. There were %d ',...
    'iterations needed, and the computation time was %3.2f seconds.\n'],...
    info);
% We are done with part 3. However, I want to add an additional graph which
% includes the orbits of both planets and marks the locations of the two
% planets where they are the closest (the points are those computed using 
% steepest descent). This will allows us to visualize how well steepest 
% descent works with different epsilon values.
% Here are the positions of both orbits at minimal distance
XMin1 = double(R(theta1)*v(P1,A1,minT1)); 
XMin2 = double(R(theta2)*v(P2,A2,minT2));
% And here's our handsome plot!
figure(3)
plot(x1,y1,'k',x2,y2,'--k',[XMin1(1) XMin2(1)],[XMin1(2) XMin2(2)],'k*');
xlabel('x-axis'); ylabel('y-axis');
title('The Plots of Two Orbits With Minimal Points');
legend('Orbit 1', 'Orbit 2','Minimal Points');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Fin
