% author: Claudio Gaz, Marco Cognetti
% date: August 2, 2019
% 
% -------------------------------------------------
% Parameters Retrieval Algorithm
% -------------------------------------------------
% C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca, 'Dynamic
% Identification of the Franka Emika Panda Robot With Retrieval of Feasible
% Parameters Using Penalty-Based Optimization'. IEEE RA-L, 2019.
%
% the following code has been tested on Matlab 2018b

% error_fcn_gM_LMI returns the value of the cost function (loss) when 
% phi(p_k) = || pi(p_k) - pi_hat ||^2 is used

function loss = error_fcn_gM_LMI(x)

global SA_step

% set the proper penalty coefficient (that is variable \xi_k in (29))
if SA_step == 1
    penalty = 0;
else
    penalty = 10*(SA_step-1);
end

% get the current values of the parameters
m1 = x(1);
m2 = x(2);
m3 = x(3);
m4 = x(4);
m5 = x(5);
m6 = x(6);
m7 = x(7);
c1x = x(8);
c1y = x(9);
c1z = x(10);
c2x = x(11);
c2y = x(12);
c2z = x(13);
c3x = x(14);
c3y = x(15);
c3z = x(16);
c4x = x(17);
c4y = x(18);
c4z = x(19);
c5x = x(20);
c5y = x(21);
c5z = x(22);
c6x = x(23);
c6y = x(24);
c6z = x(25);
c7x = x(26);
c7y = x(27);
c7z = x(28);
I1xx = x(29);
I1xy = x(30);
I1xz = x(31);
I1yy = x(32);
I1yz = x(33);
I1zz = x(34);
I2xx = x(35);
I2xy = x(36);
I2xz = x(37);
I2yy = x(38);
I2yz = x(39);
I2zz = x(40);
I3xx = x(41);
I3xy = x(42);
I3xz = x(43);
I3yy = x(44);
I3yz = x(45);
I3zz = x(46);
I4xx = x(47);
I4xy = x(48);
I4xz = x(49);
I4yy = x(50);
I4yz = x(51);
I4zz = x(52);
I5xx = x(53);
I5xy = x(54);
I5xz = x(55);
I5yy = x(56);
I5yz = x(57);
I5zz = x(58);
I6xx = x(59);
I6xy = x(60);
I6xz = x(61);
I6yy = x(62);
I6yz = x(63);
I6zz = x(64);
I7xx = x(65);
I7xy = x(66);
I7xz = x(67);
I7yy = x(68);
I7yz = x(69);
I7zz = x(70);

l_2x = c2x*m2;
l_2y = c2y*m2;
l_2z = c2z*m2;

l_3x = c3x*m3;
l_3y = c3y*m3;
l_3z = c3z*m3;

l_4x = c4x*m4;
l_4y = c4y*m4;
l_4z = c4z*m4;

l_5x = c5x*m5;
l_5y = c5y*m5;
l_5z = c5z*m5;

l_6x = c6x*m6;
l_6y = c6y*m6;
l_6z = c6z*m6;

l_7x = c7x*m7;
l_7y = c7y*m7;
l_7z = c7z*m7;


% initialize error vector
e = zeros(43,1);

% compute error vector, as the difference of current dynamic coeff values
% and previously estimated dyn coeff values, as follows:
% pi(p_k) - pi_hat

e(1) = I1zz + I2yy - 0.028990695968381247;
e(2) = I2xx - I2yy + I3yy + 79*l_3z/125 + 6241*m3/62500 + 426649*m4/4000000 + 426649*m5/4000000 + 426649*m6/4000000 + 426649*m7/4000000 - 1.0651474834659744;
e(3) = I2xy + 0.00533474017635381;
e(4) = I2xz - 0.02841350086210234;
e(5) = I2yz + 0.003497281766749531;
e(6) = I2zz + I3yy + 79*l_3z/125 + 6241*m3/62500 + 426649*m4/4000000 + 426649*m5/4000000 + 426649*m6/4000000 + 426649*m7/4000000 - 1.126074044119151;
e(7) = l_2x + 0.005723832971175722;
e(8) = l_2y - l_3z - 79*m3/250 - 79*m4/250 - 79*m5/250 - 79*m6/250 - 79*m7/250 + 3.3476040986026705;
e(9) = I3xx - I3yy + I4yy - 1089*m4/160000 - 0.010583010018193418;
e(10) = I3xy + 33*l_4z/400 + 0.00027987698780632375;
e(11) = I3xz + 0.01068746828327422;
e(12) = I3yz + 0.00479584645727147;
e(13) = I3zz + I4yy + 1089*m4/160000 + 1089*m5/80000 + 1089*m6/80000 + 1089*m7/80000 - 0.12747395936074463;
e(14) = l_3x + 33*m4/400 + 33*m5/400 + 33*m6/400 + 33*m7/400 - 0.7511271889437469;
e(15) = l_3y - l_4z - 0.023746864837443177;
e(16) = I4xx - I4yy + I5yy + 96*l_5z/125 + 562599*m5/4000000 + 562599*m6/4000000 + 562599*m7/4000000 - 0.6421652193659803;
e(17) = I4xy + 33*l_5z/400 + 99*m5/3125 + 99*m6/3125 + 99*m7/3125 - 0.1749839301652996;
e(18) = I4xz - 0.004746415222037061;
e(19) = I4yz + 0.002680876792530093;
e(20) = I4zz + I5yy + 96*l_5z/125 + 617049*m5/4000000 + 617049*m6/4000000 + 617049*m7/4000000 - 0.757277047268459;
e(21) = l_4x - 33*m5/400 - 33*m6/400 - 33*m7/400 + 0.5533095121670956;
e(22) = l_4y + l_5z + 48*m5/125 + 48*m6/125 + 48*m7/125 - 2.0172931018145204;
e(23) = I5xx - I5yy + I6yy + 121*m7/15625 - 0.038205097072257936;
e(24) = I5xy + 0.0037325801634169138;
e(25) = I5xz + 0.006142863580895254;
e(26) = I5yz - 0.007704066157478862;
e(27) = I5zz + I6yy + 121*m7/15625 - 0.024116544089101216;
e(28) = l_5x + 0.010041386674633604;
e(29) = l_5y - l_6z - 0.07703496608496434;
e(30) = I6xx - I6yy + I7yy - 121*m7/15625 - 0.002429525137091463;
e(31) = I6xy + 11*l_7z/125 - 0.014774993919912372;
e(32) = 0; %I6xz + 1.40042922372252e-05;
e(33) = I6yz - 0.0008203567928726396;
e(34) = I6zz + I7yy + 121*m7/15625 - 0.046106541111364543;
e(35) = l_6x + 11*m7/125 - 0.2342839505174128;
e(36) = l_6y - l_7z + 0.17457564517289592;
e(37) = I7xx - I7yy + 0.003251270335544678;
e(38) = I7xy - 0.0011840514614580022;
e(39) = I7xz - 0.002394976990240447;
e(40) = I7yz + 0.0005793529177883886;
e(41) = I7zz - 0.0029618770670512525;
e(42) = l_7x + 0.0015550515772522664;
e(43) = l_7y + 0.0004662327233065212;

loss = e'*e;

%------------------------------------------------
% External Penalties
%------------------------------------------------

% conditions on total mass

min_mass = 16;
max_mass = 20;

if m1+m2+m3+m4+m5+m6+m7 < min_mass
    loss = loss + penalty*(min_mass-(m1+m2+m3+m4+m5+m6+m7));
end
if m1+m2+m3+m4+m5+m6+m7 > max_mass
    loss = loss + penalty*(m1+m2+m3+m4+m5+m6+m7-max_mass);
end

% conditions on inertia tensors: triangle inequalities

% link 1
I1 = [I1xx,I1xy,I1xz ; I1xy,I1yy,I1yz ; I1xz,I1yz,I1zz];
loss = check_inertia_condition(I1,loss,penalty);
% link 2
I2 = [I2xx,I2xy,I2xz ; I2xy,I2yy,I2yz ; I2xz,I2yz,I2zz];
loss = check_inertia_condition(I2,loss,penalty);
% link 3
I3 = [I3xx,I3xy,I3xz ; I3xy,I3yy,I3yz ; I3xz,I3yz,I3zz];
loss = check_inertia_condition(I3,loss,penalty);
% link 4
I4 = [I4xx,I4xy,I4xz ; I4xy,I4yy,I4yz ; I4xz,I4yz,I4zz];
loss = check_inertia_condition(I4,loss,penalty);
% link 5
I5 = [I5xx,I5xy,I5xz ; I5xy,I5yy,I5yz ; I5xz,I5yz,I5zz];
loss = check_inertia_condition(I5,loss,penalty);
% link 6
I6 = [I6xx,I6xy,I6xz ; I6xy,I6yy,I6yz ; I6xz,I6yz,I6zz];
loss = check_inertia_condition(I6,loss,penalty);
% link 7
I7 = [I7xx,I7xy,I7xz ; I7xy,I7yy,I7yz ; I7xz,I7yz,I7zz];
loss = check_inertia_condition(I7,loss,penalty);

end