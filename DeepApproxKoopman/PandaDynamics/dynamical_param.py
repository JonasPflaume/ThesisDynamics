from itertools import chain

def get_dyna() -> list:
    ''' dirty implementation of getting dynamical parameters
    '''
    m1 = 3.8303
    m2 = 5.7113
    m3 = 3.5256
    m4 = 1.1148
    m5 = 1.7786
    m6 = 1.8548
    m7 = 2.0418
    c1x = -0.0073
    c1y = 0.0274
    c1z = -0.1957
    c2x = -0.0014
    c2y = -0.0349
    c2z = -0.0023
    c3x = 0.0541
    c3y = 0.0063
    c3z = -0.0317
    c4x = -0.0749
    c4y = 0.1298
    c4z = -0.0012
    c5x = -0.0057
    c5y = 0.0231
    c5z = -0.1728
    c6x = 0.0284
    c6y = 0.0051
    c6z = -0.0206
    c7x = -7.1399e-04
    c7y = -8.3331e-05
    c7z = 0.0899
    I1xx = 0.4216
    I1xy = 4.5679e-04
    I1xz = 0.0045
    I1yy = 0.4142
    I1yz = -0.0560
    I1zz = 0.0080
    I2xx = 0.0281
    I2xy = 0.0024
    I2xz = 0.0182
    I2yy = 0.0372
    I2yz = -4.9687e-04
    I2zz = 0.0351
    I3xx = 0.0664
    I3xy = -0.0019
    I3xz = -0.0144
    I3yy = 0.0769
    I3yz = -0.0107
    I3zz = 0.0151
    I4xx = 0.0348
    I4xy = 0.0222
    I4xz = 0.0052
    I4yy = 0.0272
    I4yz = -0.0043
    I4zz = 0.0495
    I5xx = 0.0743
    I5xy = -4.1791e-04
    I5xz = -0.0021
    I5yy = 0.0703
    I5yz = 0.0034
    I5zz = 0.0058
    I6xx = 0.0150
    I6xy = -0.0033
    I6xz = 0.0042
    I6yy = 0.0105
    I6yz = 0.0025
    I6zz = 0.0114
    I7xx = 0.0110
    I7xy = 8.7968e-04
    I7xz = 0.0024
    I7yy = 0.0143
    I7yz = -3.9522e-04
    I7zz = 0.0071
    scope = locals()
    param = [eval('[I{0}xx, I{0}xy, I{0}xz, I{0}yy, I{0}yz, I{0}zz, c{0}x, c{0}y, c{0}z, m{0}]'.format(i), scope) 
                 for i in range(1,8)]

    param = list(chain(*param))

    return param
