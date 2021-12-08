import matplotlib.pyplot as plt
import numpy as np

#  draw_mql  Renders an MQL creatrure
#
#  Input:
#     Most arguments are angles which are descretized into num_angles and thus
#     ar represented by an integer between 1 and num_angles. The number of
#     possible angles, num_angles, must be odd, and the unique central index
#     corresponds to the bisector (e.g., 45 degrees) with respect to the
#     relavent quadrant.
#
#     num_angles = Number of allowable angles which must be an odd integer.
#                  For the current version of the assignment, this should be 9.
#                  This sets the range of  the discretized angles which must be
#                  in 1:num_angles.
#
#     angle_step = The differnce between two successive angles. In the current
#                  version of the assignmet, this is 15 degress.
#
#     shape   =    Set to 1 if the body is a circle, 0 for a hexagon
#     riaa    =    The inner angle for the right arm.
#     rila    =    The inner angle for the right leg.
#     liaa    =    The inner angle for the left arm.
#     lila    =    The inner angle for the left leg.
#     roaa    =    The outer angle for the right arm.
#     rola    =    The right leg outer angle.
#     loaa    =    The left arm outer angle.
#     lola    =    The left leg outer angle.
#
#   Output:
#      figure_handle, a handle to the figure where the MQL creature was
#      rendered. Use functions close to close the figure, and saveas
#      to save it as an image.
#
#   Credit:
#        This implementation is based heavily on code written by Luca del Pero, class of 2011.

def draw_mql(num_angles, angle_step, shape, riaa, rila, liaa, lila, roaa, rola, loaa, lola,fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    plt.title("RIAA =" + str(riaa) + ", LIAA = " + str(liaa) +", RILA =" + str(rila) + ", LILA = " + str(lila) +"\nROAA =" + str(roaa) + ", LOAA = " + str(loaa) +", ROLA =" + str(rola) + ", LOLA = " + str(lola))

    if (riaa < 1 or riaa > num_angles):
        print("Error, riaa must be between 1 and ", num_angles)
        return
    if (rila < 1 or rila > num_angles):
        print("Error, rila must be between 1 and ", num_angles)
        return
    if (liaa < 1 or liaa > num_angles):
        print("Error, liaa must be between 1 and ", num_angles)
        return
    if (lila < 1 or lila > num_angles):
        print("Error, lila must be between 1 and ", num_angles)
        return
    if (roaa < 1 or roaa > num_angles):
        print("Error, roaa must be between 1 and ", num_angles)
        return
    if (rola < 1 or rola > num_angles):
        print("Error, rola must be between 1 and ", num_angles)
        return
    if (loaa < 1 or loaa > num_angles):
        print("Error, loaa must be between 1 and ", num_angles)
        return
    if (lola < 1 or lola > num_angles):
        print("Error, lola must be between 1 and ", num_angles)
        return
    if (shape < 0 or shape > 1):
        print("Error, shape must be either 0 or 1")

    angle_step_rad = angle_step * np.pi / 180

    riaa_rad = (1 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, riaa)
    rila_rad = (7 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, rila)

    liaa_rad = (3 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, liaa)
    lila_rad = (5 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, lila)

    roaa_rad = (1 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, roaa)
    rola_rad = (7 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, rola)

    loaa_rad = (3 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, loaa)
    lola_rad = (5 / 4) * np.pi + bins_to_rad(num_angles, angle_step_rad, lola)

    limb_radius_x = 1.0
    limb_radius_y = 0.4

    if shape == 1:
        radius_body = 1
        myEllipse(0, 0, 1, 0, radius_body, radius_body, [0.5, 0.5, 0.5], ax)
    else:
        t = np.arange(1 / 12, 1, 1 / 6) * 2 * np.pi
        x = (1.2) * np.sin(t)
        y = (1.2) * np.cos(t)
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        ax.fill(x, y,color=[0.5,0.5,0.5])

    contact_riaa_rad_x = 1 / np.sqrt(2)
    contact_riaa_rad_y = 1 / np.sqrt(2)
    vertex_x = -limb_radius_x
    dx = vertex_x - vertex_x * np.cos(riaa_rad)
    dy = -vertex_x * np.sin(riaa_rad)
    myEllipse(contact_riaa_rad_x + limb_radius_x + dx, contact_riaa_rad_y + dy, 1, riaa_rad, limb_radius_x,
              limb_radius_y, [1.0, 1.0, 0.0], ax)

    vertex_x = limb_radius_x
    contact_roaa_rad_x = vertex_x * np.cos(riaa_rad) + contact_riaa_rad_x + limb_radius_x + dx
    contact_roaa_rad_y = vertex_x * np.sin(riaa_rad) + contact_riaa_rad_y + dy
    vertex_x = -limb_radius_x
    dx = vertex_x - vertex_x * np.cos(roaa_rad)
    dy = -vertex_x * np.sin(roaa_rad)
    myEllipse(contact_roaa_rad_x + limb_radius_x + dx, contact_roaa_rad_y + dy, 1, roaa_rad, limb_radius_x,
              limb_radius_y, [1.0, 0.0, 1.0], ax)

    contact_liaa_rad_x = -1 / np.sqrt(2)
    contact_liaa_rad_y = 1 / np.sqrt(2)
    vertex_x = limb_radius_x
    dx = vertex_x + vertex_x * np.cos(liaa_rad)
    dy = vertex_x * np.sin(liaa_rad)
    myEllipse(contact_liaa_rad_x - limb_radius_x + dx, contact_liaa_rad_y + dy, 1, liaa_rad, limb_radius_x,
              limb_radius_y, [1.0, 1.0, 0.0], ax)
    vertex_x = -limb_radius_x
    contact_loaa_rad_x = -vertex_x * np.cos(liaa_rad) + contact_liaa_rad_x - limb_radius_x + dx
    contact_loaa_rad_y = -vertex_x * np.sin(liaa_rad) + contact_liaa_rad_y + dy
    vertex_x = limb_radius_x
    dx = vertex_x + vertex_x * np.cos(loaa_rad)
    dy = vertex_x * np.sin(loaa_rad)
    myEllipse(contact_loaa_rad_x - limb_radius_x + dx, contact_loaa_rad_y + dy, 1, loaa_rad, limb_radius_x,
              limb_radius_y, [1.0, 0.0, 1.0], ax)
    contact_rila_rad_x = 1 / np.sqrt(2)
    contact_rila_rad_y = -1 / np.sqrt(2)
    vertex_x = -limb_radius_x
    vertex_y = 0
    dx = vertex_x - vertex_x * np.cos(rila_rad)
    dy = -vertex_x * np.sin(rila_rad)
    myEllipse(contact_rila_rad_x + limb_radius_x + dx, contact_rila_rad_y + dy, 1, rila_rad, limb_radius_x,
              limb_radius_y, [1.0, 0.0, 0.0], ax)
    vertex_x = limb_radius_x
    contact_rola_rad_x = vertex_x * np.cos(rila_rad) + contact_rila_rad_x + limb_radius_x + dx
    contact_rola_rad_y = vertex_x * np.sin(rila_rad) + contact_rila_rad_y + dy
    vertex_x = -limb_radius_x
    dx = vertex_x - vertex_x * np.cos(rola_rad)
    dy = -vertex_x * np.sin(rola_rad)
    myEllipse(contact_rola_rad_x + limb_radius_x + dx, contact_rola_rad_y + dy, 1, rola_rad, limb_radius_x,
              limb_radius_y, [0.0, 1.0, 0.0], ax)

    contact_lila_rad_x = -1 / np.sqrt(2)
    contact_lila_rad_y = -1 / np.sqrt(2)
    vertex_x = limb_radius_x
    dx = vertex_x + vertex_x * np.cos(lila_rad)
    dy = vertex_x * np.sin(lila_rad)
    myEllipse(contact_lila_rad_x - limb_radius_x + dx, contact_lila_rad_y + dy, 1, lila_rad, limb_radius_x,
              limb_radius_y, [1.0, 0.0, 0.0], ax)
    vertex_x = -limb_radius_x
    contact_lola_rad_x = -vertex_x * np.cos(lila_rad) + contact_lila_rad_x - limb_radius_x + dx
    contact_lola_rad_y = -vertex_x * np.sin(lila_rad) + contact_lila_rad_y + dy
    vertex_x = limb_radius_x
    dx = vertex_x + vertex_x * np.cos(lola_rad)
    dy = vertex_x * np.sin(lola_rad)
    myEllipse(contact_lola_rad_x - limb_radius_x + dx, contact_lola_rad_y + dy, 1, lola_rad, limb_radius_x,
              limb_radius_y, [0.0, 1.0, 0.0], ax)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


def bins_to_rad(num_angles, angle_step_rad, i):
    angle = angle_step_rad * (i - 1 - (num_angles - 1) / 2)
    return angle


def myEllipse(x0, y0, dilation, rotation, rx, ry, thecolor, ax):
    # x0, y0: coordinates of center point % rx, ry: radii in x and y directions
    # Part I: Compute ellipse with axes parallel to % x and y axes and center at (0, 0)
    delt = 2 * np.pi / 30
    t = np.arange(0, 2 * np.pi, delt)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    # Translate, dilate and rotate the ellipse
    xxxx = x0 + dilation * (np.cos(rotation) * x - np.sin(rotation) * y)
    xxxx = np.append(xxxx, xxxx[0])
    yyyy = y0 + dilation * (np.sin(rotation) * x + np.cos(rotation) * y)
    yyyy = np.append(yyyy, yyyy[0])
    ax.plot(xxxx, yyyy, 'k-')
    ax.fill(xxxx, yyyy, color=thecolor)


