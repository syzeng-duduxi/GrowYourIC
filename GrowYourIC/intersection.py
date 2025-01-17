#!/usr/bin/env python3
# Author: Marine Lasbleis

from __future__ import division
from __future__ import absolute_import


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

# dummies function for testing


def function_f(x, *args):
    return np.sin(x + 1.)


def function_g(x):
    return np.sin(x)


def function_diff(x, *args):
    return function_f(x, *args) - function_g(x)


# solver functions

def zero_fsolve(f, x0, *args):
    """ find the zero of the function f using fsolve.

    Please be careful that this method is not a good one when the derivative of the function f changes a lot
    x0 is the first point tested by fsolve. In the IC code, please use either 0 or age_ic."""
    return fsolve(lambda x: f(x), x0)


def zero_newton():
    pass  # TODO


def zero_brentq(f, *args, **kwargs):
    """ Find the zero of the function f, using the brentq method.

    two possibilities:
        - either the interval a,b is already set (and we just have to check the signs of f(a) and f(b).
        - or it is not, and a figure pops up to define the interval.

        the program can also learn the value of the interval from other already calculated values. But need to be a little bit more careful (need to check the closests points, try to use this value, etc.)

    *args are used by the function f, while **kwargs are input for the intervals.
    """
    if 'a' in kwargs and 'b' in kwargs:
        # simple one: the interval [a,b] is known
        a = kwargs["a"]
        b = kwargs["b"]
    else:
        a, b = 0., 1.  # choose_interval_on_graph(f, 0, 2, *args)

    # add an "interval" variable, that will be an output (save all the different intervals that have been previously used).
    # first check if any of the intervals proposed would be valid. If yes,
    # then use it.


#     iterations = 0
#     while np.sign(f(a, *args)) == np.sign(f(b, *args)) and iterations <= 10 and a!=b:
#         if iterations == 0:
#             print "Please choose other values for the interval so that the function changes sign between the two borns."
#             a, b = choose_interval_on_graph(f, a, b, *args)
#         elif iterations == 10:
#             print "To solve this, you need to choose the interval such as the function changes sign. You have a last chance to find one: please type the two values of the interval."
#             a = float(input("Enter first value: "))
#             b = float(input("Enter second value: "))
#         else:
#             a, b = choose_interval_on_graph(f, a-(b-a), b+(b-a), *args)
#         iterations += 1

    a, b = interval(f, a, b, *args)

    if a == b:
        solution = 0.
    else:
        solution = brentq(f, a, b, args=args)
    return solution


def check_validity_interval(f, a, b, *args):
    """ Return True if the function change sign between point a and b.

    *args are additionnal arguments needed by the function f.
    Brentq method work only if the function change sign at least one in the choosen interval. """
    return np.sign(f(a, *args)) != np.sign(f(b, *args))


def interval(f, a, b, *args, **kwargs):
    """ Return a correct interval to use the brentq method.

    *args are additionnal parameters used by the function. **kwargs are parameters for this particular function ( #TODO : add different way to define the new interval, either by hand or automatic)
    Brentq method work only if the function change sign at least one in the choosen  interval. """
    # a,b given to the function are the first guesses.
    if a > b:
        a, b = b, a
    if check_validity_interval(f, a, b, *args):
        return a, b
    elif check_validity_interval(f, b / 2., b * 1.001, *args):
        return b / 2., b * 1.001
    elif check_validity_interval(f, 3. * b / 4., b * 1.01, *args):
        return 3. * b / 4., b * 1.01
    elif check_validity_interval(f, b * 0.9, b * 1.1, *args):
        return b * 0.9, b * 1.1
    elif check_validity_interval(f, b * 0.9, b * 1.01, *args):
        return b * 0.98, b * 1.01
    else:
        print("now we have a problem")
        iterations = 0
        while not check_validity_interval(f, a, b, *args) and iterations <= 10:
            if iterations == 0:
                a, b = choose_interval_on_graph(f, a, b, *args)
            elif iterations == 10:
                print("To solve this, you need to choose the interval such as the funct    ion changes sign. You have a last chance to find one: please type the two values of the interval.")
                a = float(input("Enter first value: "))
                b = float(input("Enter second value: "))
            else:
                a, b = choose_interval_on_graph(
                    f, a - (b - a), b + (b - a), *args)
            iterations += 1
        return a, b


def choose_interval_on_graph(f, a, b, *args):
    """ allow the user to click on graph to choose an interval."""
    global coords
    print(args)
    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_val = np.linspace(a, b, 100)
    f_val = np.zeros_like(x_val)
    for i, x in enumerate(x_val):
        f_val[i] = f(x, *args)
    ax.plot(x_val, f_val, label='function')
    ax.plot(x_val, x_val * 0., label='zero')
    ax.set_title("Please click on two points to define the interval.")
    cid = fig.canvas.mpl_connect('button_press_event', onclick_two)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.legend()
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(cid2)
    return coords[0], coords[1]


def onclick(event):
    global position_x
    # print('button=%d, xdata=%f, ydata=%f' %
    #       (event.button, event.xdata, event.ydata))
    position_x = event.xdata
    plt.close()


def onclick_two(event):
    global coords
    # print('button=%d, xdata=%f, ydata=%f' %
    #      (event.button, event.xdata, event.ydata))
    ix = event.xdata
    coords.append(ix)
    if len(coords) == 2:
        plt.close()


def onkey(event):
    global coords
    coords = [0., 0.]
    plt.close()
