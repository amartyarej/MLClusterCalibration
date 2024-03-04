# -*- coding: utf-8 -*-

""" Collection of utility tools."""

# Basic import(s)
import os

# Scientific import(s)
import ROOT
try:
    from root_numpy import tree2array

    import numpy as np
    from numpy.lib.recfunctions import append_fields
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source snippets/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise

# Global definitions
inf = np.finfo(float).max
eps = np.finfo(float).eps


def get_maximum (hist):
    """ Return the maximum bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return get_maximum(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = -inf, ROOT.Double(0), ROOT.Double(-inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            output = max(float(output),float(y))
            pass
        return output

    try:
        return max(map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)))
    except ValueError:
        warning("get_maximum: No bins were found.")
    except:
        warning("get_maximum: Something didn't work here, for intput:")
        print hist
        pass
    return None


def get_minimum (hist):
    """ Return the minimum bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return get_minimum(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = inf, ROOT.Double(0), ROOT.Double(inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            output = min(float(output),float(y))
            pass
        return output

    try:
        return min(map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)))
    except ValueError:
        warning("get_minimum: No bins were found.")
    except:
        warning("get_minimum: Something didn't work here, for intput:")
        print hist
        pass
    return None


def get_minimum_positive (hist):
    """ Return the minimum positive bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return inf if hist.GetNhists() == 0 else get_minimum_positive(hist.GetStack()[0])#get_minimum_positive(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = inf, ROOT.Double(0), ROOT.Double(inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            if x <= 0: continue
            output = min(float(output),float(y))
            pass
        return output

    try:
        return min([m for m in map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)) if m > 0])
    except ValueError:
        warning("get_minimum_positive: No bins were found.")
    except:
        warning("get_minimum_positive: Something didn't work here, for intput:")
        print hist
        pass
    return None


def get_stack_sum (stack, only_first=True):
    """ ... """

    # Kinda hacky...
    if only_first:
        sumHisto = stack.GetHists()[0].Clone('sumHisto')
    else:
        # @TODO: Errors are not being treated properly...
        sumHisto = None
        for hist in stack.GetHists(): ##stack.GetStack():
            if sumHisto is None:
                sumHisto = hist.Clone('sumHisto')
            else:
                sumHisto.Add(hist)
                pass
            pass
        pass
    return sumHisto


def is_overlay (pad):
    """ Determine whether input pad is of type 'overlay' """
    return type(pad).__name__.endswith('overlay')


def is_canvas (pad):
    """ Determine whether input pad is of type 'canvas' """
    return type(pad).__name__.endswith('canvas')


def warning (string):
    """ ... """
    print '\033[91m\033[1mWARNING\033[0m ' + string
    return


def snapToAxis (x, axis):
    """ ... """

    bin = axis.FindBin(x)
    if   bin <= 0:
        xdraw = axis.GetXmin()
    elif bin > axis.GetNbins():
        xdraw = axis.GetXmax()
    else:
        down   = axis.GetBinLowEdge(bin)
        up     = axis.GetBinUpEdge (bin)
        middle = axis.GetBinCenter (bin)

        # Assuming snap to nearest edge. # @TODO: Improve?
        d1 = abs(x - down);
        d2 = abs(x - up);
        if d1 == d2:
            warning("Coordinate exactly in the middle of bin. Returning lower edge.")
            pass
        if (d1 <= d2): xdraw = down
        else:          xdraw = up
        pass
    return xdraw


def wait ():
    """ Generic wait function.

    Halts the execution of the script until the user presses ``Enter``.
    """
    raw_input("\033[1mPress 'Enter' to continue...\033[0m")
    return
