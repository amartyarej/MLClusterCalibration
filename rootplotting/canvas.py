# -*- coding: utf-8 -*-

""" Wrapper around ROOT TCanvas, handling pads, showing, and saving."""

# Scientific import(s)
import ROOT
try:
    import numpy as np
    from root_numpy import fill_hist, array2hist
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source snippets/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise

# Project import(s)
from rootplotting.tools import *
from rootplotting.style import *
from rootplotting import pad


# Class definition
class canvas (object):
    """
    docstring for canvas
    @TODO: Elaborate!
    """

    def __init__ (self, num_pads=1, size=None, fraction=0.3, batch=False, ratio=True):
        """ Constructor. """
        super(canvas, self).__init__()

        ROOT.gROOT.SetBatch(batch)

        # Check(s)
        #assert type(num_pads) == int, "Number of pads must be an integer"
        #assert num_pads < 3, "Requested number of pads {} is too large".format(num_pads)
        #assert num_pads > 0, "Requested number of pads {} is too small".format(num_pads)

        # Member variables
        self._num_pads = num_pads
        self._fraction = fraction if num_pads == 2 else 0.
        self._size = size or ((600, int(521.79/float(1. - 0.3))) if (num_pads == 2 and ratio) else (600,500))
        self._canvas = ROOT.TCanvas('c_{}'.format(id(self)), "", self._size[0], self._size[1])
        self._ratio = ratio and self._num_pads == 2
        self._setup = False
        self._existinglines = set() # X-axis lines already for previous regions

        # -- Pads
        self._pads = list()
        if isinstance(self._num_pads, int) and self._num_pads <= 2:
            if   self._num_pads == 1:
                self._pads.append(pad(self, (0, 0, 1, 1)))
            elif self._num_pads == 2:
                self._pads.append(pad(self, (0, fraction, 1, 1)))
                self._pads.append(pad(self, (0, 0, 1, fraction)))
                pass
        elif isinstance(fraction, (list, tuple)):

            assert len(fraction) == num_pads
            if set(map(type,fraction)) == {int}:
                fraction = [f / float(sum(fraction)) for f in fraction]
            else:
                assert np.abs(np.sum(fraction) - 1.0) < 1.0E-06, "Fractions don't sum to one: {}".format(sum(fraction))
                pass

            fractions = 1. - np.cumsum([0] + list(fraction))
            
            margin = 0.1
            m1, m2 = margin, 0.
            for y2, y1 in zip(fractions[:-1], fractions[1:]):
                # @TODO: Margin?
                self._pads.append(pad(self, (0, y1, 1, y2 )))
                m2 += margin / float(num_pads - 1)
                m1 -= margin / float(num_pads - 1)
                pass

        else: # Assuming regular pad placement
            if isinstance(self._num_pads, int):
                self._num_pads = [self._num_pads]
                pass
            self._num_pads = list(self._num_pads)
            while len(self._num_pads) < 2: self._num_pads.insert(0, 1)
            assert len(self._num_pads) == 2

            nx, ny = self._num_pads
            fx, fy = map(lambda n: 1. / float(n), self._num_pads)
            margin = 0.01
            n = nx * ny
            for ipad in reversed(range(n)):
                ix = (nx - 1 - ipad) %  nx
                iy = ipad // nx

                self._pads.append(pad(self, ( ix      * fx,  iy      * fy,
                                             (ix + 1) * fx, (iy + 1) * fy)))
                pass
            pass

        '''
        else:  # Assuming regular, horizontal pad placement
            f = 1. / float(self._num_pads)
            offset = 0.2
            height = 1 - offset
            for idx in reversed(range(self._num_pads)):
                self._pads.append(pad(self, (0, offset + idx * f * height, 1, offset + (idx + 1) * f * height)))
                pass
            pass
            '''


        # Draw pads
        for p in self._pads:
            self._canvas.cd()
            p._pad.Draw()
            self._canvas.Update()
            pass
        return


    def __del__ (self):
        """ Destructor. """
        for p in self._pads:
            del p
            pass
        del self._canvas
        return



    # Public accessor methods
    # ----------------------------------------------------------------

    def pad (self):
        """ ... """
        assert len(self._pads) == 1, "More than one pad exists; call is ambiguous"
        return self._pads[0]

    def pad (self, idx=0):
        """ ... """
        return self.pads()[idx]


    def pads (self):
        """ ... """
        return self._pads



    # Public deferral methods
    # ----------------------------------------------------------------

    def defer (idx_default): # Decorators taking arguments need an additional level of nesting.
        def inner(func):
            def wrapper (self, *args, **kwargs):
                idx_pad = kwargs.pop('idx_pad', idx_default)
                assert idx_pad == -1 or (idx_pad >= 0 and idx_pad < self._num_pads), "Requested pad index {} is no good.".format(idx_pad)
                # Find non-overlay pad
                if is_overlay(self._pads[idx_pad]):
                    idx_pad = len(self._pads) - 1
                    while idx_pad >= 0:
                        idx_pad -= 1
                        if not is_overlay(self._pads[idx_pad]): break
                        pass
                    pass
                assert idx_pad == -1 or (idx_pad >= 0 and idx_pad < self._num_pads), "Requested pad index {} is no good.".format(idx_pad)
                method = getattr(self._pads[idx_pad], func.__name__)
                # Note: func(self, ...) is never called
                return method(*args, **kwargs)
            return wrapper
        return inner


    @defer(0)
    def text (self, *args, **kwargs): return
    """ Defer method to _first_ pad, by default """

    @defer(0)
    def latex (self, *args, **kwargs): return

    @defer(0)
    def legend (self, *args,**kwargs): return

    @defer(-1)
    def xlabel (self, *args, **kwargs): return
    """ Defer method to _last_ non-overlay pad, by default """

    @defer(0)
    def ylabel (self, *args, **kwargs): return

    @defer(0)
    def plot (self, *args, **kwargs): return

    @defer(0)
    def hist (self, *args, **kwargs): return

    @defer(0)
    def hist2d (self, *args, **kwargs): return

    @defer(0)
    def stack (self, *args, **kwargs): return

    @defer(0)
    def graph (self, *args, **kwargs): return

    @defer(-1)
    def ratio_plot (self, *args, **kwargs): return

    @defer(-1)
    def diff_plot (self, *args, **kwargs): return

    @defer(0)
    def getStackSum (self, *args, **kwargs): return

    @defer(0)
    def xlim (self, *args, **kwargs): return

    @defer(0)
    def ylim (self, *args, **kwargs): return

    @defer(0)
    def ymin (self, *args, **kwargs): return

    @defer(0)
    def padding (self, *args, **kwargs): return

    @defer(0)
    def log (self, *args, **kwargs): return

    @defer(0)
    def logx (self, *args, **kwargs): return

    @defer(0)
    def logy (self, *args, **kwargs): return

    @defer(0)
    def line (self, *args, **kwargs): return

    @defer(0)
    def lines (self, *args, **kwargs): return

    @defer(0)
    def xline (self, *args, **kwargs): return

    @defer(0)
    def yline (self, *args, **kwargs): return

    @defer(0)
    def xlines (self, *args, **kwargs): return

    @defer(0)
    def ylines (self, *args, **kwargs): return



    # Public high-level/management methods
    # ----------------------------------------------------------------

    def _update (self):
        """ ... """

        self._canvas.Update()

        # Set up main- and ratio pads, in the most common case
        if self._ratio: # and not self._setup (?)
            self._setup_ratio_pads()
            pass

        # Update children pads
        for p in self._pads:
            p._update()
            pass

        return

    def show (self):
        """ ... """
        self._update()
        wait()
        return


    def save (self, path):
        """ ... """

        self._update()
        self._canvas.SaveAs(path)
        return


    def region (self, name, xmin, xmax, offset=0.10): # @TODO: do **kwargs for text- and line styling
        """ ... """

        # Check(s)
        # ...

        # Main pad
        pad = self._pads[0]
        pad._bare().cd()

        axis = pad._get_first_primitive().GetXaxis()
        xmin = snapToAxis(xmin, axis)
        xmax = snapToAxis(xmax, axis)

        if xmax <= self.xlim()[0] or xmin >= self.xlim()[1]: return

        drawmin = xmin not in self._existinglines
        drawmax = xmax not in self._existinglines

        self._existinglines.add(xmin)
        self._existinglines.add(xmax)

        if drawmin: xmin = pad.xline(xmin, linewidth=2)
        if drawmax: xmax = pad.xline(xmax, linewidth=2)

        xNDC = (0.5 * (xmin + xmax) - axis.GetXmin()) / (axis.GetXmax() - axis.GetXmin())
        xNDC = pad._pad.GetLeftMargin() + xNDC * (1 - pad._pad.GetLeftMargin() - pad._pad.GetRightMargin())

        pad.latex(name, xNDC, offset, NDC=True)

        # Remaining pads (opt.)
        for idx in range(1,len(self._pads)):
            pad = self._pads[idx]
            if is_overlay(pad): continue
            pad._bare().cd()

            if drawmin: xmin = pad.xline(xmin, linewidth=2)
            if drawmax: xmax = pad.xline(xmax, linewidth=2)
            pass

        return



    # Private accessor methods
    # ----------------------------------------------------------------

    def _bare (self):
        """ ... """
        return self._canvas


    def _setup_ratio_pads (self):
        """ ... """

        # Main pad(s) -- incl. overlay
        for main_pad in self._pads[:-1]:

            if main_pad._xaxis() is None: continue

            main_pad._xaxis().SetLabelOffset(9999.)
            main_pad._xaxis().SetTitleOffset(9999.)
            main_pad._bare().SetBottomMargin(0.030) # 0.03
            main_pad._bare().Modified()
            main_pad._bare().Update()

            if is_overlay(main_pad):
                main_pad._yaxis().SetLabelOffset(9999.)
                main_pad._yaxis().SetTitleOffset(9999.)
                main_pad._xaxis().SetTickLength(0)
                main_pad._yaxis().SetTickLength(0)
                pass
            pass

        # Ratio pad
        ratio_pad = self._pads[-1]
        if ratio_pad._get_first_primitive():
            ratio_pad._bare().SetBottomMargin(0.30) # 0.30
            ratio_pad._bare().SetTopMargin   (0.040) # 0.04

            ratio_pad._xaxis().SetTitleOffset(ROOT.gStyle.GetTitleOffset('x') * ratio_pad._scale[1])
            ratio_pad._yaxis().SetTitleOffset(ROOT.gStyle.GetTitleOffset('y') * ratio_pad._scale[0])
            ratio_pad._yaxis().SetNdivisions(505)
            ratio_pad._xaxis().SetTickLength (ROOT.gStyle.GetTickLength ('x') * ratio_pad._scale[1])
            if ratio_pad._ylim is not None:
                axisrange = ratio_pad._ylim
                ratio_pad._yaxis().SetRangeUser(*axisrange)
                ratio_pad._get_first_primitive().SetMinimum(axisrange[0])
                ratio_pad._get_first_primitive().SetMaximum(axisrange[1])
                pass
            ratio_pad._bare().Modified()
            ratio_pad._bare().Update()
            pass

        self._setup = True
        return

    pass
