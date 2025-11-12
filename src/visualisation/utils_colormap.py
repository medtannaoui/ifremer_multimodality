from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorsys
import matplotlib


class CMAP:
    @classmethod
    def create_cmap(cls, values, colors, under=None, over=None, name="custom_cmap", return_vminvmax=False, return_levels=False):
        values = np.array(values)
        values_norm = (values - values.min()) / (values.max() - values.min())
        colors = np.array(colors)

        cmap = LinearSegmentedColormap.from_list(name, list(zip(values_norm, colors)))
        
        if under is not None:
            cmap.set_under(under)
        if over is not None:
            cmap.set_over(over)
        
        if return_vminvmax:
            return cmap, values.min(), values.max()
        elif return_levels:
            return cmap, values
        else:
            return cmap
    
    @classmethod
    def cira_ir(cls, return_vminvmax=False, return_levels=False, units = 'celsius'):
        color_mapping = [
            (-100, "#8c1e8b"),
            (-90, "#fcfefc"),
            (-89.99, "#262627"),
            (-80, "#e7fb0b"),
            (-79.99, "#f30305"),
            (-70, "#770104"),
            (-69.99, "#09f405"),
            (-60, "#067305"),
            (-59.99, "#0403fa"),
            (-50, "#040378"),
            (-49.99, "#55544e"),
            (-30, "#9cf5f4"),
            (-29.99, "#fcfcfc"),
            (40, "#4d4d4d")
        ]
        
        if units == 'kelvin': 
            color_mapping = [(T+273.15, color) for (T, color) in color_mapping]
            print("cest kevin")
        
        name = "cira_ir"
        return cls.create_cmap(np.array(color_mapping)[:,0].astype(float), np.array(color_mapping)[:,1], name=name, return_vminvmax=return_vminvmax, return_levels=return_levels)
    


    @classmethod
    def cmap_sar(cls):
        def gmtColormap(fileName):
            """
            Originally from http://wiki.scipy.org/Cookbook/Matplotlib/Loading_a_colormap_dynamically
            Modifications : - move imports outside function
        - replace Numeric by numpy
        - remove GMTPath argument, fileName is now the complete path of ctp file
        - file opening
            """
            with open(fileName) as f:
                lines = f.readlines()

            x = []
            r = []
            g = []
            b = []
            colorModel = "RGB"
            for line in lines:
                ls = line.split()
                if line[0] == "#":
                    if ls[-1] == "HSV":
                        colorModel = "HSV"
                        continue
                    else:
                        continue
                if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
                    pass
                else:
                    x.append(float(ls[0]))
                    r.append(float(ls[1]))
                    g.append(float(ls[2]))
                    b.append(float(ls[3]))
                    xtemp = float(ls[4])
                    rtemp = float(ls[5])
                    gtemp = float(ls[6])
                    btemp = float(ls[7])

            x.append(xtemp)
            r.append(rtemp)
            g.append(gtemp)
            b.append(btemp)

            nTable = len(r)
            x = np.array(x)
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            if colorModel == "HSV":
                for i in range(r.shape[0]):
                    rr,gg,bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
                    r[i] = rr
                    g[i] = gg
                    b[i] = bb
            if colorModel == "HSV":
                for i in range(r.shape[0]):
                    rr,gg,bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
                    r[i] = rr
                    g[i] = gg
                    b[i] = bb
            if colorModel == "RGB":
                r = r / 255.
                g = g / 255.
                b = b / 255.
            xNorm = (x - x[0]) / (x[-1] - x[0])

            red = []
            blue = []
            green = []
            for i in range(len(x)):
                red.append([xNorm[i], r[i], r[i]])
                green.append([xNorm[i], g[i], g[i]])
                blue.append([xNorm[i], b[i], b[i]])
            colorDict = {"red":red, "green":green, "blue":blue}
            return colorDict

        clrbar = '/scale/user/mtannaou/alternance/src/visualisation/wind_faozi.cpt'

        #cmap = getColorMap(clrbar)

        colordict = gmtColormap(clrbar)
        cmap_sar = matplotlib.colors.LinearSegmentedColormap('custom', colordict)
        return cmap_sar
        