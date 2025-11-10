from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


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
    def cmap_sar(cls, return_vminvmax=False, return_levels=False):
            # seuils NOAA (en kt)
        levels = np.array([0, 17, 34, 50, 64, 83, 96, 113, 137, 150], dtype=float)

        colors = [
        "#ffffff",  # 0
        "#c8f1dd",  # 17 vert clair
        "#6ad1df",  # 34 turquoise
        "#ff72bd",  # 50 rose saturé
        "#ffa031",  # 64 orange
        "#8d5dd1",  # 83 violet
        "#3b2e7f",  # 96 violet foncé
        "#0d0c29",  # 113 presque noir
        "#010014",  # 137 presque noir
        "#000000"  # 150 noir
        ]

        name = "sar_noaa"
        return cls.create_cmap(
            values=levels,
            colors=colors,
            name=name,
            return_vminvmax=return_vminvmax,
            return_levels=return_levels
        )

