import numpy as np
from scipy.ndimage import gaussian_filter

def _disk_mask(X, Y, rmax_km):
    """
    Crée un masque booléen pour ne garder que les pixels
    à l'intérieur d'un disque de rayon rmax_km autour de (0,0).
    """
    R = np.hypot(X, Y)  # distance radiale
    return (R <= rmax_km)


def find_eye_IR(x, y, BT, rmax_km=250, sigma=2.0):
    """
    Trouve le centre de l'œil du cyclone dans une image IR.
    
    Paramètres :
        x, y : 1D arrays (km)
        BT : 2D array (Brightness Temperature en K)
        rmax_km : rayon max pour la recherche (km)
        sigma : écart-type du lissage gaussien
        
    Retour :
        x_eye, y_eye : coordonnées (km)
        i, j : indices du pixel trouvé
    """
    X, Y = np.meshgrid(x, y)
    M = _disk_mask(X, Y, rmax_km)
    BTs = gaussian_filter(BT, sigma=sigma)         # lissage
    BTs_masked = np.where(M, BTs, -np.inf)         # on cherche un MAX
    i, j = np.unravel_index(np.nanargmax(BTs_masked), BT.shape)
    return x[j], y[i], i, j


def find_eye_SAR(x, y, WS, rmax_km=250, sigma=1.5, p=10):
    """
    Trouve le centre de l'œil du cyclone dans un champ SAR (vent).
    
    Paramètres :
        x, y : 1D arrays (km)
        WS : 2D array (vitesse du vent SAR)
        rmax_km : rayon max pour la recherche (km)
        sigma : lissage gaussien
        p : percentile pour définir la 'zone calme' (ex. 10 → 10% les plus faibles)
    
    Retour :
        x_eye, y_eye : coordonnées (km)
        i, j : indices du pixel trouvé
    """
    X, Y = np.meshgrid(x, y)
    M = _disk_mask(X, Y, rmax_km)
    WSs = gaussian_filter(WS, sigma=sigma)

    # valeurs valides dans le disque
    vals = WSs[M & np.isfinite(WSs)]
    if vals.size == 0:
        return np.nan, np.nan, None, None

    # seuil calme (10% les plus faibles vitesses)
    thr = np.percentile(vals, p)
    calm = (WSs <= thr) & M & np.isfinite(WSs)

    if not np.any(calm):
        # fallback : pixel min simple
        WSs_masked = np.where(M, WSs, np.inf)
        i, j = np.unravel_index(np.nanargmin(WSs_masked), WS.shape)
        return x[j], y[i], i, j

    # centroïde de la zone calme
    yy, xx = np.nonzero(calm)
    x_eye = x[xx].mean()
    y_eye = y[yy].mean()
    j = np.argmin(np.abs(x - x_eye))
    i = np.argmin(np.abs(y - y_eye))
    return x_eye, y_eye, i, j
