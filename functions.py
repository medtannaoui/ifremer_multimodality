import numpy as np
from scipy.ndimage import gaussian_filter
from datetime import datetime,timedelta
import os
import xarray as xr
import pyproj

def convert_sar_to_xy(sar_file, sargeo_df, output_dir):
    """
    Convertit un fichier SAR en coordonnées (x, y) centrées sur le cyclone.
    Ajoute les champs x_sar, y_sar et sauvegarde un nouveau NetCDF.

    sar_file : str → chemin du fichier SAR .nc
    sargeo_df : DataFrame → contient les centres (lat_centre, lon_centre)
    output_dir : str → dossier de sortie
    """

    # 1️⃣ Charger le fichier SAR
    ds = xr.open_dataset(sar_file)
    filename = os.path.basename(sar_file).replace(".nc", "")
    #print(f"🔹 Traitement du fichier : {filename}")

    # 2️⃣ Trouver le centre du cyclone correspondant
    inter_key = filename.split("_centered")[0].split("_aeqd")[0]
    match = sargeo_df[sargeo_df["inter_list_sar"].str.contains(inter_key, case=False, na=False)]

    if match.empty:
        print(f"⚠️ Aucun centre trouvé pour {filename}")
        return

    lat_centre = float(match.iloc[0]["lat_centre"])
    lon_centre = float(match.iloc[0]["lon_centre"])
    if lon_centre > 180:
        lon_centre -= 360

    # 3️⃣ Définir la projection AEQD centrée sur le cyclone
    proj_geo = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_xy = pyproj.Proj(proj='aeqd', lon_0=lon_centre, lat_0=lat_centre, datum='WGS84', units='km')

    # 4️⃣ Calculer les x, y à partir des lat/lon SAR
    x, y = pyproj.transform(
        proj_geo, proj_xy,
        ds["owiLon"].values,
        ds["owiLat"].values,
        always_xy=True
    )

    # 5️⃣ Ajouter les champs x_sar, y_sar
    ds["x_sar"] = (("owiAzSize", "owiRaSize"), x)
    ds["y_sar"] = (("owiAzSize", "owiRaSize"), y)

    # 6️⃣ Sauvegarder le nouveau fichier
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{filename}_aeqd.nc")
    #recuperer l indice de la ligne courante dans le dataframe
    current_index = sargeo_df[sargeo_df["inter_list_sar"].str.contains(inter_key, case=False, na=False)].index[0]
    #mettre a jour le chemin du fichier modifié dans le dataframe
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_modifiee.csv", index=False)  # Mettre à jour le CSV avec le nouveau chemin
    ds.to_netcdf(out_file)
    


    ds.close()


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





def find_eye_SAR(x, y, WS, vent_min, vent_oeil, r_max, r_oeil):
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)

    # Pixels dans la zone où on cherche l'oeil
    mask_zone = (dist <= r_max) & np.isfinite(WS)
    if not np.any(mask_zone):
        return np.nan, np.nan, None, None

    # Pixels candidats = vent faible
    mask_candidat = (WS <= vent_min) & mask_zone
    if not np.any(mask_candidat):
        return np.nan, np.nan, None, None

    best_score = -1
    x_eye = y_eye = np.nan
    i_eye = j_eye = None

    # Boucle sur les candidats
    for i, j in zip(*np.nonzero(mask_candidat)):
        xc, yc = x[j], y[i]

        # Disque local autour du candidat
        dist_local = np.sqrt((X - xc)**2 + (Y - yc)**2)
        mask_ring = (dist_local <= r_oeil) & np.isfinite(WS)

        # Vérifier que la zone calme est assez large (pas 1 pixel)
        calm_neighbors = np.sum((WS <= vent_min) & mask_ring)
        if calm_neighbors < 5:
            continue

        # Vent fort autour du disque
        strong = (WS >= vent_oeil) & mask_ring
        if not np.any(strong):
            continue

        # Vérification en 8 directions (45° secteurs)
        dx = X - xc
        dy = Y - yc
        angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        valid_eye = True
        for a in [0, 90, 180, 270]:     # 4 directions cardinales
            sector = (angles >= a) & (angles < a + 90) & strong
            if not np.any(sector):
                valid_eye = False
                break


        if not valid_eye:
            continue

        # Score = combien de pixels forts autour
        score = np.sum(strong)

        # Garder le meilleur
        if score > best_score:
            best_score = score
            x_eye = xc
            y_eye = yc
            i_eye = j_eye = i, j

    if best_score <= 0:
        return np.nan, np.nan, None, None,best_score

    return x_eye, y_eye, i_eye, j_eye, best_score


def is_within_deltamin(date1_str, date2_str, delta=10):
    """
    Vérifie si date1 est dans l'intervalle [date2 - 10 min, date2 + 10 min].
    
    Paramètres :
        date1_str (str) : première date, ex '2019-09-20T22:19:52.000'
        date2_str (str) : seconde date, ex '2019-09-20T22:25:00.000'
        delta : 
    
    Retourne :
        bool : True si date1 est dans l'intervalle, sinon False
    """
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    
    # Conversion en datetime
    d1 = datetime.strptime(date1_str, fmt)
    d2 = datetime.strptime(date2_str, fmt)
    
    # Création de l'intervalle ±delta minutes
    delta = timedelta(minutes=delta)
    
    return (d2 - delta) <= d1 <= (d2 + delta)




def find_center_of_strong_wind(x, y, WS, vent_oeil, r_max):
    """
    Calcule le centre du cyclone comme le centre de masse des vents forts.
    Correction : Inversion des axes à la fin si le meshgrid a interverti X et Y.
    """
    # Conservez l'appel initial qui produit les matrices de la bonne taille:
    X, Y = np.meshgrid(x, y) 
    
    # ... (le masquage est inchangé et correct car les formes correspondent)
    dist = np.sqrt(X**2 + Y**2)

    # 1. Définir la zone de recherche
    mask_zone = (dist <= r_max) & np.isfinite(WS)

    # 2. Identifier les pixels de vent fort
    mask_strong = (WS >= vent_oeil) & mask_zone

    if not np.any(mask_strong):
        return np.nan, np.nan

    # 3. Récupérer les coordonnées des pixels forts
    X_strong = X[mask_strong]
    Y_strong = Y[mask_strong]

    # 4. Calculer le centre de masse (Centroid)
    # SI X et Y ont été intervertis par le meshgrid, nous faisons la correction ici.
    # Dans le cas d'une erreur d'inversion des coordonnées X et Y, 
    # X_strong contient en fait les coordonnées de l'axe Y, et vice-versa.
    
    y_center = np.mean(X_strong) # Utiliser la moyenne de X_strong pour Y
    x_center = np.mean(Y_strong) # Utiliser la moyenne de Y_strong pour X

    return x_center, y_center # Retourne le X corrigé et le Y corrigé