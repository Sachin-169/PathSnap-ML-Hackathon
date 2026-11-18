
from pyproj import Transformer


def get_utm_transformer(longitudes, latitudes):
    """
    Select a UTM coordinate system based on the centre
    of the trip's geographic coordinates.

    Returns:
        transformer: WGS84 to the selected UTM projection
        epsg_code: EPSG code of the selected projection
    """

    longitude = float(longitudes.mean())
    latitude = float(latitudes.mean())

    if not (-180 <= longitude <= 180):
        raise ValueError(f"Invalid longitude centre: {longitude}")

    if not (-90 <= latitude <= 90):
        raise ValueError(f"Invalid latitude centre: {latitude}")

    # UTM covers latitudes from 80 degrees south
    # to 84 degrees north.
    if not (-80 <= latitude <= 84):
        raise ValueError(
            f"Trip latitude {latitude:.4f} is outside UTM coverage."
        )

    # Determine the UTM zone from longitude.
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    # Northern hemisphere: EPSG:326xx
    # Southern hemisphere: EPSG:327xx
    if latitude >= 0:
        epsg_code = 32600 + zone
    else:
        epsg_code = 32700 + zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True
    )

    return transformer, epsg_code
