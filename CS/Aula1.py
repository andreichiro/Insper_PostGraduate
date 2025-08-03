# Aula1.py
"""
Unified geodesic toolbox with two interchangeable engines and DataFrame feature engineering.

Principles
----------------------------
* Interchangeable distance engines, easily adding new ones;
* 'GeoFacade' offers a single public surface;
* User‑friendly constructors choose the engine;

Running
-------
python Aula1.py
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from geopy.distance import geodesic  # use geopy's geodesic function

# Public symbols when users write 'from Aula1 import *'
__all__ = ["GeoFacade", "main"]

# Strategy interface                                                   #
class DistanceStrategy(ABC):
    """
    Abstract base class for every distance engine.

    1. The class inherits from 'ABC' so that we can mark methods abstract.
    2. 'distance()' and 'pairwise_distances()' form the 'contract' all 
        concrete strategies must honour.
    """

    # scalar API #
    @abstractmethod
    def distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Return the distance (km) between two latitude/longitude pairs.

        1. Parameters are decimal degrees (keeps caller convenient).
        2. Return a single 'float' – the arc length in kilometres.
        """

    # vector API #
    def pairwise_distances(
        self,
        points_a: Iterable[Tuple[float, float]],
        points_b: Iterable[Tuple[float, float]],
    ) -> list[float]:
        """
        Default O(N) implementation that simply loops over pairs.

        Step-by-step
        -----------------
        * The base class  work even for engines that do not supply a vectorised override.
        * Child classes that can vectorise (e.g. haversine) provide
          their own highly efficient implementation without conditionals
          in this superclass.
        """

        return [
            self.distance(*a, *b) for a, b in zip(points_a, points_b, strict=True)
        ]

    # Shared coordinate validator – can be overridden per‑strategy       #
    @staticmethod
    def _validate_coords(*coords: float) -> None:
        """
        Guard‑rail for latitude/longitude tuples.

        Raises
        ------
        ValueError
            * if any value is NaN or ±Inf;
            * if any latitude ∉ [‑90, 90] or longitude ∉ [‑180, 180].

        Notes
        -----
        * Being a `@staticmethod` keeps it reusable and side‑effect‑free.
        * Individual strategy classes **may override** this method when
          they need ellipsoid‑specific or datum‑specific checks, thereby
          respecting the *Open/Closed* principle without code repetition.
        """
        lats = coords[0::2]                      # even indices → latitudes
        lons = coords[1::2]                      # odd indices  → longitudes
        if not (
            all(np.isfinite(coords))             # finite numbers
            and all(-90 <= lat <= 90 for lat in lats)
            and all(-180 <= lon <= 180 for lon in lons)
        ):
            raise ValueError(
                "Coordinates must be finite; "
                "latitude ∈ [‑90, 90], longitude ∈ [‑180, 180]."
            )

# Concrete Strategy 1 – Spherical‑law‑of‑cosines                       #
@dataclass(slots=True, frozen=True)
class SphericalCosineStrategy(DistanceStrategy):
    """
    Dependency‑free engine using the spherical‑law‑of‑cosines.

    For sides: cos(c) = cos(a) * cos(b) + sin(a) * sin(b) * cos(C) 
    For angles: cos(C) = -cos(A) * cos(B) + sin(A) * sin(B) * cos(c) 

    1. '@dataclass' autogenerates boilerplate ('__init__', '__repr__').
    2. 'slots=True' removes '__dict__', saving memory for many instances.
    3. 'frozen=True' makes the object immutable   hashable & side‑effect‑free.
    """

    radius_km: float = 6_371.01  # Mean Earth radius

    # private helpers #
    @staticmethod
    def _deg_to_rad(angle: float) -> float:
        """
        Convert degrees to radians.

        1. Wraps 'math.radians' so we avoid repeating it.
        2. Marked '@staticmethod' because it does not touch 'self'.
        """
        return math.radians(angle)

    # scalar API #
    def distance(self, lat1, lon1, lat2, lon2) -> float: 
        """
        Actual great‑circle formula implementation.

        1. 'map(self._deg_to_rad, …)' converts the 4 inputs to radians in
           one terse call.
        2. 'cos_angle' computes the cosine of the central angle *θ* using
           the spherical‑law‑of‑cosines.
        3. 'max(-1.0, min(1.0, cos_angle))' clamps rounding errors that
           might push the value outside [‑1, 1], preventing 'acos' from
           raising a domain error.
        4. Multiply by 'radius_km' to convert angle   distance.
        """
        self._validate_coords(lat1, lon1, lat2, lon2)

        φ1, λ1, φ2, λ2 = map(self._deg_to_rad, (lat1, lon1, lat2, lon2))
        cos_angle = (
            math.sin(φ1) * math.sin(φ2)
            + math.cos(φ1) * math.cos(φ2) * math.cos(λ1 - λ2)
        )
        cos_angle = math.copysign(1.0, cos_angle) if abs(cos_angle) > 1 else cos_angle

        return self.radius_km * math.acos(cos_angle)

# Concrete Strategy 2 – Haversine (NumPy + scikit‑learn)               #
class HaversineVectorisedStrategy(DistanceStrategy):
    """
    Vectorised engine that calls 'sklearn.metrics.pairwise.haversine_distances'.

    'The Haversine (or great circle) distance is the angular distance between 
    two points on the surface of a sphere. The first coordinate of each point is assumed to be the latitude, 
    the second is the longitude, given in radians. The dimension of the data must be 2.

    From Scikit-learn pairwise.py documentation:

       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x_{lat} - y_{lat}) / 2)
                                + \\cos(x_{lat})\\cos(y_{lat})\\
                                sin^2((x_{lon} - y_{lon}) / 2)}]

        Where X is an array-like, sparse matrix of shape n_samples_X, 2;                             
        And Y is an array-like, sparse matrix of shape n_samples_Y, 2
        Returning the distance matrix as a ndarray of shape (n_samples_X, n_samples_Y)

    Advantages
    ----------
    * Numerical stability for small angular separations.
    * Vectorised pairwise kernel in Cython, fast for large N.
    """

    RADIUS_KM: float = 6_371.008_8  

    # scalar API #
    def distance(self, lat1, lon1, lat2, lon2) -> float:  
        """
        Compute the distance using a tiny 2×2 haversine matrix.

        1. Build a 2×2 array with both points (shape == (2, 2)).
        2. Convert to radians once, saving two calls.
        3. Call 'haversine_distances',result is a full matrix; we grab
           '[0, 1]', the off‑diagonal entry.
        4. Multiply by mean Earth radius to shift from radians to km.
        """
        self._validate_coords(lat1, lon1, lat2, lon2)

        lat_rad = np.radians([[lat1, lon1], [lat2, lon2]])
        return haversine_distances(lat_rad)[0, 1] * self.RADIUS_KM

    # vector API (override) #
    def pairwise_distances(
        self,
        points_a: Iterable[Tuple[float, float]],
        points_b: Iterable[Tuple[float, float]],
    ) -> list[float]:
        """
        High‑throughput diagonal extraction:

        1. Cast the iterables to NumPy arrays ('dtype=float' avoids
           object arrays).
        2. 'np.radians' vectorises conversion.
        3. The kernel returns an *n × n* matrix; '.diagonal()' gives the
           element‑wise distances we want in O(n) time.
        """
        a = np.array(list(points_a), dtype=float)
        b = np.array(list(points_b), dtype=float)
        d_rad = haversine_distances(np.radians(a), np.radians(b)).diagonal()
        return (d_rad * self.RADIUS_KM).tolist()

# New strategy for WGS84 ellipsoidal distances (Vincenty/Geodesic)
class GeodesicStrategy(DistanceStrategy):
    def __init__(self):
        self._geodesic = geodesic  # store reference to geodesic function

    def distance(self, lat1, lon1, lat2, lon2) -> float:
        self._validate_coords(lat1, lon1, lat2, lon2)
        # geodesic returns a Distance object; get kilometers
        return self._geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Feature‑engineering helper #
class DistanceFeatureEngineer:
    """
    Compute extra geodesic features for a 'pandas.DataFrame'.

    Features available
    ------------------
    * distance_km   – scalar distance via the chosen engine.
    * bearing_deg   – initial bearing (forward azimuth).
    * delta_lat     – raw delta latitude in degrees.
    * delta_lon     – raw delta longitude in degrees.
    """

    RADIUS_KM: float = HaversineVectorisedStrategy.RADIUS_KM

    # public API #
    def add_features(
        self,
        df: pd.DataFrame,
        *,
        lat1: str,
        lon1: str,
        lat2: str,
        lon2: str,
        distance_func: Callable[[float, float, float, float], float] | None = None,

        prefix: str | None = "geo",
        inplace: bool = False,
        features: tuple[
            Literal["distance_km", "bearing_deg", "delta_lat", "delta_lon"]
        ] = ("distance_km", "bearing_deg"),
    ) -> pd.DataFrame:
        """
        Attach the chosen features to df.

        Walkthrough
        ------------------------
        1. 'work = df if inplace else df.copy()' – for caller's mutability choice.
        2. Extract columns and convert to NumPy arrays for vector maths.
        3. When 'distance_km' is requested:
           3.1 If a custom 'distance_func' is supplied, vectorise with
                'Series.apply'; otherwise fall back to haversine kernel.
        4. When 'bearing_deg' is requested – apply the classic azimuth formula.
        5. Plain deltas are cheap subtractions.
        6. Return the mutated DataFrame for chaining.
        """

        work = df if inplace else df.copy()

        # Common radian arrays #
        lat1_rad, lon1_rad = np.radians(work[[lat1, lon1]].values.T)
        lat2_rad, lon2_rad = np.radians(work[[lat2, lon2]].values.T)

        # feature map #
        def _distance_series() -> pd.Series:
            """
            Compute distance column.

            1. If caller supplied a custom 'distance_func', apply it row‑wise
               (vector‑safe when backed by NumPy).
            2. Else fall back to a diagonal haversine extraction.
            """
            if distance_func:
                return work.apply(
                    lambda s: distance_func(
                        s[lat1], s[lon1], s[lat2], s[lon2]
                    ),
                    axis=1,
                )
            d_rad = haversine_distances(
                np.column_stack((lat1_rad, lon1_rad)),
                np.column_stack((lat2_rad, lon2_rad)),
            ).diagonal()
            return pd.Series(d_rad * self.RADIUS_KM, index=work.index)

        def _bearing_series() -> pd.Series:
            """Initial bearing via the forward‑azimuth formula."""
            y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
            x = (
                np.cos(lat1_rad) * np.sin(lat2_rad)
                - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
            )
            return pd.Series(
                (np.degrees(np.arctan2(y, x)) + 360) % 360, index=work.index
            )

        feature_funcs: dict[str, callable[[], pd.Series]] = {
            "distance_km": _distance_series,
            "bearing_deg": _bearing_series,
            "delta_lat": lambda: work[lat2] - work[lat1],
            "delta_lon": lambda: work[lon2] - work[lon1],
        }

        # compute loop, populating a new column whose name is derived by '_col', using the lazily‑evaluated Series returned by the map
        for feat in features:
            work[self._col(prefix, feat)] = feature_funcs[feat]()

        return work

    # private helper #
    @staticmethod
    def _col(prefix: str | None, name: str) -> str:
        """
        Build an output column name.

        Step‑by‑step
        ------------
        1. If prefix is truthy (e.g. "geo"), concatenate it with an
           underscore and name
        2. If prefix is None/empty, return 'name' unchanged.
        3. The function is a '@staticmethod' because it relies solely
           on its inputs and has no side‑effects or need for 'self'.
        """
        return f"{prefix}_{name}" if prefix else name

# Facade                                                              #
class GeoFacade:
    """
    One‑stop entry‑point that encapsulates complexity.

    1. Holds a reference to any object conforming to 'DistanceStrategy'.
    2. Exposes thin pass‑through methods ('distance', 'pairwise_distances').
    3. Offers 'attach_features' that composes with 'DistanceFeatureEngineer'.
    """

    # construction #
    def __init__(self, strategy: DistanceStrategy) -> None:
        """Store the injected strategy; no other state is held."""
        self._strategy = strategy

    """Registry mapping backend names → concrete strategy classes."""
    _BACKENDS: dict[str, type[DistanceStrategy]] = {
        "simple": SphericalCosineStrategy,
        "haversine": HaversineVectorisedStrategy,
        "geodesic": GeodesicStrategy,
    }

    # factory method #
    @classmethod
    def from_backend(cls, name: Literal["simple", "haversine"]) -> "GeoFacade":
        """
        Instantiate GeoFacade with the chosen backend.

        Step‑by‑step
        ------------
        1. Look up name in the '_BACKENDS' dict (O(1)).
        2. If absent, raise 'ValueError' listing available keys, keeping the
           error message self‑healing when new engines are registered.
        3. Otherwise, call the constructor held in the dict and inject it
           into 'cls'.
        """
        try:
            strategy_cls = cls._BACKENDS[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown backend '{name}'. "
                f"Available: {list(cls._BACKENDS)}"
            ) from err
        return cls(strategy_cls())

    # thin façade #
    def distance(self, *args, **kwargs):
        """Delegate directly to the strategy’s 'distance' method."""
        return self._strategy.distance(*args, **kwargs)

    def pairwise_distances(self, *args, **kwargs):
        """Delegate to the strategy’s 'pairwise_distances' method."""
        return self._strategy.pairwise_distances(*args, **kwargs)

    # DataFrame helper #
    def attach_features(self, *args, **kwargs):
        """
        Convenience wrapper around 'DistanceFeatureEngineer.add_features'.

        Injects the strategy's scalar 'distance' into the engineer
        so the DataFrame uses the same backend as scalar calls.
        """
        kwargs["distance_func"] = self._strategy.distance
        return DistanceFeatureEngineer().add_features(*args, **kwargs)


# Main demo #
def main() -> None:
    """

    1. Calculates a single London–Paris distance via the simple engine.
    2. Creates a tiny DataFrame, enriches it with distance + bearing via
       the haversine backend.

    3. Examples: 
    Brasília (-15.699244, -47.829556)
    Tókio (35.652832, 139.879478)
    """
    # single‑pair #
    simple_geo = GeoFacade.from_backend("simple")
    km = simple_geo.distance(-15.699244, -47.829556, 35.652832, 139.879478)
    print(f"Brasília -> Tókio (simple): {km:.3f} km")

    # batch DataFrame #
    trips = pd.DataFrame(
        {
            "lat_a": [40.7128, 34.0522],
            "lon_a": [-74.0060, -118.2437],
            "lat_b": [37.7749, 47.6062],
            "lon_b": [-122.4194, -122.3321],
        }
    )
    hav_geo = GeoFacade.from_backend("haversine")
    enriched = hav_geo.attach_features(
        trips,
        lat1="lat_a",
        lon1="lon_a",
        lat2="lat_b",
        lon2="lon_b",
        prefix="gc",
        features=("distance_km", "bearing_deg"),
        inplace=False,
    )
    print("\nHaversine with lat_a, lon_a, lat_b, lon_b:")
    print(enriched)

# Entry‑point guard 
if __name__ == "__main__":
    main()
