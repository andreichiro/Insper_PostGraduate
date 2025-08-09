#Aula1.py
#Calcula distâncias

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from geopy.distance import geodesic

__all__ = ["GeoFacade", "main"]

#Strategy Interface                                                   #
class DistanceStrategy(ABC):
    """
    Classe abstrata p/ calcular qq distância.
    1. 'distance()' e 'pairwise_distances()' formam o "contrato" 
    """
    @abstractmethod
    def distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Retorna a distância (km) entre dois pares de latitude/longitude.

        1. Parâmetros em float
        2. Retorna um único float
        """

    def pairwise_distances(
        self,
        points_a: Iterable[Tuple[float, float]],
        points_b: Iterable[Tuple[float, float]],
    ) -> list[float]:
        """
        Calcula n distâncias usando zip (iteráveis em tuplas)
        """
        return [
            self.distance(*a, *b) for a, b in zip(points_a, points_b, strict=True)
        ]

    #Valida coordenadas, pode override 
    @staticmethod
    def _validate_coords(*coords: float) -> None:
        """
        Tuplas p/ latitude/longitude:

        ValueError
            Se qualquer valor for NaN 
            Latitude precisa estar entre ‑90, 90 e longitude entre ‑180 e 180

        Método estático, sem self
        """
        lats = coords[0::2]                    
        lons = coords[1::2]               
        if not (
            all(np.isfinite(coords))           
            and all(-90 <= lat <= 90 for lat in lats)
            and all(-180 <= lon <= 180 for lon in lons)
        ):
            raise ValueError(
                "Coordenadas devem ser finitas; "
                "latitude ∈ [‑90 - 90], longitude ∈ [‑180 - 180]."
            )

#Estratégia 1 – Cossenos             
@dataclass(slots=True, frozen=True)
class SphericalCosineStrategy(DistanceStrategy):
    """
    Usa cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(C)
    """

    radius_km: float = 6_371.01 

    #Helpers 
    @staticmethod
    def _deg_to_rad(angle: float) -> float:
        """
        Converte graus em radianos.
        """
        return math.radians(angle)

    def distance(self, lat1, lon1, lat2, lon2) -> float:
        """
        Calcula a distância via lei esferica dos cossenos
        """
        self._validate_coords(lat1, lon1, lat2, lon2)

        #Graus p/ radianos 
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
            self._deg_to_rad, (lat1, lon1, lat2, lon2)
        )

        #Lei esférica dos cossenos
        cos_angle = (
            math.sin(lat1_rad) * math.sin(lat2_rad)
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon1_rad - lon2_rad)
        )

        #Clamp
        cos_angle = math.copysign(1.0, cos_angle) if abs(cos_angle) > 1 else cos_angle
        return self.radius_km * math.acos(cos_angle)

#Estratégia 2 – Haversine 
class HaversineVectorisedStrategy(DistanceStrategy):
    """
    Usa sklearn.metrics.pairwise.haversine_distances:

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    """

    RADIUS_KM: float = 6_371.008_8

    def distance(self, lat1, lon1, lat2, lon2) -> float:
        """
        Calcula a distância via matriz 2×2 haversine.

        Constrói array 2×2 com os pontos,
        Converte p/ radianos,
        Chama 'haversine_distances' e pega [0, 1],
        Multiplica pelo raio p/ converter em km
        """
        self._validate_coords(lat1, lon1, lat2, lon2)

        lat_rad = np.radians([[lat1, lon1], [lat2, lon2]])
        return haversine_distances(lat_rad)[0, 1] * self.RADIUS_KM

    def pairwise_distances(
        self,
        points_a: Iterable[Tuple[float, float]],
        points_b: Iterable[Tuple[float, float]],
    ) -> list[float]:
        """
        Converte iteráveis em arrays,
        Vetoriza radianos,
        Usa '.diagonal()'
        """
        a = np.array(list(points_a), dtype=float)
        b = np.array(list(points_b), dtype=float)
        d_rad = haversine_distances(np.radians(a), np.radians(b)).diagonal()
        return (d_rad * self.RADIUS_KM).tolist()

#Estratégia 3: Distâncias elipsoidais      
class GeodesicStrategy(DistanceStrategy):
    """
    Distâncias elipsoidais referem-se à medição da distância entre dois pontos na superfície da Terra, considerando-a como um elipsoide de revolução em vez de uma esfera perfeita
    """
    def __init__(self):
        self._geodesic = geodesic  

    def distance(self, lat1, lon1, lat2, lon2) -> float:
        self._validate_coords(lat1, lon1, lat2, lon2)
        return self._geodesic((lat1, lon1), (lat2, lon2)).kilometers

#Helpers                              
class DistanceFeatureEngineer:
    """
    Calcula features geodésicas c/ dataframe pandas

    distance_km p/ distância escalar 
    bearing_deg p/ rumo inicial
    delta_lat p/ diferença bruta de latitude
    delta_lon p/ diferença bruta de longitude
    """

    RADIUS_KM: float = HaversineVectorisedStrategy.RADIUS_KM

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

        Dado um conjunto de coordenadas (lat1, lon1) ↔ (lat2, lon2), calcula e ADD colunas c/:
        distance_km, a distância em km
        bearing_deg, direção inicial em graus
        delta_lat, diferença simples de latitude (lat2 - lat1) em graus;
        delta_lon, diferença simples de longitude (lon2 - lon1) em graus.

        """
        work = df if inplace else df.copy()

        #Arrays em radianos
        lat1_rad, lon1_rad = np.radians(work[[lat1, lon1]].values.T)
        lat2_rad, lon2_rad = np.radians(work[[lat2, lon2]].values.T)

        #Gera as séries de features
        def _distance_series() -> pd.Series:
            if distance_func:
                return work.apply(
                    lambda s: distance_func(
                        s[lat1], s[lon1], s[lat2], s[lon2]
                    ),
                    axis=1,
                )
            # Distância via Haversine 
            d_rad = haversine_distances(
                np.column_stack((lat1_rad, lon1_rad)),
                np.column_stack((lat2_rad, lon2_rad)),
            ).diagonal()
            return pd.Series(d_rad * self.RADIUS_KM, index=work.index)

        def _bearing_series() -> pd.Series:
            y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
            x = (
                np.cos(lat1_rad) * np.sin(lat2_rad)
                - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
            )
            return pd.Series(
                (np.degrees(np.arctan2(y, x)) + 360) % 360, index=work.index
            )

        feature_funcs: dict[str, Callable[[], pd.Series]] = {
            "distance_km": _distance_series,
            "bearing_deg": _bearing_series,
            "delta_lat": lambda: work[lat2] - work[lat1],
            "delta_lon": lambda: work[lon2] - work[lon1],
        }
        #Cria oq foi requisitado
        for feat in features:
            work[self._col(prefix, feat)] = feature_funcs[feat]()
        return work

    #Coluna de saída
    @staticmethod
    def _col(prefix: str | None, name: str) -> str:
        """
        Constrói o nome da coluna de saída.

        1. Se prefixo é truthy (ex.: "geo"), concatena com sublinhado.
        2. Se prefixo é None/vazio, devolve 'name' inalterado.
        3. '@staticmethod' pois não depende de 'self'.
        """
        return f"{prefix}_{name}" if prefix else name

#Facade                                                                    #
class GeoFacade:
    """
    Ponto de entrada p/

    calcular a distância (sendo possível calcular várias de uma vez);
    adicionar colunas geo a um DataFrame (attach_features)
    """

    def __init__(self, strategy: DistanceStrategy) -> None:
        """Armazena a estratégia injetada; não mantém outro estado."""
        self._strategy = strategy

    #Registra a estratégia
    _BACKENDS: dict[str, type[DistanceStrategy]] = {
        "simple": SphericalCosineStrategy,
        "haversine": HaversineVectorisedStrategy,
        "geodesic": GeodesicStrategy,
    }

    #Passa p/ a estratégia interna 
    @classmethod
    def from_backend(cls, name: Literal["simple", "haversine"]) -> "GeoFacade":
        """
        Instancia GeoFacade com o backend escolhido.

        Procura 'name' em '_BACKENDS'
        Lança 'ValueError' se n existir
        Se presente, usa a estratégia e injeta em cls
        """
        try:
            strategy_cls = cls._BACKENDS[name]
        except KeyError as err:
            raise ValueError(
                f"Backend desconhecido '{name}'. "
                f"Disponíveis: {list(cls._BACKENDS)}"
            ) from err
        return cls(strategy_cls())
    
    def distance(self, *args, **kwargs):
        """Delegação direta para 'distance' da estratégia."""
        return self._strategy.distance(*args, **kwargs)

    def pairwise_distances(self, *args, **kwargs):
        """Delegação para 'pairwise_distances' da estratégia."""
        return self._strategy.pairwise_distances(*args, **kwargs)

    def attach_features(self, *args, **kwargs):
        """
        Wrapper p/ DistanceFeatureEngineer.add_features
        """
        kwargs["distance_func"] = self._strategy.distance
        return DistanceFeatureEngineer().add_features(*args, **kwargs)

#Demo
def main() -> None:
    """
    Calcula distância Brasília‑Tóquio via engine simples;
    Cria DataFrame mínimo e enriquece com distância + azimute 

    Exemplos:
    Brasília (-15.699244, -47.829556)
    Tóquio   (35.652832, 139.879478)
    """
    simple_geo = GeoFacade.from_backend("simple")
    km = simple_geo.distance(-15.699244, -47.829556, 35.652832, 139.879478)
    print(f"Brasília -> Tóquio (simple): {km:.3f} km")

    trips = pd.DataFrame(
        {
        "from_city": ["New York", "Los Angeles"],
        "to_city":   ["San Francisco", "Seattle"],
        "lat_a":  [40.7128, 34.0522],
        "lon_a":  [-74.0060, -118.2437],
        "lat_b":  [37.7749, 47.6062],
        "lon_b":  [-122.4194, -122.3321],
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

    print("\n📍 Distância e rumo")
    print(
        enriched[
            ["from_city", "to_city", "gc_distance_km", "gc_bearing_deg"]
        ].to_string(index=False, formatters={
            "gc_distance_km": "{:.1f} km".format,
            "gc_bearing_deg": "{:.1f}°".format,
        })
    )

    #legenda
    print("""
    Legenda das colunas:
    gc_distance_km – caminho mais curto sobre a superfície da Terra em KMs
    gc_bearing_deg – azimute inicial: direção de partida em graus (0° = norte, 90° = leste, 180° = sul, 270° = oeste).
    """)



#Main
if __name__ == "__main__":
    main()
