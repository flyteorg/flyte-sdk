# /// script
# dependencies = [
#    "geopandas",
#    "shapely",
#    "pyarrow",
#    "numpy",
#    "flyte",
# ]
# ///

"""
Test: geopandas GeoDataFrame nested inside a dataclass via SerializableType.

This demonstrates the pattern for making a third-party type (gpd.GeoDataFrame)
work inside a dataclass when passed between Flyte tasks. The key insight is that
Flyte serializes dataclasses via mashumaro's MessagePackEncoder, which doesn't
know about Flyte TypeTransformers. So the type must implement mashumaro's
SerializableType interface (_serialize / _deserialize) to tell mashumaro how
to convert it to/from a msgpack-friendly dict.

For types you control, you can add SerializableType directly. For third-party
types like gpd.GeoDataFrame, you need a thin wrapper.

Usage:
    pip install geopandas shapely

    # Run unit tests (no Union connection needed):
    python test_geopandas_in_dataclass.py --unit

    # Run remote workflow test (needs Union connection + flyte.init_from_config):
    python test_geopandas_in_dataclass.py --workflow
"""

import asyncio
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
from mashumaro.codecs.msgpack import MessagePackDecoder, MessagePackEncoder
from mashumaro.types import SerializableType
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# GeoDF: a SerializableType wrapper around gpd.GeoDataFrame
# ---------------------------------------------------------------------------

GEOPANDAS_PARQUET = "geopandas/parquet"


def _get_storage_base_dir() -> str:
    """Get a base directory for storing parquet files.

    Uses Flyte remote storage when available (inside a task pod),
    falls back to a local temp directory otherwise.
    """
    try:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        if ctx.has_raw_data:
            return str(ctx.raw_data.get_random_remote_path())
    except Exception:
        pass
    return tempfile.mkdtemp(prefix="geodf_")


def _get_storage_options(uri: str) -> Optional[Dict[str, Any]]:
    """Get fsspec storage options for writing to remote URIs (S3/GCS)."""
    try:
        import flyte.storage as storage

        if storage.is_remote(uri):
            if uri.startswith("s3"):
                return storage.get_configured_fsspec_kwargs("s3")
            elif uri.startswith("gs"):
                return storage.get_configured_fsspec_kwargs("gs")
            return {}
    except Exception:
        pass
    return None


@dataclass
class GeoDF(SerializableType):
    """Thin wrapper that lets a gpd.GeoDataFrame live inside a dataclass.

    On serialize: writes the GeoDataFrame to a parquet file (local or remote)
    and stores only the URI. On deserialize: reads the parquet back lazily.

    Works both locally and in remote Flyte executions -- when running inside
    a task pod, writes to the configured remote storage (S3/GCS). When running
    locally, writes to a temp directory.
    """

    uri: Optional[str] = None

    # Private fields excluded from mashumaro serialization (init=False).
    _gdf: Optional[gpd.GeoDataFrame] = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "GeoDF":
        """Create a GeoDF from an in-memory GeoDataFrame."""
        obj = cls()
        obj._gdf = gdf
        return obj

    # -- mashumaro SerializableType interface --

    def _serialize(self) -> Dict[str, Any]:
        if self.uri is not None:
            return {"uri": self.uri, "format": GEOPANDAS_PARQUET}

        if self._gdf is None:
            raise ValueError("No GeoDataFrame to serialize")

        remote_dir = _get_storage_base_dir()

        try:
            import flyte.storage as storage

            if storage.is_remote(remote_dir):
                # Write locally first, then upload to remote storage
                local_dir = tempfile.mkdtemp(prefix="geodf_")
                local_path = str(Path(local_dir) / "data.parquet")
                self._gdf.to_parquet(local_path)
                remote_path = remote_dir.rstrip("/") + "/data.parquet"
                from flyte._utils.asyn import loop_manager
                loop_manager.run_sync(storage.put, local_path, remote_path)
                self.uri = remote_dir
                return {"uri": self.uri, "format": GEOPANDAS_PARQUET}
        except ImportError:
            pass

        # Local-only path
        Path(remote_dir).mkdir(parents=True, exist_ok=True)
        local_path = str(Path(remote_dir) / "data.parquet")
        self._gdf.to_parquet(local_path)
        self.uri = remote_dir

        return {"uri": self.uri, "format": GEOPANDAS_PARQUET}

    @classmethod
    def _deserialize(cls, value: Dict[str, Any]) -> "GeoDF":
        uri = value.get("uri")
        if uri is None:
            raise ValueError("Cannot deserialize GeoDF without a URI")
        obj = cls(uri=uri)
        return obj

    @property
    def df(self) -> gpd.GeoDataFrame:
        """Materialize the GeoDataFrame (reads from parquet if needed)."""
        if self._gdf is None:
            if self.uri is None:
                raise ValueError("No GeoDataFrame or URI available")
            remote_path = self.uri.rstrip("/") + "/data.parquet"
            try:
                import flyte.storage as storage

                if storage.is_remote(remote_path):
                    local_path = str(storage.get_random_local_path("data.parquet"))
                    from flyte._utils.asyn import loop_manager
                    loop_manager.run_sync(storage.get, remote_path, local_path)
                    self._gdf = gpd.read_parquet(local_path)
                    return self._gdf
            except ImportError:
                pass
            self._gdf = gpd.read_parquet(remote_path)
        return self._gdf


# ---------------------------------------------------------------------------
# Dataclasses that use GeoDF as a field
# ---------------------------------------------------------------------------


@dataclass
class SpatialResult:
    name: str
    geodata: GeoDF
    score: float


@dataclass
class MultiGeoResult:
    label: str
    layers: List[GeoDF]
    metadata: Dict[str, str]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_sample_gdf(n: int = 5) -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame with random points."""
    rng = np.random.default_rng(42)
    lons = rng.uniform(-122.5, -122.3, n)
    lats = rng.uniform(37.7, 37.8, n)
    return gpd.GeoDataFrame(
        {"name": [f"point_{i}" for i in range(n)], "value": rng.integers(0, 100, n)},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


# ===========================================================================
# Layer 1: mashumaro MessagePack roundtrip (what DataclassTransformer uses)
# ===========================================================================


def test_geodf_roundtrip():
    """Test that a single GeoDF survives msgpack encode/decode."""
    print("Test 1: GeoDF standalone msgpack roundtrip")
    gdf = make_sample_gdf()
    wrapper = GeoDF.from_gdf(gdf)

    encoder = MessagePackEncoder(GeoDF)
    decoder = MessagePackDecoder(GeoDF)

    packed = encoder.encode(wrapper)
    restored: GeoDF = decoder.decode(packed)

    assert restored.uri is not None, "URI should be set after deserialization"
    restored_gdf = restored.df
    assert len(restored_gdf) == len(gdf), f"Expected {len(gdf)} rows, got {len(restored_gdf)}"
    assert list(restored_gdf.columns) == list(gdf.columns)
    assert restored_gdf.crs == gdf.crs
    print(f"  OK: {len(restored_gdf)} rows, CRS preserved")


def test_geodf_in_dataclass():
    """Test that a dataclass containing GeoDF survives msgpack encode/decode."""
    print("Test 2: GeoDF inside a dataclass (SpatialResult) - msgpack")
    gdf = make_sample_gdf(10)
    result = SpatialResult(
        name="san_francisco",
        geodata=GeoDF.from_gdf(gdf),
        score=0.95,
    )

    encoder = MessagePackEncoder(SpatialResult)
    decoder = MessagePackDecoder(SpatialResult)

    packed = encoder.encode(result)
    restored: SpatialResult = decoder.decode(packed)

    assert restored.name == "san_francisco"
    assert restored.score == 0.95
    assert restored.geodata.uri is not None
    restored_gdf = restored.geodata.df
    assert len(restored_gdf) == 10
    assert "geometry" in restored_gdf.columns
    print(f"  OK: name={restored.name}, score={restored.score}, rows={len(restored_gdf)}")


def test_multiple_geodfs_in_dataclass():
    """Test a dataclass with a list of GeoDFs."""
    print("Test 3: Multiple GeoDFs in a dataclass (MultiGeoResult) - msgpack")
    gdf1 = make_sample_gdf(3)
    gdf2 = make_sample_gdf(7)

    result = MultiGeoResult(
        label="multi_layer",
        layers=[GeoDF.from_gdf(gdf1), GeoDF.from_gdf(gdf2)],
        metadata={"source": "test", "version": "1.0"},
    )

    encoder = MessagePackEncoder(MultiGeoResult)
    decoder = MessagePackDecoder(MultiGeoResult)

    packed = encoder.encode(result)
    restored: MultiGeoResult = decoder.decode(packed)

    assert restored.label == "multi_layer"
    assert len(restored.layers) == 2
    assert restored.metadata == {"source": "test", "version": "1.0"}
    assert len(restored.layers[0].df) == 3
    assert len(restored.layers[1].df) == 7
    print(f"  OK: label={restored.label}, layers=[{len(restored.layers[0].df)}, {len(restored.layers[1].df)}] rows")


def test_geodf_lazy_load():
    """Test that deserialized GeoDF lazily loads from parquet."""
    print("Test 4: Lazy loading from URI")
    gdf = make_sample_gdf(4)
    wrapper = GeoDF.from_gdf(gdf)

    encoder = MessagePackEncoder(GeoDF)
    decoder = MessagePackDecoder(GeoDF)

    packed = encoder.encode(wrapper)
    restored: GeoDF = decoder.decode(packed)

    assert restored._gdf is None, "GeoDataFrame should not be loaded yet"
    loaded = restored.df
    assert loaded is not None
    assert len(loaded) == 4
    print(f"  OK: lazy loaded {len(loaded)} rows from {restored.uri}")


# ===========================================================================
# Layer 2: Flyte TypeEngine roundtrip (DataclassTransformer to_literal / to_python_value)
# ===========================================================================


def test_type_engine_roundtrip():
    """Test SpatialResult through Flyte's DataclassTransformer."""
    print("Test 5: Flyte TypeEngine roundtrip (DataclassTransformer)")
    from flyte.types import TypeEngine

    gdf = make_sample_gdf(6)
    original = SpatialResult(
        name="type_engine_test",
        geodata=GeoDF.from_gdf(gdf),
        score=0.42,
    )

    lit_type = TypeEngine.to_literal_type(SpatialResult)
    literal = asyncio.run(TypeEngine.to_literal(original, SpatialResult, lit_type))
    restored: SpatialResult = asyncio.run(TypeEngine.to_python_value(literal, SpatialResult))

    assert restored.name == "type_engine_test"
    assert restored.score == 0.42
    assert restored.geodata.uri is not None
    restored_gdf = restored.geodata.df
    assert len(restored_gdf) == 6
    assert restored_gdf.crs is not None
    print(f"  OK: name={restored.name}, score={restored.score}, rows={len(restored_gdf)}")


def test_type_engine_multi_roundtrip():
    """Test MultiGeoResult through Flyte's DataclassTransformer."""
    print("Test 6: Flyte TypeEngine roundtrip (MultiGeoResult)")
    from flyte.types import TypeEngine

    gdf1 = make_sample_gdf(3)
    gdf2 = make_sample_gdf(8)
    original = MultiGeoResult(
        label="engine_multi",
        layers=[GeoDF.from_gdf(gdf1), GeoDF.from_gdf(gdf2)],
        metadata={"engine": "test"},
    )

    lit_type = TypeEngine.to_literal_type(MultiGeoResult)
    literal = asyncio.run(TypeEngine.to_literal(original, MultiGeoResult, lit_type))
    restored: MultiGeoResult = asyncio.run(TypeEngine.to_python_value(literal, MultiGeoResult))

    assert restored.label == "engine_multi"
    assert len(restored.layers) == 2
    assert len(restored.layers[0].df) == 3
    assert len(restored.layers[1].df) == 8
    assert restored.metadata == {"engine": "test"}
    print(f"  OK: label={restored.label}, layers=[{len(restored.layers[0].df)}, {len(restored.layers[1].df)}] rows")


# ===========================================================================
# Layer 3: Remote workflow test (tasks at module level for resolver)
# ===========================================================================

try:
    import flyte

    _wf_env = flyte.TaskEnvironment(
        name="geodf_test",
        image=flyte.Image.from_uv_script(__file__, name="geodf-test"),
    )

    @_wf_env.task
    async def produce_spatial_result() -> SpatialResult:
        gdf = make_sample_gdf(10)
        return SpatialResult(
            name="remote_test",
            geodata=GeoDF.from_gdf(gdf),
            score=0.99,
        )

    @_wf_env.task
    async def consume_spatial_result(result: SpatialResult) -> str:
        gdf = result.geodata.df
        return f"Got {len(gdf)} rows, name={result.name}, score={result.score}, crs={gdf.crs}"

    @_wf_env.task
    async def produce_multi() -> MultiGeoResult:
        gdf1 = make_sample_gdf(4)
        gdf2 = make_sample_gdf(8)
        return MultiGeoResult(
            label="multi_remote",
            layers=[GeoDF.from_gdf(gdf1), GeoDF.from_gdf(gdf2)],
            metadata={"test": "remote"},
        )

    @_wf_env.task
    async def consume_multi(result: MultiGeoResult) -> str:
        rows = [len(layer.df) for layer in result.layers]
        return f"Got {len(result.layers)} layers with {rows} rows, label={result.label}"

    @_wf_env.task
    async def main() -> str:
        spatial = await produce_spatial_result()
        summary1 = await consume_spatial_result(result=spatial)

        multi = await produce_multi()
        summary2 = await consume_multi(result=multi)

        return f"{summary1} | {summary2}"

except ImportError:
    pass


def run_workflow_test():
    """Test GeoDF inside a dataclass across actual Flyte task boundaries."""
    flyte.init_from_config(root_dir=Path(__file__).parent)

    print("Submitting geodf workflow to Union...")
    run = flyte.run(main)
    print(f"  Run: {run.name}")
    print(f"  URL: {run.url}")
    run.wait()
    print("  Done!")


# ===========================================================================
# Main
# ===========================================================================


def run_unit_tests():
    print("=" * 60)
    print("Layer 1: mashumaro MessagePack roundtrip")
    print("=" * 60)
    test_geodf_roundtrip()
    test_geodf_in_dataclass()
    test_multiple_geodfs_in_dataclass()
    test_geodf_lazy_load()

    print()
    print("=" * 60)
    print("Layer 2: Flyte TypeEngine roundtrip")
    print("=" * 60)
    test_type_engine_roundtrip()
    test_type_engine_multi_roundtrip()

    print()
    print("All unit tests passed!")


if __name__ == "__main__":
    if "--workflow" in sys.argv:
        run_workflow_test()
    elif "--unit" in sys.argv or len(sys.argv) == 1:
        run_unit_tests()
    else:
        print("Usage: python test_geopandas_in_dataclass.py [--unit | --workflow]")
