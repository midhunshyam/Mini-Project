#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gnaf orientation pipeline

miniproject.py

----------------------------
Single-script pipeline that:
1) Loads GNAF addresses (Parquet) and Roads (GeoPackage).
2) Computes nearest-road orientation for each address and writes gnaf_orientation.csv.
3) Recomputes/validates the results against source data and writes gnaf_orientation_validated.csv.
4) Logs every piece of information (schemas, counts, bboxes, QC) to a text file.

Run:
  python gnaf_orientation_pipeline.py \
    --gnaf gnaf_prop.parquet \
    --roads roads.gpkg \
    --results gnaf_orientation.csv \
    --qc gnaf_orientation_validated.csv \
    --log run_log.txt

Requirements: `pip install pandas geopandas pyarrow shapely tqdm`

Run:
python miniproejct.py \
  --gnaf gnaf_prop.parquet \
  --roads roads.gpkg \
  --epsg_metric 7856 \
  --radii 30 60 120 250 500 \
  --log miniproejct_output.txt

"""

# ==== Standard library imports ====
import argparse               # Parse command line arguments
import logging                # Structured logging to file (and optional console)
import math                   # Trigonometry for bearings
from pathlib import Path      # Safer file path handling

# ==== Third-party imports ====
import numpy as np            # Efficient numeric arrays
import pandas as pd           # Tabular data handling
import geopandas as gpd       # Geospatial data handling
import pyarrow.parquet as pq  # Reading Parquet files (GNAF)
from shapely.geometry import box  # Bounding box geometry

# ---------------------------
# Algorithm helper functions
# ---------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes all output to a text file (and to console)."""
    logger = logging.getLogger("gnaf_pipeline")              # Create/get named logger
    logger.setLevel(logging.INFO)                            # Set global level

    # File handler writes everything to the specified log file
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console handler echoes essential info to stdout (optional; can remove if you want file-only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Consistent formatting for both handlers
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # Avoid duplicate handlers if rerun in notebooks
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def bearing_deg(p_from, p_to):
    """Compute bearing in degrees from p_from -> p_to assuming EPSG:4326 coordinates."""
    lat1, lon1 = math.radians(p_from.y), math.radians(p_from.x)  # Convert 'from' to radians
    lat2, lon2 = math.radians(p_to.y), math.radians(p_to.x)      # Convert 'to' to radians
    dlon = lon2 - lon1                                           # Δlongitude
    x = math.sin(dlon) * math.cos(lat2)                          # Bearing calc helper (x)
    y = (math.cos(lat1)*math.sin(lat2)
         - math.sin(lat1)*math.cos(lat2)*math.cos(dlon))         # Bearing calc helper (y)
    brng = math.degrees(math.atan2(x, y))                        # Convert atan2 result to degrees
    return (brng + 360) % 360                                    # Normalize to [0, 360)


def bucket_compass8(deg):
    """Map a degree value to one of 8 compass directions (N, NE, E, SE, S, SW, W, NW)."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]          # 8-wind rose
    step = 360 / 8                                               # 45-degree buckets
    return dirs[int((deg + step/2) // step) % 8]                 # Round to nearest bucket


def nearest_point_on_line(pt, line):
    """Return the nearest point on a (Multi)LineString to the given point (same CRS)."""
    if line.is_empty:                                            # Guard: empty geometry
        return None
    if line.geom_type == "MultiLineString":                      # For MultiLineString, test each segment
        best_d = float("inf")
        best_pt = None
        for seg in line.geoms:
            # project(pt) gives distance along line; interpolate maps distance back to a point on the line
            cand = seg.interpolate(seg.project(pt))
            d = pt.distance(cand)
            if d < best_d:
                best_d, best_pt = d, cand
        return best_pt
    # For simple LineString, one projection is enough
    return line.interpolate(line.project(pt))


# ---------------------------
# Core pipeline steps
# ---------------------------

def load_gnaf(parquet_path: Path, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load GNAF parquet into a GeoDataFrame with EPSG:4326 points."""
    logger.info("Loading GNAF parquet: %s", parquet_path)       # Log file path
    table = pq.read_table(parquet_path, use_pandas_metadata=False)  # Read Parquet into Arrow table
    logger.info("GNAF schema:\n%s", table.schema)               # Log Arrow schema
    logger.info("GNAF rows: %s | cols: %s", table.num_rows, table.num_columns)  # Log dimensions

    df = table.to_pandas()                                      # Convert Arrow to pandas DataFrame
    has_xy = {"longitude", "latitude"}.issubset(df.columns)     # Check for coordinate columns

    if not has_xy:                                              # Fail early if no coordinate columns
        raise ValueError("GNAF file missing 'longitude'/'latitude' columns")

    gdf = gpd.GeoDataFrame(                                     # Create GeoDataFrame with point geometry
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Log a small sample and bounding box for sanity
    logger.info("GNAF head(5):\n%s", gdf.head(5).to_string(index=False))
    logger.info("GNAF total bounds [minx,miny,maxx,maxy]: %s", gdf.total_bounds.tolist())
    return gdf


def load_roads(gpkg_path: Path, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load roads GeoPackage as GeoDataFrame in EPSG:4326."""
    logger.info("Loading Roads GPKG: %s", gpkg_path)             # Log path
    roads = gpd.read_file(gpkg_path)                             # Read GeoPackage

    if roads.crs is None:                                        # If no CRS, assume WGS84
        roads = roads.set_crs(4326)
    else:
        roads = roads.to_crs(4326)                               # Reproject to WGS84

    # Log structure and key attributes
    logger.info("Roads len: %d  |  CRS: %s", len(roads), roads.crs)
    logger.info("Roads geometry types: %s", roads.geom_type.unique().tolist())
    logger.info("Roads total bounds [minx,miny,maxx,maxy]: %s", roads.total_bounds.tolist())
    logger.info("Roads head(5):\n%s", roads.head(5).to_string(index=False))
    return roads


def clip_gnaf_to_roads_bbox(gnaf: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Clip GNAF points to the overall bounding box of the roads to reduce workload."""
    bbox_gnaf = box(*gnaf.total_bounds)                          # Create shapely box from GNAF bounds
    bbox_roads = box(*roads.total_bounds)                        # Create shapely box from roads bounds
    overlap = bbox_gnaf.intersects(bbox_roads)                   # Quick bbox overlap test
    logger.info("Do datasets overlap spatially? %s", overlap)    # Log overlap boolean

    if not overlap:                                              # If no overlap, return original (nothing to clip)
        logger.info("No spatial overlap; using full GNAF dataset.")
        return gnaf.copy()

    minx, miny, maxx, maxy = roads.total_bounds                  # Unpack roads bbox
    clipped = gnaf.cx[minx:maxx, miny:maxy].copy()               # Fast clip by bounding box
    logger.info("Using %d nearby GNAF properties within roads bbox.", len(clipped))
    return clipped


def compute_orientation(gnaf: gpd.GeoDataFrame,
                        roads: gpd.GeoDataFrame,
                        results_csv: Path,
                        logger: logging.Logger,
                        epsg_metric: int = 7856,
                        radius_list=(20, 40, 80, 160, 320, 500)):
    """Compute nearest road orientation and write results CSV."""
    logger.info("Computing nearest-road orientation (metric EPSG: %s)", epsg_metric)

    gnaf_m = gnaf.to_crs(epsg_metric)                            # Project GNAF to metric CRS for distances
    roads_m = roads.to_crs(epsg_metric)                          # Project roads similarly
    sidx = roads_m.sindex                                        # Spatial index for fast neighbor queries

    results = []                                                 # List to accumulate per-address outputs

    for i, row in gnaf_m.iterrows():                             # Iterate through each property
        pt = row.geometry                                        # The property point in metric CRS
        idxs = None                                              # Reset candidate index list

        # Expand search radius until at least one road is found
        for r in radius_list:
            idxs = list(sidx.query(pt.buffer(r)))                # Query index with a buffer around point
            if idxs:
                break                                            # Stop once we have candidates

        if not idxs:                                             # If no roads found within max radius
            results.append({
                "address": row.get("address"),
                "orientation": None,
                "bearing_deg": np.nan,
                "distance_to_road_m": np.nan,
                "note": "no_road_within_max_radius",
            })
            continue

        # Among candidates, choose the actual closest by Euclidean distance
        best_d = float("inf")
        best_np = None
        for ridx in idxs:
            line = roads_m.geometry.iloc[ridx]                   # Candidate road geometry
            np_on = nearest_point_on_line(pt, line)              # Nearest point on this road to the property
            if np_on is None:
                continue
            d = pt.distance(np_on)                               # Metric distance to that nearest point
            if d < best_d:
                best_d, best_np = d, np_on                       # Keep best-so-far

        if best_np is None:                                      # Safety guard
            results.append({
                "address": row.get("address"),
                "orientation": None,
                "bearing_deg": np.nan,
                "distance_to_road_m": np.nan,
                "note": "no_valid_road_geometry",
            })
            continue

        # Convert the two points back to WGS84 for bearing calculation
        pt_wgs = gpd.GeoSeries([pt], crs=epsg_metric).to_crs(4326).iloc[0]
        np_wgs = gpd.GeoSeries([best_np], crs=epsg_metric).to_crs(4326).iloc[0]
        brg = bearing_deg(pt_wgs, np_wgs)                        # Bearing from property to nearest road point
        facing = bucket_compass8(brg)                            # Compass bucket label

        # Append result row for this address
        results.append({
            "address": row.get("address"),
            "orientation": facing,
            "bearing_deg": round(float(brg), 1),
            "distance_to_road_m": round(float(best_d), 2),
            "note": "ok",
        })

    # Convert list of dicts to DataFrame and save to CSV
    out = pd.DataFrame(results)
    out.to_csv(results_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out), results_csv)
    # Log a small preview
    logger.info("Results head(5):\n%s", out.head(5).to_string(index=False))


def validate_results(gnaf_src: gpd.GeoDataFrame,
                     roads_src: gpd.GeoDataFrame,
                     results_csv: Path,
                     qc_csv: Path,
                     logger: logging.Logger,
                     epsg_metric: int = 7856,
                     radius_list=(20, 40, 80, 160, 320, 500)):
    """Recompute orientation for result addresses and compare to CSV; write QC CSV and log summary."""
    logger.info("Validation: loading results CSV: %s", results_csv)
    res = pd.read_csv(results_csv)                               # Load produced CSV
    res["address_norm"] = res["address"].astype(str).str.strip() # Normalize address for join
    res["row_id"] = np.arange(len(res))                          # Stable ID per row

    # Keep a minimal set of GNAF columns for the join and geometry rebuild
    keep_cols = ["address", "latitude", "longitude", "street_name", "street_type", "street_suffix"]
    src = gnaf_src.copy()                                        # Work on a copy
    for c in keep_cols:
        if c not in src.columns:                                 # Ensure columns exist
            src[c] = None
    src = src[keep_cols].copy()                                  # Trim to the minimal set
    src["address_norm"] = src["address"].astype(str).str.strip() # Normalized address for join

    # Join computed results back to source GNAF to recover coordinates
    merged = res.merge(src, on="address_norm", how="left", suffixes=("", "_gnaf"), indicator=True)

    # Count how many GNAF rows matched each result row to spot ambiguous matches
    match_counts = merged.groupby("row_id")["_merge"].count().rename("n_matches")
    merged = merged.merge(match_counts, on="row_id", how="left")

    # Keep the first match for each result row (many-to-one simplification)
    merged = (merged.sort_values("row_id")
                    .drop_duplicates(subset=["row_id"], keep="first")
                    .reset_index(drop=True))

    # Wrap into a GeoDataFrame for recomputation
    props = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["longitude"], merged["latitude"]),
        crs=4326,
    )

    # Clip to roads extent, just like the compute step
    minx, miny, maxx, maxy = roads_src.total_bounds
    props = props.cx[minx:maxx, miny:maxy].copy()
    logger.info("Validating %d rows (of %d). Unmatched/out-of-bounds kept as missing where applicable.",
                len(props), len(res))

    # Project both layers to metric CRS and make a spatial index
    props_m = props.to_crs(epsg_metric)
    roads_m = roads_src.to_crs(epsg_metric)
    sidx = roads_m.sindex

    # Prepare arrays to hold recomputed values
    re_bearing = np.full(len(props_m), np.nan, dtype=float)
    re_dist    = np.full(len(props_m), np.nan, dtype=float)
    re_orient  = np.array([None] * len(props_m), dtype=object)
    re_note    = np.array([""] * len(props_m), dtype=object)

    # Recompute nearest-road orientation for each property
    for i, row in props_m.iterrows():
        pt = row.geometry
        idxs = None
        for r in radius_list:
            idxs = list(sidx.query(pt.buffer(r)))
            if idxs:
                break
        if not idxs:
            re_note[i] = "no_road_within_max_radius"
            continue

        best_d = float("inf")
        best_np = None
        for ridx in idxs:
            line = roads_m.geometry.iloc[ridx]
            np_on = nearest_point_on_line(pt, line)
            if np_on is None:
                continue
            d = pt.distance(np_on)
            if d < best_d:
                best_d, best_np = d, np_on

        if best_np is None:
            re_note[i] = "no_valid_road_geometry"
            continue

        pt_wgs = gpd.GeoSeries([pt], crs=epsg_metric).to_crs(4326).iloc[0]
        np_wgs = gpd.GeoSeries([best_np], crs=epsg_metric).to_crs(4326).iloc[0]
        brg = bearing_deg(pt_wgs, np_wgs)
        re_bearing[i] = brg
        re_dist[i]    = best_d
        re_orient[i]  = bucket_compass8(brg)
        re_note[i]    = "ok"

    # Attach recomputed columns to the frame
    props_m["re_bearing_deg"]      = np.round(re_bearing, 1)
    props_m["re_distance_to_road"] = np.round(re_dist, 2)
    props_m["re_orientation"]      = re_orient
    props_m["re_note"]             = re_note

    # Rename original CSV columns to avoid confusion
    props_m = props_m.rename(columns={
        "orientation": "csv_orientation",
        "bearing_deg": "csv_bearing_deg",
        "distance_to_road_m": "csv_distance_to_road_m",
    })

    # Boolean exact orientation match
    props_m["orientation_match"] = (props_m["csv_orientation"].astype(str)
                                    == props_m["re_orientation"].astype(str))
    # Numeric diffs for diagnostics (safe with NaN)
    props_m["bearing_diff_deg"]  = np.abs(props_m["csv_bearing_deg"] - props_m["re_bearing_deg"])
    props_m["distance_diff_m"]   = np.abs(props_m["csv_distance_to_road_m"] - props_m["re_distance_to_road"])

    # Flags from the join
    props_m["ambiguous_address"] = props_m["n_matches"].fillna(0).astype(int).gt(1)
    props_m["unmatched_address"] = props_m["_merge"].ne("both")

    # Row-wise QC verdict function
    def qc_verdict(row):
        if row["unmatched_address"]:
            return "unmatched_address"
        if row["re_note"] == "no_road_within_max_radius":
            return "no_road_nearby"
        if not row["orientation_match"]:
            return "orientation_mismatch"
        return "ok"

    # Apply QC verdicts
    props_m["qc_verdict"] = props_m.apply(qc_verdict, axis=1)

    # Choose output columns for the QC CSV
    cols_out = [
        "address",
        "csv_orientation", "csv_bearing_deg", "csv_distance_to_road_m",
        "re_orientation", "re_bearing_deg", "re_distance_to_road",
        "orientation_match", "bearing_diff_deg", "distance_diff_m",
        "ambiguous_address", "unmatched_address", "qc_verdict"
    ]
    props_m[cols_out].to_csv(qc_csv, index=False)               # Save QC CSV
    logger.info("Wrote validation CSV: %s", qc_csv)

    # Log overall accuracy and QC counts
    total = len(props_m)
    acc = int(props_m["orientation_match"].sum())
    logger.info("Overall orientation match: %d/%d = %.1f%%", acc, total, (acc/total)*100.0)

    counts = props_m["qc_verdict"].value_counts(dropna=False)
    logger.info("QC verdict counts:\n%s", counts.to_string())

    # Log a small sample of mismatches for inspection
    mismatches = props_m.loc[props_m["qc_verdict"] == "orientation_mismatch", cols_out].head(10)
    if len(mismatches):
        logger.info("Examples of mismatches (top 10):\n%s", mismatches.to_string(index=False))
    else:
        logger.info("No orientation mismatches found in sample view.")

    return props_m  # Return DataFrame in case caller wants to inspect further


# ---------------------------
# CLI entry point
# ---------------------------

def main():
    """Parse CLI, run pipeline, log everything to a text file."""
    parser = argparse.ArgumentParser(description="Compute and validate nearest-road orientations for GNAF addresses.")
    parser.add_argument("--gnaf",    type=Path, required=True, help="Path to GNAF Parquet (e.g., gnaf_prop.parquet)")
    parser.add_argument("--roads",   type=Path, required=True, help="Path to Roads GeoPackage (e.g., roads.gpkg)")
    parser.add_argument("--results", type=Path, default=Path("gnaf_orientation.csv"),
                        help="Output CSV for orientations (default: gnaf_orientation.csv)")
    parser.add_argument("--qc",      type=Path, default=Path("gnaf_orientation_validated.csv"),
                        help="Output CSV for validation (default: gnaf_orientation_validated.csv)")
    parser.add_argument("--log",     type=Path, default=Path("run_log.txt"),
                        help="Path to log text file (default: run_log.txt)")
    parser.add_argument("--epsg_metric", type=int, default=7856,
                        help="Metric EPSG for distances (default: 7856 = GDA2020/MGA Zone 56)")
    parser.add_argument("--radii", nargs="+", type=int, default=[20, 40, 80, 160, 320, 500],
                        help="Search radii (meters) for candidate roads (default: 20 40 80 160 320 500)")

    args = parser.parse_args()                                    # Parse command-line arguments
    logger = setup_logger(args.log)                               # Set up logging to text file (and console)
    logger.info("=== GNAF Orientation Pipeline started ===")      # Start banner
    logger.info("Arguments: %s", vars(args))                      # Log chosen arguments

    # Load sources
    gnaf = load_gnaf(args.gnaf, logger)                           # Load GNAF as GeoDataFrame (EPSG:4326)
    roads = load_roads(args.roads, logger)                        # Load roads as GeoDataFrame (EPSG:4326)

    # Clip GNAF to roads extent (performance & sanity)
    gnaf_near = clip_gnaf_to_roads_bbox(gnaf, roads, logger)      # Keep only points within roads bbox

    # Compute primary results CSV
    compute_orientation(
        gnaf=gnaf_near,
        roads=roads,
        results_csv=args.results,
        logger=logger,
        epsg_metric=args.epsg_metric,
        radius_list=args.radii
    )

    # Validation pass + QC CSV
    validate_results(
        gnaf_src=gnaf,
        roads_src=roads,
        results_csv=args.results,
        qc_csv=args.qc,
        logger=logger,
        epsg_metric=args.epsg_metric,
        radius_list=args.radii
    )

    logger.info("=== Pipeline finished successfully ===")         # End banner




if __name__ == "__main__":
    main()                                                        # Invoke CLI entry point

