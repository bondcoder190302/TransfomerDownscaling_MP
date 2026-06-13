import os
import numpy as np

# ── LOCAL OUTPUT DIRECTORY ───────────────────────────────────────────────────
# Saving them to the SRTM folder alongside your other high-res structural assets
OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("🏗️ GENERATING FULL-INDIA GEOSPATIAL COORDINATE MESHES (768x768)")
print("="*70)

# ── 1. EXACT BOUNDING BOX ANCHORS (FROM ELEVATION REFERENCE) ────────────────
XMIN = 62.2998717527286487  # Longitude West
XMAX = 100.6998717527286544 # Longitude East
YMIN = 1.3000558348497222   # Latitude South
YMAX = 39.7000558348497279  # Latitude North

GRID_SIZE = 768  # 0.05 degree high-resolution target dimension

# ── 2. CREATE 1D AXIS VECTORS WITH CORRECT ORIENTATIONS ─────────────────────
# Longitude increases from West (Left) to East (Right)
lon_vector = np.linspace(XMIN, XMAX, GRID_SIZE, dtype=np.float32)

# Latitude MUST decrease from North (Top) to South (Bottom) to mirror 
# standard raster image indexing formats where row 0 represents the top/North edge.
lat_vector = np.linspace(YMAX, YMIN, GRID_SIZE, dtype=np.float32)

# ── 3. BUILD THE 2D GEOSPATIAL MESHGRIDS ─────────────────────────────────────
# np.meshgrid converts the 1D arrays into 2D matrices matching your imagery dimensions
lon_mesh, lat_mesh = np.meshgrid(lon_vector, lat_vector)

print(f"Generated Matrices:")
print(f"  -> LON_fix Shape: {lon_mesh.shape} | Range: [{lon_mesh.min():.7f}, {lon_mesh.max():.7f}]")
print(f"  -> LAT_fix Shape: {lat_mesh.shape} | Range: [{lat_mesh.min():.7f}, {lat_mesh.max():.7f}]")

# Spot-check layout orientation to confirm structural parity with older assets
print(f"\nOrientation Layout Check:")
print(f"  Top-Left Corner (Row 0, Col 0)   : Lon={lon_mesh[0, 0]:.7f}, Lat={lat_mesh[0, 0]:.7f} (North-West)")
print(f"  Bottom-Right Corner (Last Cells) : Lon={lon_mesh[-1, -1]:.7f}, Lat={lat_mesh[-1, -1]:.7f} (South-East)")

# ── 4. SAVE ARRAYS AS STATIC SYSTEM ASSETS ──────────────────────────────────
lon_out_path = os.path.join(OUTPUT_DIR, 'LON_fix.npy')
lat_out_path = os.path.join(OUTPUT_DIR, 'LAT_fix.npy')

np.save(lon_out_path, lon_mesh)
np.save(lat_out_path, lat_mesh)

print(f"\n✅ Success! New spatial coordinate files written to drive:")
print(f"   Saved: {lon_out_path}")
print(f"   Saved: {lat_out_path}")
print("="*70)