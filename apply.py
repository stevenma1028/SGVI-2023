import glob
import os
import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from scipy.ndimage import zoom
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
from model.Revit import ReViT
import warnings
from rasterio.io import MemoryFile
from rasterio.errors import NodataShadowWarning

warnings.filterwarnings("ignore", category=NodataShadowWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_zoom_data_from_memory(src, shape):
    try:
        out_image, _ = mask(src, shapes=shape, crop=True)
        meta = src.meta.copy()
        if src.nodata is not None:
            valid_data_mask = np.all(out_image != src.nodata, axis=0)
        else:
            valid_data_mask = np.all(out_image != -3.4028235e+38, axis=0)

        non_empty_rows = np.where(valid_data_mask.any(axis=1))[0]
        non_empty_cols = np.where(valid_data_mask.any(axis=0))[0]

        if non_empty_rows.size > 0 and non_empty_cols.size > 0:
            min_row, max_row = non_empty_rows[0], non_empty_rows[-1]
            min_col, max_col = non_empty_cols[0], non_empty_cols[-1]
            out_image = out_image[:, min_row:max_row + 1, min_col:max_col + 1]

        meta.update({'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': src.transform})
        return out_image, meta
    except ValueError as e:
        if 'do not overlap' in str(e):
            return None, None
        else:
            raise

def resample_to_array(out_image, meta, new_pixel_size, order):
    band = meta['count']
    cols, rows = meta["width"], meta["height"]
    transform = meta['transform']
    scale_x = transform.a / new_pixel_size
    scale_y = abs(transform.e) / new_pixel_size
    new_cols = int(round(cols * scale_x))
    new_rows = int(round(rows * scale_y))
    if new_cols < 224 or new_rows < 224:
        return None

    resampled_data = np.empty((band, new_rows, new_cols), dtype=np.float16)
    for i in range(band):
        resampled_data[i] = zoom(out_image[i], (scale_y, scale_x), order=order)
    return resampled_data

def roadprcess(H, W):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    y = y.astype(np.float32) - (H - 1) / 2
    x = x.astype(np.float32) - (W - 1) / 2
    distance = np.sqrt(x**2 + y**2)
    return (1 - distance / distance.max()).reshape((1, H, W))

def crop(array):
    mcol = (array.shape[2] - 224) // 2
    mrow = (array.shape[1] - 224) // 2
    return array[:, mrow:mrow + 224, mcol:mcol + 224]

def load_raster_to_memory(Sentiniel_file, planet_file, osm_file):
    raster_data = {
        'S2': rasterio.open(Sentiniel_file),
        'S1': rasterio.open(Sentiniel_file.replace("S2", "S1")),
        'planet': rasterio.open(planet_file),
        'osm': rasterio.open(osm_file)
    }
    return raster_data

def process_one_sample_from_memory(raster_data, shape, b1max, b1min, loaded_max, loaded_min, model, roadweight):
    arrayS2, metaS2 = get_zoom_data_from_memory(raster_data['S2'], shape)
    if arrayS2 is None: return None
    arrayS2 = crop(resample_to_array(arrayS2, metaS2, 1, 0))
    if arrayS2 is None: return None

    arrayS1, metaS1 = get_zoom_data_from_memory(raster_data['S1'], shape)
    if arrayS1 is None: return None
    arrayS1 = crop(resample_to_array(arrayS1, metaS1, 1, 0))
    if arrayS1 is None: return None

    arrayosm, metaOSM = get_zoom_data_from_memory(raster_data['osm'], shape)
    if arrayosm is None: return None
    arrayosm = crop(resample_to_array(arrayosm, metaOSM, 1, 2))
    if arrayosm is None: return None

    arrayg, metaG = get_zoom_data_from_memory(raster_data['planet'], shape)
    arrayg = crop(resample_to_array(arrayg, metaG, 1, 0))[0:3].astype(np.uint8)

    depth = model.infer_image(arrayg.transpose(1, 2, 0)).astype(np.uint8).reshape(1, 224, 224)

    if arrayosm.shape != roadweight.shape:
        return None

    arrayosm = (1 - arrayosm / arrayosm.max() + roadweight) / 2

    arrays = np.concatenate((arrayS2, arrayS1, arrayg, arrayosm, depth), axis=0)
    mb = (arrays[0:1] - b1min) / (b1max - b1min)
    for i in range(1, 17):
        mb = np.concatenate((mb, (arrays[i:i+1] - loaded_min[i]) / (loaded_max[i] - loaded_min[i])), axis=0)
    mb = np.concatenate((mb, arrays[17:19]), axis=0)

    return torch.tensor(mb).float().unsqueeze(0)

if __name__ == '__main__':
    city = 'Shenzhen'
    year= '2023'
    shapefile = f"./apply/shp/{city}_buffered.shp"
    vector_data = gpd.read_file(shapefile)
    roadweight = roadprcess(224, 224)

    file_path = 'normalization_stats.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
        loaded_max = list(map(float, lines[lines.index("Overall Max:\n") + 1].strip().split(',')))
        loaded_min = list(map(float, lines[lines.index("Overall Min:\n") + 1].strip().split(',')))
    b1max, b1min = loaded_max[0], loaded_min[0]

    lonmax, lonmin, latmax, latmin = 130.5, 102, 46.5, 22

    class Config:
        class DATA:
            crop_size = 224
        class MODEL:
            num_classes = 1
            dropout_rate = 0.1
            head_act = None
        class ReViT:
            mode = "conv"
            pool_first = False
            patch_kernel = [16, 16]
            patch_stride = [16, 16]
            patch_padding = [0, 0]
            embed_dim = 768
            num_heads = 12
            mlp_ratio = 4
            qkv_bias = True
            drop_path = 0.2
            depth = 12
            dim_mul = []
            head_mul = []
            pool_qkv_kernel = []
            pool_kv_stride_adaptive = []
            pool_q_stride = []
            zero_decay_pos = False
            use_abs_pos = True
            use_rel_pos = False
            rel_pos_zero_init = False
            residual_pooling = False
            dim_mul_in_att = False
            alpha = True
            visualize = True
            cls_embed_on = False

    cfg = Config()
    model = ReViT(cfg).to(DEVICE)
    model.load_state_dict(torch.load('./checkpoint/Revit_checkpoint.bin', weights_only=True))
    model.eval()

    encoder = 'vitl'
    model1 = DepthAnythingV2(**{
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }[encoder]).to(DEVICE).eval()

    for time in range(1, 5):
        outputname = f"{city}apply{time}_{year}.csv"
        Sentiniel_file = f"./apply/RS_data/{city}/{year}/Q{time}/S2/{city}.tif"
        planet_file = f"./apply/RS_data/{city}/{year}/Q{time}/planet/{city}.tif"
        osm_file = f"./apply/osm/{city}osm.tif"

        raster_data = load_raster_to_memory(Sentiniel_file, planet_file, osm_file)

        save_every = 10000
        part = 0
        all_veg = []
        batch_id_all = []

        batch_x_accumulated, batch_position_accumulated, batch_count = [], [], 0

        for idx, row in tqdm(vector_data.iterrows(), total=len(vector_data), desc=f"Processing Q{time}", unit="sample"):
            shape = [row["geometry"]]
            batch_x = process_one_sample_from_memory(raster_data, shape, b1max, b1min, loaded_max, loaded_min, model1, roadweight)
            if batch_x is None:
                continue

            lon = (row["lon"] - lonmin) / (lonmax - lonmin)
            lat = (row["lat"] - latmin) / (latmax - latmin)
            batch_position = torch.tensor([[lon, lat]], dtype=torch.float32)

            batch_x_accumulated.append(batch_x)
            batch_position_accumulated.append(batch_position)
            batch_id_all.append(row["Id"])
            batch_count += 1

            if batch_count == 256:
                batch_x_tensor = torch.cat(batch_x_accumulated, dim=0).to(DEVICE)
                batch_position_tensor = torch.cat(batch_position_accumulated, dim=0).to(DEVICE)
                with torch.no_grad():
                    preds = model(batch_x_tensor, batch_position_tensor)
                all_veg.extend(preds.cpu().numpy())
                batch_x_accumulated, batch_position_accumulated, batch_count = [], [], 0

            if len(all_veg) >= save_every:
                result = pd.DataFrame(np.hstack((np.array(batch_id_all).reshape(-1, 1), np.array(all_veg).reshape(-1, 1))),
                                      columns=["Id", "Prediction"])
                result.to_csv(f"{city}_Q{time}_part{part}.csv", index=False)
                part += 1
                all_veg, batch_id_all = [], []

        if batch_count > 0:
            batch_x_tensor = torch.cat(batch_x_accumulated, dim=0).to(DEVICE)
            batch_position_tensor = torch.cat(batch_position_accumulated, dim=0).to(DEVICE)
            with torch.no_grad():
                preds = model(batch_x_tensor, batch_position_tensor)
            all_veg.extend(preds.cpu().numpy())

        if len(all_veg) > 0:
            result = pd.DataFrame(np.hstack((np.array(batch_id_all).reshape(-1, 1), np.array(all_veg).reshape(-1, 1))),
                                  columns=["Id", "Prediction"])
            result.to_csv(f"{city}_Q{time}_part{part}.csv", index=False)

        all_parts = sorted(glob.glob(f"{city}_Q{time}_part*.csv"))
        df_all = pd.concat([pd.read_csv(p) for p in all_parts], ignore_index=True)
        df_all.to_csv(outputname, index=False)

        for p in all_parts:
            os.remove(p)

        for key in raster_data:
            raster_data[key].close()
