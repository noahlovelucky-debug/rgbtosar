from pathlib import Path
import tempfile
from PIL import Image
from rgb2sar.data import DirectionDataset, angular_distance, parse_sar
from rgb2sar.models import Generator, Discriminator
assert angular_distance(355, 0) == 5
assert parse_sar(Path("X_HH_30_355_123.tif"))["azimuth"] == 355
with tempfile.TemporaryDirectory() as td:
    root = Path(td); (root/"rgb"/"car").mkdir(parents=True); (root/"sar"/"car").mkdir(parents=True)
    Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(root/"rgb"/"car"/"1.png")
    Image.new("L", (128, 128), 100).save(root/"sar"/"car"/"X_HH_30_355_123.tif")
    ds = DirectionDataset(root/"rgb", root/"sar", 1, image_size=64, angle_tolerance=10); item = ds[0]
    assert item["rgb"].shape == (3,64,64) and item["sar"].shape == (1,64,64)
    generator = Generator(3,1,16,1); discriminator = Discriminator(1,16)
    y = generator(item["rgb"].unsqueeze(0)); assert y.shape == (1,1,64,64)
    score = discriminator(y); assert score.ndim == 4
    (score.mean() + y.abs().mean()).backward()
print("smoke test passed")
