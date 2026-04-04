"""
Generate demo sample images for hackathon demonstrations.
Creates placeholder crop leaf images with synthetic patterns.
"""
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


def create_leaf_pattern(
    width: int = 512,
    height: int = 384,
    base_color: tuple = (34, 139, 34),
    disease_color: tuple = (139, 69, 19),
    disease_level: float = 0.3,
) -> Image.Image:
    """Create a synthetic leaf image with disease-like patterns."""
    img = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(img)

    # Add leaf vein pattern
    for i in range(5):
        x = width // 2 + random.randint(-50, 50)
        draw.line([(x, 0), (x + random.randint(-30, 30), height)], 
                  fill=(20, 100, 20), width=2)

    # Add disease spots based on level
    num_spots = int(20 * disease_level)
    for _ in range(num_spots):
        x = random.randint(20, width - 20)
        y = random.randint(20, height - 20)
        r = random.randint(10, 40)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=disease_color)

    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img


def generate_all_samples(output_dir: str = "data/plantvillage_samples") -> None:
    """Generate sample images for all demo scenarios."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {
            "name": "maize_blight",
            "base": (40, 120, 40),
            "disease": (110, 70, 30),
            "level": 0.4,
        },
        {
            "name": "tomato_late_blight",
            "base": (50, 130, 50),
            "disease": (60, 40, 30),
            "level": 0.5,
        },
        {
            "name": "potato_early_blight",
            "base": (45, 125, 45),
            "disease": (80, 50, 25),
            "level": 0.35,
        },
        {
            "name": "rice_bacterial_leaf_blight",
            "base": (60, 140, 60),
            "disease": (150, 140, 90),
            "level": 0.45,
        },
        {
            "name": "wheat_leaf_rust",
            "base": (55, 135, 55),
            "disease": (180, 100, 50),
            "level": 0.3,
        },
        {
            "name": "ragi_blast",
            "base": (50, 125, 50),
            "disease": (70, 50, 35),
            "level": 0.4,
        },
        {
            "name": "sugarcane_red_rot",
            "base": (45, 130, 45),
            "disease": (140, 40, 40),
            "level": 0.5,
        },
        {
            "name": "healthy_leaf",
            "base": (34, 139, 34),
            "disease": (34, 139, 34),
            "level": 0.0,
        },
    ]

    print(f"Generating {len(scenarios)} demo sample images...")

    for scenario in scenarios:
        img = create_leaf_pattern(
            base_color=scenario["base"],
            disease_color=scenario["disease"],
            disease_level=scenario["level"],
        )
        filename = out_path / f"{scenario['name']}.jpg"
        img.save(filename, quality=90)
        print(f"  ✓ Created {filename}")

    print(f"\nDone! {len(scenarios)} samples created in {output_dir}/")


if __name__ == "__main__":
    generate_all_samples()
