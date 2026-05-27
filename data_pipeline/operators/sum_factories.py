"""Summarize factory output directory structure.

Usage:
    python summarize_factories.py /path/to/output_factories
"""

import argparse
import os
from pathlib import Path


def summarize(root: str):
    root = Path(root)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return

    categories = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    total_variants = 0
    total_seeds = 0
    total_images = 0

    print(f"{'Category':<25} {'Variants':>10} {'Seeds(total)':>14} {'Seeds/Variant':>15} {'Images(total)':>15} {'Imgs/Seed':>11}")
    print("-" * 94)

    for cat in categories:
        variants = sorted([v for v in cat.iterdir() if v.is_dir()], key=lambda v: v.name)
        n_variants = len(variants)
        seed_counts = []
        img_counts = []
        for v in variants:
            seeds = [s for s in v.iterdir() if s.is_dir()]
            n_seeds = len(seeds)
            seed_counts.append(n_seeds)
            n_imgs = 0
            for s in seeds:
                n_imgs += sum(1 for f in s.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr'))
            # Also count images directly in the variant dir (not in seed subdirs)
            n_imgs += sum(1 for f in v.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr'))
            img_counts.append(n_imgs)

        cat_total_seeds = sum(seed_counts)
        cat_total_imgs = sum(img_counts)
        if seed_counts:
            min_s, max_s = min(seed_counts), max(seed_counts)
            seed_range = f"{min_s}-{max_s}" if min_s != max_s else str(min_s)
        else:
            seed_range = "0"

        if cat_total_seeds > 0:
            avg_imgs = f"{cat_total_imgs / cat_total_seeds:.1f}"
        else:
            avg_imgs = "0"

        print(f"{cat.name:<25} {n_variants:>10} {cat_total_seeds:>14} {seed_range:>15} {cat_total_imgs:>15} {avg_imgs:>11}")
        total_variants += n_variants
        total_seeds += cat_total_seeds
        total_images += cat_total_imgs

    print("-" * 94)
    avg_total = f"{total_images / total_seeds:.1f}" if total_seeds > 0 else "0"
    print(f"{'TOTAL':<25} {total_variants:>10} {total_seeds:>14} {'':>15} {total_images:>15} {avg_total:>11}")

    # Detail: list variants with < 60 seeds
    print()
    print("Variants with < 60 seeds:")
    found = False
    for cat in categories:
        for v in sorted(cat.iterdir()):
            if v.is_dir():
                n = sum(1 for s in v.iterdir() if s.is_dir())
                if n < 60:
                    print(f"  {cat.name}/{v.name}: {n} seeds")
                    found = True
    if not found:
        print("  (none)")

    # Detail: list seeds with incomplete images (< mode frame count)
    print()
    print("Seeds with fewer images than siblings:")
    found = False
    for cat in categories:
        for v in sorted(cat.iterdir()):
            if not v.is_dir():
                continue
            seeds = sorted([s for s in v.iterdir() if s.is_dir()], key=lambda s: s.name)
            if not seeds:
                continue
            per_seed = []
            for s in seeds:
                n = sum(1 for f in s.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr'))
                per_seed.append((s.name, n))
            if not per_seed:
                continue
            counts = [c for _, c in per_seed]
            # Find the most common count (mode)
            from collections import Counter
            mode_count = Counter(counts).most_common(1)[0][0]
            for name, c in per_seed:
                if c < mode_count:
                    print(f"  {cat.name}/{v.name}/{name}: {c} images (expected {mode_count})")
                    found = True
    if not found:
        print("  (none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize factory output directory structure")
    parser.add_argument("root", help="Root directory (e.g. output_factories)")
    args = parser.parse_args()
    summarize(args.root)
