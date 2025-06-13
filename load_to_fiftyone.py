"""Load generated samples into FiftyOne for visualization."""

import json
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.core.metadata as fom

    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("WARNING: FiftyOne not fully available. Creating data for manual viewing.")


def create_fiftyone_dataset(metadata_file="transformation_samples/metadata.json"):
    """Create a FiftyOne dataset from the generated samples."""

    if not FIFTYONE_AVAILABLE:
        print("\nFiftyOne is not available on this system.")
        print("However, your transformation samples have been generated successfully!")
        print("\nYou can view the results in:")
        print("  - transformation_samples/transformation_grid.jpg (overview)")
        print("  - transformation_samples/before_after_comparisons.jpg (comparisons)")
        print("  - transformation_samples/ (individual samples)")
        return

    # Load metadata
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_file}")
        print("Please run visualize_simple.py first.")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Create dataset
    dataset_name = "fastscp_transformations"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    print(f"\nCreating FiftyOne dataset '{dataset_name}'...")

    # Group samples by original image
    originals = {}
    for sample_data in metadata["samples"]:
        orig_file = sample_data["original"]
        if orig_file not in originals:
            originals[orig_file] = {
                "original": orig_file,
                "transformations": [],
                "config": sample_data["config"],
                "background": sample_data["background"],
                "object_counts": sample_data["object_counts"],
            }
        originals[orig_file]["transformations"].append(sample_data)

    # Add samples to dataset
    for orig_file, data in originals.items():
        # Add original image
        orig_path = metadata_path.parent / orig_file
        sample = fo.Sample(filepath=str(orig_path))

        # Add metadata
        sample["is_original"] = True
        sample["background_type"] = data["background"]
        sample["transform_config"] = data["config"]

        # Add tags
        sample.tags = ["original", data["background"]]

        dataset.add_sample(sample)

        # Add transformed versions
        for trans_data in data["transformations"]:
            trans_path = metadata_path.parent / trans_data["transformed"]
            trans_sample = fo.Sample(filepath=str(trans_path))

            # Add metadata
            trans_sample["is_original"] = False
            trans_sample["background_type"] = trans_data["background"]
            trans_sample["transform_config"] = trans_data["config"]
            trans_sample["variation"] = trans_data["variation"]
            trans_sample["transform_time_ms"] = trans_data["transform_time_ms"]
            trans_sample["blend_mode"] = trans_data["blend_mode"]
            trans_sample["scale_range"] = str(trans_data["scale_range"])

            # Add object counts as fields
            for obj_type, count in trans_data["object_counts"].items():
                trans_sample[f"count_{obj_type}"] = count

            # Add tags
            trans_sample.tags = [
                "transformed",
                trans_data["background"],
                trans_data["blend_mode"],
            ]

            dataset.add_sample(trans_sample)

    # Compute metadata
    dataset.compute_metadata()

    print(f"\nDataset created with {len(dataset)} samples")
    print(f"  - Originals: {len(dataset.match_tags('original'))}")
    print(f"  - Transformed: {len(dataset.match_tags('transformed'))}")

    # Launch app
    print("\nLaunching FiftyOne App...")
    session = fo.launch_app(dataset)

    print("\nFiftyOne app is running!")
    print("You can:")
    print("  - Filter by tags (original/transformed)")
    print("  - Sort by transform_time_ms")
    print("  - Group by background_type or transform_config")
    print("\nPress Ctrl+C to exit...")

    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    """Main function."""
    print("FastSCP FiftyOne Loader")
    print("=" * 50)

    # Check if samples exist
    metadata_file = "transformation_samples/metadata.json"
    if not Path(metadata_file).exists():
        print("\nNo transformation samples found.")
        print("Please run 'python visualize_simple.py' first to generate samples.")
        return

    create_fiftyone_dataset(metadata_file)


if __name__ == "__main__":
    main()
