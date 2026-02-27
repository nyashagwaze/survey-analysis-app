import argparse
import re
import shutil
from pathlib import Path


def _update_profile_name(profile_path: Path, name: str) -> None:
    if not profile_path.exists():
        return
    text = profile_path.read_text(encoding="utf-8")
    text = re.sub(r'profile:\s*\"[^\"]+\"', f'profile: "{name}"', text)
    profile_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new profile from template.")
    parser.add_argument("--name", required=True, help="New profile name")
    parser.add_argument(
        "--from-profile",
        default="template",
        help="Profile template to copy (default: template)",
    )
    parser.add_argument(
        "--with-taxonomy",
        action="store_true",
        help="Copy taxonomy assets from --from-taxonomy",
    )
    parser.add_argument(
        "--from-taxonomy",
        default="general",
        help="Source taxonomy profile to copy (default: general)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    profiles_dir = root / "config" / "profiles"
    assets_dir = root / "assets" / "taxonomy"

    src_profile = profiles_dir / args.from_profile
    dest_profile = profiles_dir / args.name
    if dest_profile.exists():
        raise FileExistsError(f"Profile already exists: {dest_profile}")
    if not src_profile.exists():
        raise FileNotFoundError(f"Source profile not found: {src_profile}")

    shutil.copytree(src_profile, dest_profile)
    _update_profile_name(dest_profile / "profile.yaml", args.name)

    if args.with_taxonomy:
        src_tax = assets_dir / args.from_taxonomy
        dest_tax = assets_dir / args.name
        if not src_tax.exists():
            raise FileNotFoundError(f"Source taxonomy not found: {src_tax}")
        if dest_tax.exists():
            raise FileExistsError(f"Taxonomy target already exists: {dest_tax}")
        shutil.copytree(src_tax, dest_tax)

    print(f"Created profile: {dest_profile}")
    if args.with_taxonomy:
        print(f"Copied taxonomy assets to: {dest_tax}")
    print("Next steps:")
    print(f"- Edit {dest_profile / 'profile.yaml'} to map your columns")
    print(f"- Update config/pipeline_settings.yaml to set profile: \"{args.name}\"")


if __name__ == "__main__":
    main()
