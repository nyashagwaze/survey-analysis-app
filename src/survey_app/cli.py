
#!/usr/bin/env python3
"""
CLI entrypoint for the survey analysis pipeline (pandas mode).
Thin wrapper around pipeline.run_pipeline().
"""
from .pipeline import _main as pipeline_main


def main():
    pipeline_main()


if __name__ == "__main__":
    main()
