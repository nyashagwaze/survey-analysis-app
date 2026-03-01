import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def wait_for_server(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start_streamlit(app_path: Path, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(proc: subprocess.Popen):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def try_click(page, locator):
    try:
        locator.click(timeout=2000)
        return True
    except Exception:
        return False


def navigate_sidebar(page, labels):
    if isinstance(labels, str):
        labels = [labels]
    sidebar = page.get_by_test_id("stSidebar")
    nav = sidebar.locator("[data-testid='stSidebarNav']")
    for label in labels:
        if nav.count() > 0:
            if try_click(page, nav.get_by_role("link", name=label)):
                return
            if try_click(page, nav.get_by_text(label, exact=True)):
                return
        if try_click(page, sidebar.get_by_role("link", name=label)):
            return
        if try_click(page, sidebar.get_by_text(label, exact=True)):
            return
    raise RuntimeError(f"Could not navigate to page '{labels[0]}'")


def maybe_use_sample(page):
    button = page.get_by_role("button", name="Use sample data")
    if button.count() > 0 and button.is_visible():
        button.click()
        page.wait_for_timeout(1500)

def maybe_demo_run(page):
    button = page.get_by_role("button", name="Demo run (sample data)")
    if button.count() > 0 and button.is_visible():
        button.click()
        return True
    return False

def wait_for_run_complete(page, timeout: int = 60000):
    try:
        page.get_by_text("Run complete").wait_for(timeout=timeout)
    except Exception:
        pass


def capture_page(page, output_path: Path):
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(1200)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output_path), full_page=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Streamlit page screenshots.")
    parser.add_argument("--output-dir", default="docs/images", help="Output folder for screenshots.")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server.")
    parser.add_argument("--no-start-server", action="store_true", help="Assume Streamlit is already running.")
    parser.add_argument("--use-sample", action="store_true", help="Click 'Use sample data' if available.")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        print("Then run: python -m playwright install", file=sys.stderr)
        return 1

    app_path = ROOT / "streamlit_app.py"
    if not app_path.exists():
        print(f"Missing {app_path}", file=sys.stderr)
        return 1

    base_url = f"http://localhost:{args.port}"
    proc = None
    if not args.no_start_server:
        proc = start_streamlit(app_path, args.port)

    try:
        if not wait_for_server(base_url, timeout=90):
            raise RuntimeError("Streamlit server did not start in time.")

        output_dir = Path(args.output_dir)
        pages = [
            ("Home", base_url, "home.png"),
            ("Data Settings", base_url, "data-settings.png"),
            ("Results", base_url, "results.png"),
            ("Dashboard", base_url, "dashboard.png"),
            ("Taxonomy Builder", base_url, "taxonomy-builder.png"),
        ]

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            capture_page(page, output_dir / "home.png")

            navigate_sidebar(page, ["Data & Settings", "Data Settings"])
            if args.use_sample:
                did_demo = maybe_demo_run(page)
                if not did_demo:
                    maybe_use_sample(page)
                    run_button = page.get_by_role("button", name="Run pipeline")
                    if run_button.count() > 0 and run_button.is_visible():
                        run_button.click()
                wait_for_run_complete(page)
                page.wait_for_timeout(1500)
            capture_page(page, output_dir / "data-settings.png")

            navigate_sidebar(page, "Results")
            capture_page(page, output_dir / "results.png")

            navigate_sidebar(page, "Dashboard")
            capture_page(page, output_dir / "dashboard.png")

            navigate_sidebar(page, "Taxonomy Builder")
            capture_page(page, output_dir / "taxonomy-builder.png")

            browser.close()

        print(f"Saved screenshots to {output_dir}")
        return 0
    finally:
        stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
