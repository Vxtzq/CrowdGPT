#!/usr/bin/env python3
"""
Patch CrowdGPT client.py:
  1. DirectML crash: never send a bf16 tensor to the device (convert to fp32 on the CPU first)
  2. DirectML: explicit fp32 / no autocast
  3. Icons: prefer pre-rendered docs/logo-app.ico (Windows) and docs/logo-app.png (Linux)
     instead of rendering the SVG at runtime
  4. Optional: --make-icons builds docs/logo-app.ico + docs/logo-app.png from docs/logo-app.svg

Usage (run with any Python 3):
    python patch_client.py path\\to\\client.py
    python patch_client.py path\\to\\client.py --make-icons

Safe to run twice. A backup is written to client.py.prepatch.
"""
import sys
import shutil
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
target = Path(args[0] if args else "client.py").resolve()
if not target.is_file():
    sys.exit(f"[error] not found: {target}")

src = target.read_text(encoding="utf-8")

PATCHES = [
    (
        "DirectML: convert bf16 -> fp32 on the CPU before moving to the device",
        "                p.data.copy_(ft[o:o + s].view(p.shape).to(p.device))\n",
        "                # bf16 -> fp32 on the CPU, one tensor at a time. A bf16 tensor must\n"
        "                # never reach DirectML (native abort, not a catchable exception).\n"
        "                chunk = ft[o:o + s].view(p.shape).to(torch.float32)\n"
        "                p.data.copy_(chunk.to(p.device))\n",
    ),
    (
        "DirectML: explicit fp32, no autocast",
        "def _pick_autocast(backend):\n",
        "def _pick_autocast(backend):\n"
        "    if backend == \"DIRECTML\":\n"
        "        return False, None          # DirectML: fp32 only\n",
    ),
    (
        "Icons: look for docs/logo-app.ico first",
        '    ico_candidates = [base / "icon.ico", base / "docs" / "icon.ico"]\n',
        '    ico_candidates = [base / "docs" / "logo-app.ico", base / "icon.ico", base / "docs" / "icon.ico"]\n',
    ),
    (
        "Icons: use .ico only on Windows (Linux/GTK uses the PNG)",
        "    for ico in ico_candidates:\n        if ico.is_file():\n",
        "    for ico in (ico_candidates if sys.platform == \"win32\" else []):\n        if ico.is_file():\n",
    ),
    (
        "Icons: prefer a pre-rendered PNG over rendering/using the raw SVG",
        "    for svg in svg_candidates:\n",
        "    if any(p.is_file() for p in png_candidates):\n"
        "        svg_candidates = []\n"
        "    for svg in svg_candidates:\n",
    ),
]

changed = False
for name, old, new in PATCHES:
    if new in src:
        print(f"[skip]  already applied: {name}")
        continue
    n = src.count(old)
    if n != 1:
        sys.exit(f"[error] '{name}': expected 1 match, found {n}. "
                 f"client.py differs from the version this patch was written for.")
    src = src.replace(old, new)
    changed = True
    print(f"[ok]    {name}")

if changed:
    shutil.copy2(target, target.with_name(target.name + ".prepatch"))
    target.write_text(src, encoding="utf-8")
    print(f"[done]  patched {target} (backup: {target.name}.prepatch)")
else:
    print("[done]  nothing to change")


def make_icons(base: Path):
    docs = base / "docs"
    svg, png, ico = docs / "logo-app.svg", docs / "logo-app.png", docs / "logo-app.ico"
    if not svg.is_file():
        print(f"[icons] {svg} not found; nothing to build")
        return
    try:
        from PIL import Image
    except ImportError:
        print("[icons] need Pillow:  pip install pillow")
        return

    tmp = docs / "_icon_tmp.png"
    rendered = False
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(tmp), output_width=512)
        rendered = True
        print("[icons] rendered with cairosvg (transparent)")
    except Exception as e:
        print(f"[icons] cairosvg unavailable ({type(e).__name__})")
    if not rendered:
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            d = svg2rlg(str(svg))
            s = 512 / max(float(d.width), float(d.height))
            d.scale(s, s)
            d.width, d.height = d.width * s, d.height * s
            renderPM.drawToFile(d, str(tmp), fmt="PNG")
            rendered = True
            print("[icons] rendered with svglib (background may be opaque)")
        except Exception as e:
            print(f"[icons] svglib unavailable ({type(e).__name__}: {e})")
    if not rendered:
        print("[icons] could not render the SVG. Export it to a 512x512 PNG with Inkscape "
              "or any converter, save it as docs/logo-app.png, and run with --make-icons again.")
        return

    img = Image.open(tmp).convert("RGBA")
    img.thumbnail((512, 512))
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))   # pad to square, no stretching
    canvas.paste(img, ((512 - img.width) // 2, (512 - img.height) // 2), img)
    canvas.save(png)
    canvas.save(ico, sizes=[(s, s) for s in (256, 128, 64, 48, 32, 24, 16)])
    try:
        tmp.unlink()
    except Exception:
        pass
    print(f"[icons] wrote {png.name} and {ico.name} in {docs}")
    print("[icons] commit both files to the repo so the installer ships them.")


if "--make-icons" in flags:
    make_icons(target.parent)
