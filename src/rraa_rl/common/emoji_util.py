from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from emoji import emojize
from PIL import Image

from rraa_rl.common.path_utils import get_root_dir

try:
    import requests  # optional

    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class TwemojiSVGSource:
    """
    Standalone Twemoji SVG fetcher with caching.

    - Fetches SVG from jsDelivr's @twemoji/svg package.
    - Caches on disk (and in-memory) to avoid re-downloading.
    - Optionally rasterizes SVG -> PNG at a chosen size (useful for matplotlib imshow).

    Notes on filename "slug" generation:
      - Twemoji assets are named by the emoji's Unicode codepoint sequence, lowercased hex,
        joined by '-' (e.g. 😄 -> 1f604.svg, 🇪🇺 -> 1f1ea-1f1fa.svg).
      - This implementation:
          * removes U+FE0E (text presentation)
          * keeps U+FE0F (emoji presentation) if present
        This works for many emojis, but there are edge cases across ZWJ sequences and
        variation selectors where you might want a full emoji data table.
    """

    cache_dir: Path = get_root_dir() / ".emoji_cache/twemoji"
    version: str = "15.0.0"
    use_requests: bool = True
    user_agent: str = "Mozilla/5.0"
    timeout: float = 20.0

    # internal caches
    _mem_svg: Dict[str, bytes] = field(default_factory=dict, init=False)
    _mem_png: Dict[Tuple[str, int, int], bytes] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.use_requests and not _HAS_REQUESTS:
            self.use_requests = False  # silently fall back

    # ----------------------------
    # Public API
    # ----------------------------

    def get_svg(self, emoji: str) -> Optional[BytesIO]:
        """
        Return SVG bytes for the given emoji as a BytesIO stream, or None if not found.
        """
        slug = self.emoji_to_slug(emoji)

        # 1) memory cache
        if slug in self._mem_svg:
            return BytesIO(self._mem_svg[slug])

        # 2) disk cache
        svg_path = self.cache_dir / f"{slug}.svg"
        if svg_path.exists():
            data = svg_path.read_bytes()
            self._mem_svg[slug] = data
            return BytesIO(data)

        # 3) download
        url = self._svg_url(slug)
        try:
            data = self._http_get(url)
        except (HTTPError, URLError, TimeoutError):
            return None
        except Exception:
            return None

        # persist safely
        self._atomic_write(svg_path, data)
        self._mem_svg[slug] = data
        return BytesIO(data)

    def get_png(self, emoji: str, width: int = 512, height: int = 512) -> Optional[BytesIO]:
        """
        Rasterize the emoji SVG to PNG bytes at (width, height) and return a BytesIO stream.

        Requires: pip install cairosvg

        Caches:
          - in memory (keyed by emoji+size)
          - on disk (cache_dir/<slug>_<width>x<height>.png)
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        slug = self.emoji_to_slug(emoji)
        key = (slug, width, height)

        # 1) memory cache
        if key in self._mem_png:
            return BytesIO(self._mem_png[key])

        # 2) disk cache
        png_path = self.cache_dir / f"{slug}_{width}x{height}.png"
        if png_path.exists():
            data = png_path.read_bytes()
            self._mem_png[key] = data
            return BytesIO(data)

        # 3) fetch svg
        svg_bio = self.get_svg(emoji)
        if svg_bio is None:
            return None

        # 4) rasterize
        try:
            import cairosvg  # type: ignore
        except ImportError as e:
            raise RuntimeError("Rasterizing SVG requires cairosvg: pip install cairosvg") from e

        png_bytes = cairosvg.svg2png(
            bytestring=svg_bio.getvalue(),
            output_width=width,
            output_height=height,
        )

        # persist & cache
        self._atomic_write(png_path, png_bytes)
        self._mem_png[key] = png_bytes
        return BytesIO(png_bytes)

    @staticmethod
    def emoji_to_slug(emoji: str) -> str:
        """
        Convert an emoji string to a Twemoji filename slug: lowercased hex codepoints joined by '-'.

        Removes U+FE0E (text presentation). Keeps U+FE0F if present.
        """
        cps = [ord(ch) for ch in emoji if ord(ch) != 0xFE0E]
        return "-".join(f"{cp:x}" for cp in cps)

    # ----------------------------
    # Internals
    # ----------------------------

    def _svg_url(self, slug: str) -> str:
        # Example: https://cdn.jsdelivr.net/npm/@twemoji/svg@15.0.0/1f604.svg
        return f"https://cdn.jsdelivr.net/npm/@twemoji/svg@{self.version}/{slug}.svg"

    def _http_get(self, url: str) -> bytes:
        headers = {"User-Agent": self.user_agent}

        if self.use_requests and _HAS_REQUESTS:
            # requests is available and enabled
            resp = requests.get(url, headers=headers, timeout=self.timeout)  # type: ignore
            resp.raise_for_status()
            return resp.content

        # urllib fallback
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout) as r:
            return r.read()

    @staticmethod
    def _atomic_write(path: Path, data: BytesLike) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(bytes(data))
        tmp.replace(path)


def plot_emoji(
    pos: np.ndarray,
    size_data: float,
    emoji_str: str = ":key:",
    size: int = 512,
    ax: plt.Axes = None,
    extent: str = "lower",
):
    assert ax is not None
    source = TwemojiSVGSource()
    im = np.array(Image.open(source.get_png(emojize(emoji_str), width=size, height=size)).convert("RGBA"))
    # Flip vertically for matplotlib
    im = im[::-1, :]

    half_size = size_data / 2.0
    ax.imshow(
        im, extent=(-half_size + pos[0], half_size + pos[0], -half_size + pos[1], half_size + pos[1]), origin=extent
    )
