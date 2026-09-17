"""Can the pixel-based duplicate check see a repeated frame through h264?

No. This test exists to keep that answer from being rediscovered the hard way.

The ground truth is known by construction: byte-identical input frames at known
indices, encoded through the same LeRobot call the recorder uses. LeRobot passes
`g=2`, so every other frame is an I-frame, and the two independently quantised
codings of one identical source differ across the whole image.

This is the false-negative direction. The suite already covered the
false-positive direction (`test_a_small_change_in_a_wide_view_is_not_a
_duplicate`), and covering only that direction is how a metric that can never
fire came to be read as "there are no repeats".

Scope, deliberately: this pins that the configured cutoff finds none of the
duplicates. It does *not* try to show that no cutoff would work. On this scene
the duplicates land around 45-55 while the moving frames are near 180, so a
cutoff in between would separate them here. The overlap that makes the check
unfixable was measured on recorded footage, where genuine motion in the quiet
parts of an episode falls to a peak change of 16 while duplicates stay near 50
(session 0810_dup_probe, top camera, 1238 pairs). Asserting that from a
hand-built scene would mean tuning the scene until it agreed -- which is how
the metric being tested here got its credibility in the first place.
"""

import numpy as np
import pytest
from PIL import Image

from rlt_so101_dual.diagnostics import dataset_acceptance as da

av = pytest.importorskip("av")
VideoDecoder = pytest.importorskip("torchcodec.decoders").VideoDecoder
encode_video_frames = pytest.importorskip("lerobot.datasets.video_utils").encode_video_frames

DUP_AT = (10, 20, 30)
N = 40
H, W = 120, 160


def _sequence() -> np.ndarray:
    """A small bright object moving across a mostly static background.

    This is the geometry that matters: the top camera is wide-angle, so genuine
    motion changes a small fraction of the frame.
    """
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 90, (H, W, 3), dtype=np.uint8)
    frames = []
    for i in range(N):
        if i in DUP_AT:
            frames.append(frames[-1])          # the same array, not a copy
            continue
        f = bg.copy()
        x = 8 + 3 * i
        f[50:70, x:x + 20] = 240
        frames.append(f)
    return np.stack(frames)


@pytest.fixture(scope="module")
def decoded(tmp_path_factory):
    seq = _sequence()
    for i in DUP_AT:
        assert np.array_equal(seq[i], seq[i - 1]), f"frame {i} is not a true duplicate"

    work = tmp_path_factory.mktemp("roundtrip")
    imgs = work / "imgs"
    imgs.mkdir()
    for i, f in enumerate(seq):
        Image.fromarray(f).save(imgs / f"frame-{i:06d}.png")

    out = work / "roundtrip.mp4"
    try:
        encode_video_frames(imgs, out, fps=30, vcodec="h264", overwrite=True)
    except Exception as exc:                                  # no usable encoder
        pytest.skip(f"h264 encoding unavailable: {type(exc).__name__}: {exc}")

    got = VideoDecoder(str(out))[0:N].numpy().astype(np.int16)
    d = np.abs(np.diff(got, axis=0))
    peak = d.reshape(len(d), -1).max(axis=1)
    is_dup = np.zeros(len(peak), bool)
    is_dup[np.array(DUP_AT) - 1] = True        # pair (i-1, i) sits at diff index i-1
    return peak, is_dup


def test_the_encoder_makes_identical_inputs_differ(decoded):
    peak, is_dup = decoded
    assert peak[is_dup].min() > da.DUP_MAX_PIXEL_DIFF, (
        "if identical inputs ever decode below the cutoff, the check works after "
        "all and this whole test should be revisited")


def test_the_configured_cutoff_finds_none_of_the_duplicates(decoded):
    peak, is_dup = decoded
    assert (peak[is_dup] <= da.DUP_MAX_PIXEL_DIFF).sum() == 0


def test_a_duplicate_survives_encoding_as_a_large_pixel_change(decoded):
    """Roughly the magnitude measured on real footage (46-58), on a scene built
    independently of it. Loose bound: it is the order that matters, not the
    value, and the value moves with content and encoder build."""
    peak, is_dup = decoded
    assert 10 < peak[is_dup].max() < 120
