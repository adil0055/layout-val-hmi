"""Capturing with this computer's webcam instead of a phone."""

from __future__ import annotations

import urllib.request

import pytest

from layoutval.cli import build_parser
from layoutval.server import CaptureSession, serve


@pytest.mark.parametrize("argv, out", [
    (["go", "--webcam"], "out/webcam"),
    (["go"], "out/captures"),
    (["go", "--webcam", "--out", "elsewhere"], "elsewhere"),
])
def test_a_webcam_keeps_its_lens_and_captures_apart_from_the_phone(tmp_path, monkeypatch,
                                                                   argv, out):
    """A lens solve is in one camera's pixels; the phone's must not be reused."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr("layoutval.cli.cmd_capture_server",
                        lambda args: seen.setdefault("args", args) and 0)
    args = build_parser().parse_args(argv)
    args.func(args)
    assert seen["args"].out == out


def test_the_page_offers_the_webcam_at_a_localhost_address(tmp_path):
    session = CaptureSession(tmp_path, mark_corners=True, calib_mode="auto",
                             size_from_phone=True)
    server = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        port = server.server_address[1]
        page = urllib.request.urlopen(
            f"http://localhost:{port}/?t={server.token}&cam=1", timeout=10).read().decode()
        assert "getUserMedia" in page and 'id="camtoggle"' in page
        assert 'id="camauto"' in page                  # hands-free lens views
    finally:
        server.shutdown()
