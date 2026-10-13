"""The CAN tester, worked out from a DBC: cluster, ignition, E2E, rest, cycles.

The names here are the ones real matrices use (VW's ZAS_Kl_15 and _BZ/_CRC,
Ford's Ignition_Status with Run, Tesla's INIT_SNA starts), including the ones
that fooled the first version: "Nachtdesign" ends in "ign", and "NM active
because of KL15" is network management, not the ignition.
"""

from __future__ import annotations

import json
import time

import pytest

can = pytest.importorskip("can")
cantools = pytest.importorskip("cantools")

from layoutval.canbus import ALGORITHMS, CanDriver, CanError, Rig, crc8, parse_bus  # noqa: E402

DBC = """VERSION ""
NS_ :
BS_:
BU_: BCM ESC SRS KOMBI GW

BO_ 160 ESC_01: 8 ESC
 SG_ ESC_01_CRC : 0|8@1+ (1,0) [0|255] "" KOMBI
 SG_ ESC_01_BZ : 8|4@1+ (1,0) [0|15] "" KOMBI
 SG_ ESC_Speed : 16|16@1+ (0.01,0) [0|655.35] "km/h" KOMBI,GW
 SG_ ESC_ABS_Lamp : 32|2@1+ (1,0) [0|3] "" KOMBI

BO_ 800 SRS_01: 4 SRS
 SG_ SRS_AliveCounter : 0|4@1+ (1,0) [0|15] "" KOMBI
 SG_ SRS_Belt_Driver : 4|2@1+ (1,0) [0|3] "" KOMBI

BO_ 912 Klemmen_Status_01: 4 BCM
 SG_ ZAS_Kl_15 : 0|1@1+ (1,0) [0|1] "" KOMBI,ESC
 SG_ DI_Display_Nachtdesign : 1|1@1+ (1,0) [0|1] "" KOMBI

BO_ 1280 NM_BCM: 8 BCM
 SG_ NM_BCM_NM_aktiv_KL15 : 0|1@1+ (1,0) [0|1] "" KOMBI

BO_ 1500 KOMBI_01: 8 KOMBI
 SG_ KOMBI_Odometer : 0|24@1+ (0.1,0) [0|1677721] "km" BCM

BO_ 1793 Diag_Request: 8 GW
 SG_ Diag_Data : 0|64@1+ (1,0) [0|0] "" KOMBI

BA_DEF_ BO_ "GenMsgCycleTime" INT 0 65535;
BA_DEF_ SG_ "GenSigStartValue" INT 0 65535;
BA_DEF_DEF_ "GenMsgCycleTime" 100;
BA_DEF_DEF_ "GenSigStartValue" 0;
BA_ "GenMsgCycleTime" BO_ 160 20;
BA_ "GenMsgCycleTime" BO_ 800 100;
BA_ "GenMsgCycleTime" BO_ 912 100;
BA_ "GenSigStartValue" SG_ 160 ESC_ABS_Lamp 3;
BA_ "GenSigStartValue" SG_ 160 ESC_Speed 65535;
BA_ "GenSigStartValue" SG_ 800 SRS_Belt_Driver 3;
VAL_ 160 ESC_ABS_Lamp 0 "Off" 1 "On" 2 "Flash" 3 "Init_SNA" ;
VAL_ 800 SRS_Belt_Driver 0 "Unbuckled" 1 "Buckled" 2 "Fault" 3 "SNA" ;
"""


@pytest.fixture
def dbc(tmp_path):
    path = tmp_path / "rig.dbc"
    path.write_text(DBC)
    return path


def test_the_rig_is_worked_out_from_the_dbc(dbc):
    rig = Rig.detect(dbc)
    assert rig.cluster_node == "KOMBI"
    # Not its own message, not diagnostics.
    assert set(rig.send) == {"ESC_01", "SRS_01", "Klemmen_Status_01", "NM_BCM"}
    assert rig.ignition == {"signal": "ZAS_Kl_15", "on": 1, "off": 0}
    assert rig.e2e["ESC_01"].counter == "ESC_01_BZ" and rig.e2e["ESC_01"].crc == "ESC_01_CRC"
    assert rig.e2e["SRS_01"].counter == "SRS_AliveCounter" and rig.e2e["SRS_01"].crc is None
    # Start values meaning "not available" are replaced by the quiet ones.
    assert rig.rest["ESC_ABS_Lamp"] == "Off"
    assert rig.rest["SRS_Belt_Driver"] == "Buckled"
    assert rig.rest["ESC_Speed"] == 0
    assert rig.rest["ZAS_Kl_15"] == 1


def test_names_that_only_look_like_the_ignition_are_not_it(tmp_path):
    path = tmp_path / "x.dbc"
    path.write_text(DBC.replace(" SG_ ZAS_Kl_15 : 0|1@1+ (1,0) [0|1] \"\" KOMBI,ESC\n", ""))
    rig = Rig.detect(path)
    assert rig.ignition is None, rig.ignition       # neither Nachtdesign nor NM's KL15


def test_a_rig_file_corrects_what_was_worked_out(dbc, tmp_path):
    f = tmp_path / "rig.json"
    f.write_text(json.dumps({
        "dbc": dbc.name, "cluster_node": "KOMBI",
        "e2e_default": {"algorithm": "crc8_h2f", "data_id_bytes": "low"},
        "e2e": {"SRS_01": None},
        "rest": {"ESC_Speed": 12.5},
        "exclude": ["DI_Display_Nachtdesign"],
        "names": {"ESC_ABS_Lamp": "ABS"}}))
    rig = Rig.load(f)
    assert rig.e2e["ESC_01"].algorithm == "crc8_h2f" and "SRS_01" not in rig.e2e
    assert rig.rest["ESC_Speed"] == 12.5 and rig.exclude == ["DI_Display_Nachtdesign"]
    again = json.loads(json.dumps(rig.to_dict()))
    assert again["cluster_node"] == "KOMBI" and again["names"] == {"ESC_ABS_Lamp": "ABS"}


def test_crc_check_values():
    assert ALGORITHMS["crc8_sae_j1850"](b"123456789") == 0x4B
    assert ALGORITHMS["crc8_h2f"](b"123456789") == 0xDF
    assert crc8(b"\x00", 0x07, 0x00, 0x00) == 0x00


def frames(listen, seconds):
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        m = listen.recv(timeout=0.05)
        if m is not None:
            out.append(m)
    return out


def test_the_tester_sends_each_message_at_its_cycle_with_counter_and_crc(dbc):
    channel = f"rig-{time.monotonic_ns()}"
    listen = can.Bus(interface="virtual", channel=channel)
    drv = CanDriver(Rig.detect(dbc), bus_obj=can.Bus(interface="virtual", channel=channel))
    db = cantools.database.load_file(str(dbc))
    try:
        got = frames(listen, 1.0)
        by_id = {}
        for m in got:
            by_id.setdefault(m.arbitration_id, []).append(m)
        assert 1500 not in by_id and 1793 not in by_id        # its own, and diagnostics
        assert 35 <= len(by_id[160]) <= 60, len(by_id[160])   # 20 ms
        assert 7 <= len(by_id[800]) <= 13, len(by_id[800])    # 100 ms
        esc = db.get_message_by_name("ESC_01")
        counters = []
        for m in by_id[160]:
            d = bytes(m.data)
            body = bytes([160, 0]) + d[1:]
            assert d[0] == ALGORITHMS["crc8_sae_j1850"](body)
            counters.append(esc.decode(d, decode_choices=False)["ESC_01_BZ"])
        steps = {(b - a) % 15 for a, b in zip(counters, counters[1:])}
        assert steps == {1}, steps                             # 0..14, every frame
        drv.apply({"ESC_ABS_Lamp": "On", "ESC_Speed": 88.0})
        last = [m for m in frames(listen, 0.2) if m.arbitration_id == 160][-1]
        d = esc.decode(bytes(last.data))
        assert str(d["ESC_ABS_Lamp"]) == "On" and d["ESC_Speed"] == pytest.approx(88.0)
        drv.ignition(False)
        last = [m for m in frames(listen, 0.3) if m.arbitration_id == 912][-1]
        assert db.decode_message(912, bytes(last.data))["ZAS_Kl_15"] == 0
        drv.quiet(True)
        time.sleep(0.1)
        frames(listen, 0.05)
        assert frames(listen, 0.3) == []                       # the bus asleep
        with pytest.raises(CanError, match="not a signal the tester sends"):
            drv.check({"KOMBI_Odometer": 1})
    finally:
        drv.close()
        listen.shutdown()


def test_parse_bus():
    assert parse_bus("udp_multicast") == {"interface": "udp_multicast", "channel": "239.74.163.2"}
    assert parse_bus("socketcan:vcan0") == {"interface": "socketcan", "channel": "vcan0"}
    assert parse_bus("pcan:PCAN_USBBUS1@500000")["bitrate"] == 500000


def test_the_command_line_needs_only_the_dbc(dbc, tmp_path):
    import argparse

    from layoutval.cli import _can_setup

    args = argparse.Namespace(can=f"virtual:cli-{time.monotonic_ns()}", dbc=str(dbc), rig=None,
                              teach_plan=None, out=str(tmp_path / "out"))
    driver, plan, discovery = _can_setup(args)
    try:
        assert plan is None and discovery is not None
        assert {sp.signal for sp in discovery.signals} >= {"ESC_ABS_Lamp", "SRS_Belt_Driver"}
        written = json.loads((tmp_path / "out" / "rig-detected.json").read_text())
        assert written["cluster_node"] == "KOMBI" and written["ignition"]["signal"] == "ZAS_Kl_15"
        # The rig file it wrote, corrected and given back, is taken as it says.
        written["names"] = {"ESC_ABS_Lamp": "ABS"}
        rig_file = tmp_path / "rig.json"
        rig_file.write_text(json.dumps(written))
    finally:
        driver.close()
    args.rig, args.can = str(rig_file), f"virtual:cli-{time.monotonic_ns()}"
    driver, _, discovery = _can_setup(args)
    try:
        assert next(sp.name for sp in discovery.signals if sp.signal == "ESC_ABS_Lamp") == "ABS"
    finally:
        driver.close()
    with pytest.raises(SystemExit, match="--dbc"):
        _can_setup(argparse.Namespace(can="virtual:x", dbc=None, rig=None, teach_plan=None,
                                      out=str(tmp_path)))
