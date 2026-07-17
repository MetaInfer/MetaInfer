import tempfile
from pathlib import Path
import unittest

from metainfer.tasks.gen_infer_framework_cpp.orchestrator.hardware_discovery import (
    _visible_device_count,
    configure_assigned_devices,
    parse_lspci,
    parse_nvidia_smi,
    parse_rocm_smi,
    parse_rocminfo,
    validate_hardware_selection,
    write_hardware_profile,
)


LSPCI_Z200 = """\
0000:04:00.0 Display controller [0380]: Chengdu Haiguang IC Design Co., Ltd. Z200SM 80 [1d94:6210] (rev 01)
0000:26:00.0 Display controller [0380]: Chengdu Haiguang IC Design Co., Ltd. Z200SM 80 [1d94:6210] (rev 01)
0000:43:00.0 Display controller [0380]: Chengdu Haiguang IC Design Co., Ltd. Z200SM 80 [1d94:6210] (rev 01)
0000:63:00.0 Display controller [0380]: Chengdu Haiguang IC Design Co., Ltd. Z200SM 80 [1d94:6210] (rev 01)
"""

ROCM_SMI_Z100_ALIAS = """\
DCU[0]          : Card Series:           Z100SM
DCU[1]          : Card Series:           Z100SM
DCU[2]          : Card Series:           Z100SM
DCU[3]          : Card Series:           Z100SM
Driver Version: 6.2.26
DCU[0]          : vram Total Memory (MiB): 16368
DCU[1]          : vram Total Memory (MiB): 16368
DCU[2]          : vram Total Memory (MiB): 16368
DCU[3]          : vram Total Memory (MiB): 16368
"""

ROCMINFO = """\
  Agent 1
  *******
    Name:                    AMD Ryzen CPU
    Device Type:             CPU
  Agent 2
  *******
    Name:                    gfx906
    Uuid:                    GPU-abc
    Marketing Name:          Hygon DCU
    Vendor Name:             Chengdu Haiguang IC Design Co., Ltd.
    Device Type:             GPU
"""


class HardwareParserTest(unittest.TestCase):
    def test_z200_pci_and_z100_smi_alias_are_preserved(self):
        pci = parse_lspci(LSPCI_Z200)
        smi = parse_rocm_smi(ROCM_SMI_Z100_ALIAS)

        self.assertEqual(len(pci), 4)
        self.assertEqual(pci[0]["product_name"], "Chengdu Haiguang IC Design Co., Ltd. Z200SM 80")
        self.assertEqual(pci[0]["vendor_id"], "1d94")
        self.assertEqual(pci[0]["vendor_family"], "hygon")
        self.assertEqual(smi["devices"][0]["product_name"], "Z100SM")
        self.assertEqual(smi["devices"][0]["vram_total_mib"], 16368)
        self.assertEqual(smi["driver_version"], "6.2.26")

        profile = {
            "requested": {
                "target_hardware": "Hygon Z200SM",
                "accelerator_backend": "Hygon DTK / HIP",
            },
            "detected": {
                "vendor_family": "hygon",
                "device_count": 4,
                "pci_devices": pci,
                "smi_devices": smi["devices"],
                "nvidia_devices": [],
            },
            "permissions": {
                "kfd": {"exists": True, "readable": True, "writable": True},
                "render_nodes": [
                    {"path": f"/dev/dri/renderD{128 + i}", "readable": True, "writable": True}
                    for i in range(4)
                ],
            },
        }
        validation = validate_hardware_selection(profile)
        self.assertTrue(validation["runnable"])
        self.assertEqual(validation["status"], "compatible_with_warnings")
        self.assertFalse(validation["blockers"])
        self.assertTrue(any("PCI and SMI" in item for item in validation["warnings"]))

    def test_rocminfo_architecture_is_separate_from_marketing_name(self):
        parsed = parse_rocminfo(ROCMINFO)
        self.assertEqual(parsed["architectures"], ["gfx906"])
        self.assertEqual(parsed["agents"][0]["marketing_name"], "Hygon DCU")

    def test_vendor_or_model_mismatch_blocks(self):
        profile = {
            "requested": {
                "target_hardware": "NVIDIA H100",
                "accelerator_backend": "CUDA",
            },
            "detected": {
                "vendor_family": "hygon",
                "device_count": 1,
                "pci_devices": parse_lspci(LSPCI_Z200)[:1],
                "smi_devices": [],
                "nvidia_devices": [],
            },
            "permissions": {"kfd": {}, "render_nodes": []},
        }
        validation = validate_hardware_selection(profile)
        self.assertFalse(validation["runnable"])
        self.assertEqual(validation["status"], "mismatch")
        self.assertGreaterEqual(len(validation["blockers"]), 1)

    def test_missing_dtk_build_capabilities_are_blockers(self):
        profile = {
            "requested": {
                "target_hardware": "Hygon Z200SM",
                "accelerator_backend": "Hygon DTK / HIP",
            },
            "detected": {
                "vendor_family": "hygon",
                "device_count": 4,
                "pci_devices": parse_lspci(LSPCI_Z200),
                "smi_devices": [],
                "nvidia_devices": [],
                "hip_architectures": [],
            },
            "permissions": {
                "kfd": {"exists": True, "readable": True, "writable": True},
                "render_nodes": [
                    {"path": "/dev/dri/renderD128", "readable": True, "writable": True},
                ],
            },
            "toolchain": {
                "cmake": {"available": True},
                "hipcc": {"available": False},
            },
        }
        validation = validate_hardware_selection(profile)
        self.assertFalse(validation["runnable"])
        self.assertTrue(any("hipcc" in item for item in validation["blockers"]))
        self.assertTrue(any("architecture" in item for item in validation["blockers"]))

    def test_assigned_devices_are_validated_and_backend_specific(self):
        env = {"HIP_VISIBLE_DEVICES": "7"}
        applied = configure_assigned_devices(
            {
                "assigned_devices": "0, 2,3",
                "target_hardware": "Hygon Z200SM",
                "accelerator_backend": "Hygon DTK / HIP",
            },
            env,
        )
        self.assertEqual(applied["HIP_VISIBLE_DEVICES"], "0,2,3")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "0,2,3")
        localized = configure_assigned_devices(
            {
                "assigned_devices": "0\uff0c 1\u30012",
                "target_hardware": "Hygon Z200SM",
                "accelerator_backend": "Hygon DTK / HIP",
            },
            {},
        )
        self.assertEqual(localized["HIP_VISIBLE_DEVICES"], "0,1,2")
        with self.assertRaises(ValueError):
            configure_assigned_devices({"assigned_devices": "0; rm -rf /"}, {})
        self.assertEqual(
            _visible_device_count({"HIP_VISIBLE_DEVICES": "0,2"}, "hygon"),
            2,
        )
        self.assertEqual(
            _visible_device_count({"CUDA_VISIBLE_DEVICES": "-1"}, "nvidia"),
            0,
        )

    def test_other_parsers_and_atomic_write(self):
        nvidia = parse_nvidia_smi("0, NVIDIA H100, 81559, 0000:01:00.0, 550.1\n")
        self.assertEqual(nvidia[0]["vram_total_mib"], 81559)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware_profile.json"
            write_hardware_profile(path, {"schema_version": 1})
            self.assertIn('"schema_version": 1', path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
