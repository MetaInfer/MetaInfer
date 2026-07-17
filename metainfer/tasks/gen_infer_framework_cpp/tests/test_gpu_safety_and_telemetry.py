from unittest.mock import patch
import unittest

from metainfer.tasks.gen_infer_framework_cpp.orchestrator import gpu_preflight
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.gpu import (
    GpuSamples,
    GpuTelemetry,
    _parse_nvidia_samples,
    _parse_rocm_names,
    _parse_rocm_samples,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.perf import (
    _expected_device_indices,
    _preferred_telemetry_backend,
)


ROCM_SAMPLE = """\
DCU[0] : Card Series: Z100SM
DCU[1] : Card Series: Z100SM
DCU[0] : DCU use (%): 75
DCU[1] : DCU use (%): 50
DCU[0] : GPU Memory Allocated (VRAM%): 60
DCU[1] : GPU Memory Allocated (VRAM%): 55
DCU[0] : vram Total Used Memory (MiB): 9000
DCU[1] : vram Total Used Memory (MiB): 8000
"""


class SharedHostGpuSafetyTest(unittest.TestCase):
    def test_default_preflight_observes_but_never_kills(self):
        occupant = gpu_preflight.GpuOccupant(
            pid=4242,
            vram_mib=4096,
            command="python",
            source="nvidia-smi",
        )
        with (
            patch.object(gpu_preflight, "_find_nvidia_smi", return_value="nvidia-smi"),
            patch.object(gpu_preflight, "_find_rocm_smi", return_value=None),
            patch.object(gpu_preflight, "_list_nvidia_occupants", return_value=[occupant]),
            patch.object(gpu_preflight, "_kill_one") as kill_one,
        ):
            result = gpu_preflight.preflight_gpu(label="test")

        kill_one.assert_not_called()
        self.assertFalse(result.termination_enabled)
        self.assertEqual(result.killed, [])
        self.assertEqual(result.unowned[0]["pid"], 4242)

    def test_rocm_telemetry_parsers_and_active_devices(self):
        self.assertEqual(_parse_rocm_names(ROCM_SAMPLE), ["Z100SM", "Z100SM"])
        devices = _parse_rocm_samples(ROCM_SAMPLE)
        self.assertEqual(devices[0]["utilization_gpu"], 75.0)
        self.assertEqual(devices[1]["memory_used_mib"], 8000.0)

        samples = GpuSamples(
            samples=[{"utilization_gpu": 62.5, "memory_used_mib": 8500.0}],
            gpu_name="Z100SM",
            device_count=2,
            backend="rocm-smi",
            device_peaks=devices,
        )
        aggregate = samples.aggregate()
        self.assertTrue(aggregate["collected"])
        self.assertEqual(aggregate["active_device_count"], 2)
        self.assertEqual(aggregate["backend"], "rocm-smi")

    def test_nvidia_parser_keeps_per_device_values(self):
        parsed = _parse_nvidia_samples(
            "0, 80, 40, 12000, 300\n1, 0, 0, 2, [N/A]\n"
        )
        self.assertEqual(parsed[0]["memory_used_mib"], 12000.0)
        self.assertEqual(parsed[1]["power_draw_w"], 0.0)

    def test_active_device_mean_is_not_divided_by_idle_visible_devices(self):
        telemetry = GpuTelemetry()
        telemetry._record_devices({
            0: {"utilization_gpu": 80.0, "memory_used_mib": 4000.0},
            1: {"utilization_gpu": 0.0, "memory_used_mib": 2.0},
            2: {"utilization_gpu": 0.0, "memory_used_mib": 2.0},
            3: {"utilization_gpu": 0.0, "memory_used_mib": 2.0},
        })
        aggregate = telemetry.aggregate()

        self.assertEqual(aggregate["utilization_gpu_mean"], 20.0)
        self.assertEqual(aggregate["active_device_count"], 1)
        self.assertEqual(
            aggregate["active_device_utilization_gpu_mean"], 80.0,
        )
        device0 = next(
            item for item in aggregate["per_device_stats"]
            if item["index"] == 0
        )
        self.assertEqual(device0["utilization_gpu_mean"], 80.0)

    def test_preferred_backend_does_not_mix_management_tools(self):
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.gpu._find_rocm_smi",
                return_value="/opt/dtk/bin/rocm-smi",
            ),
        ):
            rocm = GpuTelemetry(preferred_backend="rocm-smi")
            nvidia = GpuTelemetry(preferred_backend="nvidia-smi")
        self.assertIsNone(rocm._nvidia_smi)
        self.assertEqual(rocm._rocm_smi, "/opt/dtk/bin/rocm-smi")
        self.assertEqual(nvidia._nvidia_smi, "/usr/bin/nvidia-smi")
        self.assertIsNone(nvidia._rocm_smi)
        req = {
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
            "assigned_devices": "0,2",
        }
        self.assertEqual(_preferred_telemetry_backend(req), "rocm-smi")
        self.assertEqual(_expected_device_indices(req), [0, 2])


if __name__ == "__main__":
    unittest.main()
