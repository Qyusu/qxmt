from types import SimpleNamespace

from qxmt.devices import amazon


def test_build_amazon_braket_local_backends_from_entry_points(mocker) -> None:
    entry_points = [
        SimpleNamespace(name="braket_dm"),
        SimpleNamespace(name="default"),
        SimpleNamespace(name="braket_sv"),
    ]
    mocker.patch.object(amazon, "entry_points", return_value=entry_points)

    backends = amazon._build_amazon_braket_local_backends()

    assert backends == ["braket_dm", "braket_sv", "default"]
    amazon.entry_points.assert_called_once_with(group="braket.simulators")


def test_build_amazon_braket_local_backends_falls_back_when_entry_points_are_empty(
    mocker,
) -> None:
    mocker.patch.object(amazon, "entry_points", return_value=[])

    backends = amazon._build_amazon_braket_local_backends()

    assert backends == amazon.FALLBACK_AMAZON_BRAKET_LOCAL_BACKENDS


def test_build_amazon_backend_types_from_braket_devices(mocker) -> None:
    ionq_devices = SimpleNamespace(
        __members__={
            "_Aria1": "aria1",
            "Forte1": "forte1",
            "ForteEnterprise1": "forte_enterprise1",
            "ForteEnterprise2": "forte_enterprise2",
        }
    )
    rigetti_devices = SimpleNamespace(
        __members__={
            "_Ankaa2": "ankaa2",
            "Ankaa3": "ankaa3",
        }
    )
    private_only_devices = SimpleNamespace(
        __members__={
            "_PrivateDevice": "private_device",
        }
    )
    devices = type(
        "Devices",
        (),
        {
            "IonQ": ionq_devices,
            "Rigetti": rigetti_devices,
            "PrivateOnly": private_only_devices,
        },
    )
    mocker.patch.object(amazon, "Devices", devices)

    backend_types = amazon._build_amazon_backend_types()

    assert backend_types["ionq_aria1"] == "aria1"
    assert backend_types["ionq_forte1"] == "forte1"
    assert backend_types["ionq_forte_enterprise1"] == "forte_enterprise1"
    assert backend_types["ionq_forte_enterprise2"] == "forte_enterprise2"
    assert backend_types["ionq"] == "forte_enterprise2"
    assert backend_types["rigetti_ankaa2"] == "ankaa2"
    assert backend_types["rigetti_ankaa3"] == "ankaa3"
    assert backend_types["rigetti"] == "ankaa3"
    assert "privateonly" not in backend_types


def test_build_amazon_backend_types_adds_legacy_amazon_simulator_aliases(
    mocker,
) -> None:
    amazon_devices = SimpleNamespace(
        __members__={
            "SV1": "sv1",
            "TN1": "tn1",
            "DM1": "dm1",
        }
    )
    devices = type("Devices", (), {"Amazon": amazon_devices})
    mocker.patch.object(amazon, "Devices", devices)

    backend_types = amazon._build_amazon_backend_types()

    assert backend_types["amazon_sv1"] == "sv1"
    assert backend_types["amazon_tn1"] == "tn1"
    assert backend_types["amazon_dm1"] == "dm1"
    assert backend_types["sv1"] == "sv1"
    assert backend_types["tn1"] == "tn1"
    assert backend_types["dm1"] == "dm1"
    assert backend_types["amazon"] == "dm1"
