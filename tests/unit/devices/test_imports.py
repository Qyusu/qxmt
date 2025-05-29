from qxmt.devices import __all__

EXPECTED_ALL = [
    "AmazonBraketDevice",
    "BaseDevice",
    "DeviceBuilder",
    "IBMQDevice",
    "PennyLaneDevice",
    "get_number_of_qubits",
    "get_platform_from_device",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
