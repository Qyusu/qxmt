from typing import Any

from braket.devices import Devices

AMAZON_PROVIDER_NAME = "Amazon_Braket"
AMAZON_BRAKET_LOCAL_DEVICES = ["braket.local.qubit"]
AMAZON_BRAKET_LOCAL_BACKENDS = ["default", "braket_sv", "braket_dm", "braket_ahs"]
AMAZON_BRAKET_REMOTE_DEVICES = ["braket.aws.qubit"]
AMAZON_BRAKET_DEVICES = AMAZON_BRAKET_LOCAL_DEVICES + AMAZON_BRAKET_REMOTE_DEVICES
AMAZON_BRAKET_SIMULATOR_BACKENDS = AMAZON_BRAKET_LOCAL_BACKENDS + ["sv1", "dm1", "tn1"]


def _get_braket_device(provider_name: str, device_name: str):
    provider = getattr(Devices, provider_name, None)
    if provider is None:
        return None
    return provider.__members__.get(device_name)


def _get_first_available_braket_device(*candidates: tuple[str, str]):
    for provider_name, device_name in candidates:
        device = _get_braket_device(provider_name, device_name)
        if device is not None:
            return device
    return None


def _build_amazon_backend_types() -> dict[str, Any]:
    backend_candidates = {
        "sv1": (("Amazon", "SV1"),),
        "dm1": (("Amazon", "DM1"),),
        "tn1": (("Amazon", "TN1"),),
        "ionq": (
            ("IonQ", "ForteEnterprise1"),
            ("IonQ", "Forte1"),
            ("IonQ", "Aria1"),
        ),
        "ionq_forte1": (("IonQ", "Forte1"),),
        "ionq_forte_enterprise1": (("IonQ", "ForteEnterprise1"),),
        "iqm": (
            ("IQM", "Garnet"),
            ("IQM", "Emerald"),
        ),
        "iqm_garnet": (("IQM", "Garnet"),),
        "iqm_emerald": (("IQM", "Emerald"),),
        "quera": (("QuEra", "Aquila"),),
        "quera_aquila": (("QuEra", "Aquila"),),
        "rigetti": (
            ("Rigetti", "Ankaa3"),
            ("Rigetti", "_Ankaa2"),
            ("Rigetti", "Cepheus1108Q"),
        ),
        "rigetti_ankaa3": (("Rigetti", "Ankaa3"),),
        "rigetti_cepheus1108q": (("Rigetti", "Cepheus1108Q"),),
    }

    return {
        backend_name: device
        for backend_name, candidates in backend_candidates.items()
        if (device := _get_first_available_braket_device(*candidates)) is not None
    }


AmazonBackendType = _build_amazon_backend_types()
