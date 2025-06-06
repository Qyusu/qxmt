from enum import Enum

from braket.devices import Devices

AMAZON_PROVIDER_NAME = "Amazon_Braket"
AMAZON_BRAKET_LOCAL_DEVICES = ["braket.local.qubit"]
AMAZON_BRAKET_LOCAL_BACKENDS = ["default", "braket_sv", "braket_dm", "braket_ahs"]
AMAZON_BRAKET_REMOTE_DEVICES = ["braket.aws.qubit"]
AMAZON_BRAKET_DEVICES = AMAZON_BRAKET_LOCAL_DEVICES + AMAZON_BRAKET_REMOTE_DEVICES
AMAZON_BRAKET_SIMULATOR_BACKENDS = AMAZON_BRAKET_LOCAL_BACKENDS + ["sv1", "dm1", "tn1"]


class AmazonBackendType(Enum):
    sv1 = Devices.Amazon.SV1
    dm1 = Devices.Amazon.DM1
    tn1 = Devices.Amazon.TN1
    ionq = Devices.IonQ.Aria1  # default IonQ device
    ionq_aria1 = Devices.IonQ.Aria1
    ionq_aria2 = Devices.IonQ.Aria2
    ionq_forte1 = Devices.IonQ.Forte1
    iqm = Devices.IQM.Garnet  # default IQM device
    iqm_garnet = Devices.IQM.Garnet
    quera = Devices.QuEra.Aquila  # default QuEra device
    quera_aquila = Devices.QuEra.Aquila
    rigetti = Devices.Rigetti.Ankaa3  # default Rigetti device
    rigetti_ankaa3 = Devices.Rigetti.Ankaa3
